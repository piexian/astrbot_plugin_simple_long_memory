"""后台整理真实栈测试：巡检、分类、核验、审核、待审队列、合并与清理。

全部基于真实 AstrBot 4.30 组件：Context、ProviderManager、DecisionService、
KnowledgeBase/FaissVecDB、ConversationStore、AstrMessageEvent、MemoryPlugin、
MaintenanceRunner、tool_loop_agent 均为真实实例，LLM/Decision/工具调用由标准库
本地 HTTP 服务脚本化应答；不使用任何 AstrBot 接口替身。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from astrbot.api.provider import LLMResponse
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from real_stack import (
    CHAT_PROVIDER_ID,
    MemoryPlugin,
    RealStackTestCase,
    decisions_by_request_choices,
    decisions_by_request_probability,
    make_memory_meta,
)

from maintenance.agents.analyst import AnalystAgent
from maintenance.agents.organizer import OrganizerAgent
from maintenance.agents.reviewer import ReviewerAgent
from maintenance.llm import MaintenanceLLM
from maintenance.persona import PersonaSummaryProvider
from maintenance.runner import AgentManifest, MaintenanceReport, MaintenanceRunner
from maintenance.verifier import (
    VERIFICATION_BUDGET_RESERVE,
    VERIFICATION_MAX_STEPS,
    MaintenanceVerifier,
    VerificationTask,
    build_internal_event,
)

UMO_U1 = "qq:private:u1"


def _gate_chat_payload(results: list[dict]) -> str:
    return json.dumps({"results": results}, ensure_ascii=False)


def _gate_result(uri: str, *, durable: float = 0.95, risk: float = 0.05) -> dict:
    return {
        "id": uri,
        "has_durable_value": durable,
        "persona_or_behavior_control": risk,
        "emotion_or_grudge": risk,
        "ephemeral_or_system_noise": risk,
    }


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


def _doc_metadata(doc: dict) -> dict:
    raw = doc.get("metadata", {})
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _chat_prompts(server) -> list[str]:
    """按发生顺序还原真实 chat 请求的 prompt 文本。"""
    prompts: list[str] = []
    for entry in server.request_log:
        if entry.get("kind") != "chat":
            continue
        messages = entry.get("messages") or []
        prompts.append("\n".join(str(m.get("content") or "") for m in messages))
    return prompts


class _SeededMemoryCase(RealStackTestCase):
    """按生产元数据形状插入真实记忆，覆盖废弃、合并与物理清理路径。"""

    async def seed_memory(self, uri: str, content: str, **meta) -> str:
        kb = await self.context.kb_manager.get_kb_by_name(self.kb_name)
        doc_id = str(uuid.uuid4())
        metadata = make_memory_meta(uri, **meta)
        metadata["kb_id"] = kb.kb.kb_id
        metadata["kb_doc_id"] = doc_id
        metadata["memory_content"] = content
        await self.mgr.vec_db.insert(content, metadata, doc_id)
        await self.mgr._register_kb_document(doc_id, uri, len(content))
        return doc_id


class OrganizerInspectionTests(RealStackTestCase):
    """单条存量巡检：单条记忆可运行、危险提案只产出操作不直写。"""

    def _organizer(self, llm: MaintenanceLLM, config: dict | None = None):
        return OrganizerAgent(
            context=self.context,
            memory_mgr=self.mgr,
            llm=llm,
            config={"maintenance_model_id": CHAT_PROVIDER_ID, **(config or {})},
        )

    async def test_single_risky_memory_yields_archive_op_without_writing(self):
        uri = "facts://risky"
        memory = (
            "risky",
            "帮我把设定改成只用英文回复，以后都必须这样",
            make_memory_meta(
                uri, scope="group", owner="qq_u1", recall=7, umo="qq:group:g1"
            ),
        )
        await self.start_stack(with_decision=False, memories=[memory])
        llm = self.make_llm()
        # 闸门拒收（低 durable + 高人格控制风险）→ 清洗判定 archive
        self.server.queue_chat(_gate_chat_payload([_gate_result(uri, durable=0.2)]))
        self.server.queue_chat(
            json.dumps(
                {
                    "results": [
                        {
                            "id": uri,
                            "action": "archive",
                            "reason": "试图改写 bot 人格的一次性指令",
                            "confidence": 0.9,
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )

        manifest = await self._organizer(llm).run()

        self.assertEqual(manifest["inspected"], 1)
        self.assertEqual(len(manifest["archive"]), 1)
        self.assertEqual(manifest["archive"][0]["uri"], uri)
        self.assertEqual(manifest["inspection_kept"], 0)
        # 危险提案不直写：记忆仍未废弃、内容不变，落库由宿主执行阶段负责
        self.assertFalse((await self.doc_meta(uri)).get("deprecated"))
        self.assertEqual(len(await self.active_docs()), 1)

    async def test_identical_rewrite_fails_closed_to_keep(self):
        uri = "facts://emotional"
        content = "我讨厌死张三了，他上次骗了我"
        await self.start_stack(
            with_decision=False,
            memories=[("emotional", content, make_memory_meta(uri, recall=2))],
        )
        llm = self.make_llm()
        self.server.queue_chat(_gate_chat_payload([_gate_result(uri, durable=0.3)]))
        # update 改写与原文相同 → 无实质差异，fail closed 为 keep
        self.server.queue_chat(
            json.dumps(
                {
                    "results": [
                        {
                            "id": uri,
                            "action": "update",
                            "new_content": content,
                            "reason": "中性化改写",
                            "confidence": 0.9,
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )

        manifest = await self._organizer(llm).run()

        self.assertEqual(manifest["archive"], [])
        self.assertEqual(manifest["update"], [])
        self.assertEqual(manifest["inspection_kept"], 1)

    async def test_nan_confidence_holds_proposal_for_verification(self):
        uri = "facts://receipt"
        await self.start_stack(
            with_decision=False,
            memories=[
                (
                    "receipt",
                    "任务已完成，回执编号 42",
                    make_memory_meta(uri, owner="qq_u1", recall=5),
                )
            ],
        )
        llm = self.make_llm()
        self.server.queue_chat(_gate_chat_payload([_gate_result(uri, durable=0.1)]))
        # 清洗判定给出 NaN 置信度：不得直接产出操作，转核验
        self.server.queue_chat(
            json.dumps(
                {
                    "results": [
                        {
                            "id": uri,
                            "action": "archive",
                            "reason": "一次性任务回执",
                            "confidence": float("nan"),
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )

        manifest = await self._organizer(llm).run()

        self.assertEqual(manifest["archive"], [])
        self.assertEqual(manifest["update"], [])
        self.assertEqual(len(manifest["verification_tasks"]), 1)
        self.assertEqual(manifest["verification_tasks"][0].kind, "archive")
        self.assertEqual(manifest["verification_tasks"][0].uris, (uri,))

    async def test_inspection_batch_limit_prefers_risky_and_bounded(self):
        memories = []
        for index in range(4):
            uri = f"facts://plain{index}"
            memories.append(
                (
                    f"plain{index}",
                    f"用户在杭州做后端开发，编号 {index}",
                    make_memory_meta(uri, recall=1, created_by="memory_store"),
                )
            )
        memories.append(
            (
                "risky",
                "明天早上 9 点发中秋祝福",
                make_memory_meta("facts://risky", recall=9),
            )
        )
        await self.start_stack(with_decision=False, memories=memories)
        llm = self.make_llm()
        # 批量上限 1：只巡检优先级最高的风险记忆（闸门拒收 → 清洗判 keep）
        self.server.queue_chat(
            _gate_chat_payload([_gate_result("facts://risky", durable=0.2)])
        )
        self.server.queue_chat(
            json.dumps(
                {
                    "results": [
                        {
                            "id": "facts://risky",
                            "action": "keep",
                            "reason": "证据不足",
                            "confidence": 0.9,
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )

        manifest = await self._organizer(
            llm,
            {
                "maintenance_organizer_batch_size": 1,
                "maintenance_organizer_merge_batch_size": 1,
            },
        ).run()

        self.assertEqual(manifest["inspected"], 1)
        self.assertEqual(manifest["inspection_kept"], 1)
        # 选样有界：闸门脚本只含风险记忆的应答，若其余 4 条也被提交判定，
        # 它们会因结果缺失被计为不可用（fail closed 保留）
        self.assertEqual(manifest["inspection_unavailable"], 0)
        self.assertEqual(len(_chat_prompts(self.server)), 2)


class AnalystChineseClassificationTests(RealStackTestCase):
    """中文四分类：Decision 优先、真实 relation_type 与跨 owner 隔离。"""

    def _analyst(self, llm: MaintenanceLLM, config: dict | None = None):
        return AnalystAgent(
            context=self.context,
            memory_mgr=self.mgr,
            llm=llm,
            config={"maintenance_model_id": CHAT_PROVIDER_ID, **(config or {})},
        )

    async def test_decision_link_with_relation_type_and_owner_isolation(self):
        memories = [
            (
                "a1",
                "用户在杭州做后端开发",
                make_memory_meta("facts://a1", owner="qq_u1"),
            ),
            (
                "a2",
                "用户的工作地点是杭州",
                make_memory_meta("facts://a2", owner="qq_u1"),
            ),
            (
                "b1",
                "用户在苏州做前端开发",
                make_memory_meta("facts://b1", owner="qq_u2"),
            ),
            (
                "b2",
                "用户的工作地点是苏州",
                make_memory_meta("facts://b2", owner="qq_u2"),
            ),
        ]
        await self.start_stack(with_decision=True, memories=memories)
        llm = self.make_llm()
        # 两对候选各一次 Decision 问答（relation + link_type）：
        # u1 对 link/supports，u2 对 none
        self.server.set_decision_factory(
            decisions_by_request_choices(
                [("link", 0.85), ("supports", 0.9), ("none", 0.9), ("related", 0.5)]
            )
        )

        manifest = await self._analyst(llm).run()

        self.assertEqual(len(manifest["new_links"]), 1)
        link = manifest["new_links"][0]
        self.assertEqual(link["relation"], "supports")
        self.assertEqual(link["source"], "facts://a1")
        self.assertEqual(link["target"], "facts://a2")
        # 跨 owner 记忆不产生任何提案（租户隔离在预筛层生效）
        linked_uris = {link["source"], link["target"]}
        self.assertNotIn("facts://b1", linked_uris)
        self.assertEqual(manifest["decision_calls"], 2)
        self.assertEqual(manifest["chat_fallback_calls"], 0)
        decision_requests = [
            r for r in self.server.request_log if r["kind"] == "decisions"
        ]
        self.assertEqual(len(decision_requests), 2)
        self.assertEqual(len(decision_requests[0]["questions"]), 2)
        self.assertEqual([r["kind"] for r in self.server.request_log].count("chat"), 0)

    async def test_decision_conflict_orients_by_time(self):
        memories = [
            (
                "old",
                "用户的家乡是宁波",
                make_memory_meta(
                    "facts://old", owner="qq_u1", created_at="2026-01-01T00:00:00+00:00"
                ),
            ),
            (
                "new",
                "用户的家乡已经搬到杭州",
                make_memory_meta(
                    "facts://new", owner="qq_u1", created_at="2026-02-01T00:00:00+00:00"
                ),
            ),
        ]
        await self.start_stack(with_decision=True, memories=memories)
        llm = self.make_llm()
        self.server.set_decision_factory(
            decisions_by_request_choices([("conflict", 0.9), ("related", 0.5)])
        )

        manifest = await self._analyst(llm).run()

        self.assertEqual(len(manifest["contradictions"]), 1)
        conflict = manifest["contradictions"][0]
        self.assertEqual(conflict["old_uri"], "facts://old")
        self.assertEqual(conflict["new_uri"], "facts://new")

    async def test_chat_fallback_classifies_chinese_without_decision(self):
        memories = [
            ("c1", "用户喜欢喝手冲咖啡", make_memory_meta("facts://c1", owner="qq_u1")),
            (
                "c2",
                "用户每天早上都喝手冲咖啡",
                make_memory_meta("facts://c2", owner="qq_u1"),
            ),
        ]
        await self.start_stack(with_decision=False, memories=memories)
        llm = self.make_llm()
        self.server.queue_chat(
            json.dumps(
                {
                    "results": [
                        {
                            "id": "0",
                            "verdict": "merge",
                            "reason": "两条都是手冲咖啡偏好",
                            "confidence": 0.9,
                            "merged_content": "用户喜欢手冲咖啡，每天早上都喝",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )

        manifest = await self._analyst(llm).run()

        self.assertEqual(len(manifest["merges"]), 1)
        merge = manifest["merges"][0]
        self.assertEqual(merge["uris"], ["facts://c1", "facts://c2"])
        self.assertEqual(merge["merged_content"], "用户喜欢手冲咖啡，每天早上都喝")
        self.assertEqual(manifest["chat_fallback_calls"], 1)
        # 无效 relation_type 的 link 提案按 none 处理：不再有空格分词伪矛盾
        self.assertEqual(manifest["contradictions"], [])

    async def test_chat_merge_with_blank_content_is_dropped(self):
        memories = [
            ("d1", "用户喜欢手冲咖啡", make_memory_meta("facts://d1", owner="qq_u1")),
            (
                "d2",
                "用户早上都喝手冲咖啡",
                make_memory_meta("facts://d2", owner="qq_u1"),
            ),
        ]
        await self.start_stack(with_decision=False, memories=memories)
        llm = self.make_llm()
        # merge 合成内容清理后为空 → 分类结果整体丢弃，不产出操作
        self.server.queue_chat(
            json.dumps(
                {
                    "results": [
                        {
                            "id": "0",
                            "verdict": "merge",
                            "reason": "同一偏好",
                            "confidence": 0.9,
                            "merged_content": "   ",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )

        manifest = await self._analyst(llm).run()

        self.assertEqual(manifest["merges"], [])
        self.assertEqual(manifest["chat_fallback_calls"], 1)


class PersonaAutoTests(RealStackTestCase):
    """persona_mode=auto 从真实 persona_manager 读取默认人格。"""

    async def test_auto_reads_real_default_persona_with_bound(self):
        await self.start_stack(with_decision=False)
        provider = PersonaSummaryProvider(self.context, {})

        summary = await provider.get()
        first_call = await provider.get()

        # DEFAULT_PERSONALITY 的真实 prompt 体现在摘要中，且按 UMO 缓存
        self.assertIn("helpful and friendly", summary)
        self.assertEqual(summary, first_call)
        self.assertLessEqual(len(summary), 700)

    async def test_manual_and_off_keep_configured_semantics(self):
        await self.start_stack(with_decision=False)
        manual = PersonaSummaryProvider(
            self.context, {"persona_mode": "manual", "persona_summary": "固定摘要"}
        )
        off = PersonaSummaryProvider(self.context, {"persona_mode": "off"})
        self.assertEqual(await manual.get(), "固定摘要")
        self.assertEqual(await off.get(), "")


class VerifierToolLoopTests(RealStackTestCase):
    """真实 tool_loop_agent 核验：工具权限、步数上限、预算与防重入。"""

    def _verifier(self, llm: MaintenanceLLM, config: dict | None = None):
        return MaintenanceVerifier(
            context=self.context,
            memory_mgr=self.mgr,
            llm=llm,
            config={"maintenance_model_id": CHAT_PROVIDER_ID, **(config or {})},
        )

    def _task(
        self,
        *,
        kind: str = "archive",
        uris=("facts://v1",),
        umo: str = UMO_U1,
        cid: str = "",
    ):
        return VerificationTask(
            kind=kind, uris=tuple(uris), umo=umo, conversation_id=cid
        )

    async def _seed_verification_context(self):
        await self.start_stack(
            with_decision=False,
            memories=[
                (
                    "v1",
                    "用户明天要在杭州开会",
                    make_memory_meta("facts://v1", owner="qq_u1", recall=5),
                )
            ],
        )
        self.cid = await self.context.conversation_manager.new_conversation(UMO_U1)
        await self.context.conversation_manager.add_message_pair(
            self.cid,
            {"role": "user", "content": "明天几点开会"},
            {"role": "assistant", "content": [{"type": "text", "text": "上午九点"}]},
        )

    async def test_tool_loop_reads_real_data_and_submits_valid_proposal(self):
        await self._seed_verification_context()
        llm = self.make_llm()
        verifier = self._verifier(llm)
        task = self._task(cid=self.cid)
        self.server.queue_chat_message(
            _tool_call("call_1", "get_memory_detail", {"uri": "facts://v1"})
        )
        self.server.queue_chat_message(
            _tool_call("call_2", "get_conversation_context", {})
        )
        self.server.queue_chat_message(
            _tool_call(
                "call_3",
                "submit_verification",
                {
                    "proposal": {
                        "kind": "archive",
                        "uri": "facts://v1",
                        "reason": "过期的相对时间事件",
                        "confidence": 0.9,
                    }
                },
            )
        )
        self.server.queue_chat("核验完成")

        # 核验上下文经真实 ConversationStore 读取（同 UMO 校验通过）
        event = build_internal_event(UMO_U1)
        context_text = await verifier._conversation_context(task)
        self.assertIn("明天几点开会", context_text)
        self.assertIn("上午九点", context_text)
        # 记忆详情工具只读允许集合内的 URI，返回真实正文
        tools = verifier._build_tools(task, [])
        detail = await tools.get_tool("get_memory_detail").handler(
            event, uri="facts://v1"
        )
        self.assertIn("用户明天要在杭州开会", detail)
        rejected = await tools.get_tool("get_memory_detail").handler(
            event, uri="facts://outside"
        )
        self.assertIn("不在本次允许核验范围", rejected)

        proposals, executed = await verifier.run([task], allowance=1)

        self.assertEqual(executed, 1)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["type"], "archive")
        self.assertEqual(proposals[0]["uri"], "facts://v1")
        # 保守预算预留：一次核验即扣减 max_steps+1 个周期调用额度
        self.assertEqual(llm.calls, VERIFICATION_BUDGET_RESERVE)
        self.assertEqual(len(_chat_prompts(self.server)), 4)
        # 核验只读：记忆未被废弃
        self.assertFalse((await self.doc_meta("facts://v1")).get("deprecated"))

    async def test_out_of_scope_proposals_are_discarded(self):
        await self._seed_verification_context()
        llm = self.make_llm()
        verifier = self._verifier(llm)
        task = self._task()
        self.server.queue_chat_message(
            _tool_call(
                "call_1",
                "submit_verification",
                {
                    "proposal": {
                        "kind": "archive",
                        "uri": "facts://not-allowed",
                        "reason": "越界提案",
                        "confidence": 0.9,
                    }
                },
            )
        )
        self.server.queue_chat_message(
            _tool_call(
                "call_2",
                "submit_verification",
                {
                    "proposal": {
                        "kind": "archive",
                        "uri": "facts://v1",
                        "reason": "过期事件",
                        "confidence": 0.9,
                    }
                },
            )
        )
        self.server.queue_chat("核验完成")

        proposals, executed = await verifier.run([task], allowance=1)

        self.assertEqual(executed, 1)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["uri"], "facts://v1")

    async def test_update_and_conflict_proposals_convert_via_real_fetches(self):
        await self.start_stack(
            with_decision=False,
            memories=[
                (
                    "v1",
                    "我讨厌死张三了，他上次骗了我",
                    make_memory_meta(
                        "facts://v1",
                        owner="qq_u1",
                        created_at="2026-01-01T00:00:00+00:00",
                    ),
                ),
                (
                    "v2",
                    "用户后来和张三和解了",
                    make_memory_meta(
                        "facts://v2",
                        owner="qq_u1",
                        created_at="2026-02-01T00:00:00+00:00",
                    ),
                ),
            ],
        )
        llm = self.make_llm()
        verifier = self._verifier(llm)
        task = self._task(kind="update", uris=("facts://v1", "facts://v2"))
        # 先提交 update（情绪记忆中性化改写），再提交 conflict（新旧评价矛盾）
        self.server.queue_chat_message(
            _tool_call(
                "call_1",
                "submit_verification",
                {
                    "proposal": {
                        "kind": "update",
                        "uri": "facts://v1",
                        "new_content": "用户对张三有负面评价，源于一次欺骗",
                        "reason": "第一人称情绪宣泄，改写为中性陈述",
                        "confidence": 0.9,
                    }
                },
            )
        )
        self.server.queue_chat_message(
            _tool_call(
                "call_2",
                "submit_verification",
                {
                    "proposal": {
                        "kind": "conflict",
                        "uris": ["facts://v1", "facts://v2"],
                        "reason": "对张三的评价前后矛盾",
                        "confidence": 0.85,
                    }
                },
            )
        )
        self.server.queue_chat("核验完成")

        proposals, executed = await verifier.run([task], allowance=1)

        self.assertEqual(executed, 1)
        # 异步查询链路完整：改写校验与冲突定向都基于真实记忆数据
        update_ops = [op for op in proposals if op["type"] == "update"]
        conflict_ops = [op for op in proposals if op["type"] == "contradiction"]
        self.assertEqual(len(update_ops), 1)
        self.assertEqual(update_ops[0]["uri"], "facts://v1")
        self.assertEqual(
            update_ops[0]["new_content"], "用户对张三有负面评价，源于一次欺骗"
        )
        self.assertEqual(len(conflict_ops), 1)
        self.assertEqual(conflict_ops[0]["old_uri"], "facts://v1")
        self.assertEqual(conflict_ops[0]["new_uri"], "facts://v2")

    async def test_nan_confidence_and_blank_merge_proposals_discarded(self):
        await self.start_stack(
            with_decision=False,
            memories=[
                (
                    "v1",
                    "用户在杭州工作",
                    make_memory_meta("facts://v1", owner="qq_u1"),
                ),
                (
                    "v2",
                    "用户做后端开发",
                    make_memory_meta("facts://v2", owner="qq_u1"),
                ),
            ],
        )
        llm = self.make_llm()
        verifier = self._verifier(llm)
        task = self._task(uris=("facts://v1", "facts://v2"))
        # NaN 置信度的 archive 与清理后为空的 merge 都不得转成操作
        self.server.queue_chat_message(
            _tool_call(
                "call_1",
                "submit_verification",
                {
                    "proposal": {
                        "kind": "archive",
                        "uri": "facts://v1",
                        "reason": "过期事件",
                        "confidence": float("nan"),
                    }
                },
            )
        )
        self.server.queue_chat_message(
            _tool_call(
                "call_2",
                "submit_verification",
                {
                    "proposal": {
                        "kind": "merge",
                        "uris": ["facts://v1", "facts://v2"],
                        "new_content": "   ",
                        "reason": "同一事实",
                        "confidence": 0.9,
                    }
                },
            )
        )
        self.server.queue_chat("核验完成")

        proposals, executed = await verifier.run([task], allowance=1)

        self.assertEqual(executed, 1)
        self.assertEqual(proposals, [])

    async def test_max_steps_bounds_model_calls(self):
        await self._seed_verification_context()
        llm = self.make_llm()
        verifier = self._verifier(llm)
        task = self._task()
        for index in range(VERIFICATION_MAX_STEPS):
            self.server.queue_chat_message(
                _tool_call(f"call_{index}", "get_memory_detail", {"uri": "facts://v1"})
            )
        # 强制收口阶段的最后一次模型调用回落到默认响应

        proposals, executed = await verifier.run([task], allowance=1)

        self.assertEqual(executed, 1)
        self.assertEqual(proposals, [])
        # 5 步工具调用 + 1 次强制收口，模型调用被 max_steps 严格封顶
        self.assertEqual(len(_chat_prompts(self.server)), VERIFICATION_BUDGET_RESERVE)

    async def test_insufficient_budget_does_not_start_verification(self):
        await self.start_stack(with_decision=False)
        llm = self.make_llm(max_calls=VERIFICATION_BUDGET_RESERVE - 1)
        runner = MaintenanceRunner(
            context=self.context,
            memory_mgr=self.mgr,
            llm=llm,
            config={"maintenance_model_id": CHAT_PROVIDER_ID},
        )

        proposals, metrics = await runner._run_verifications([self._task()])

        self.assertEqual(proposals, [])
        self.assertEqual(metrics["verification_needed"], 1)
        self.assertEqual(metrics["verification_executed"], 0)
        self.assertEqual(metrics["verification_budget_skipped"], 1)
        self.assertEqual(llm.calls, 0)
        self.assertEqual(self.server.request_log, [])

    async def test_per_cycle_cap_limits_verifications(self):
        await self._seed_verification_context()
        llm = self.make_llm()
        runner = MaintenanceRunner(
            context=self.context,
            memory_mgr=self.mgr,
            llm=llm,
            config={
                "maintenance_model_id": CHAT_PROVIDER_ID,
                "maintenance_verification_max_per_cycle": 1,
            },
        )
        # 第一条核验完成（单次模型调用即收口），第二条超出周期上限
        self.server.queue_chat("无法核验")

        proposals, metrics = await runner._run_verifications(
            [self._task(), self._task(uris=("facts://v2",))]
        )

        self.assertEqual(metrics["verification_needed"], 2)
        self.assertEqual(metrics["verification_executed"], 1)
        self.assertEqual(metrics["verification_budget_skipped"], 1)
        self.assertEqual(len(_chat_prompts(self.server)), 1)
        self.assertEqual(runner._verification_used, 1)

    async def test_internal_event_marker_blocks_recollection(self):
        await self.start_stack(
            with_decision=False,
            memories=[
                (
                    "v1",
                    "用户在杭州工作",
                    make_memory_meta("facts://v1", owner="qq_u1"),
                )
            ],
        )
        plugin = MemoryPlugin(
            self.context,
            {
                "maintenance_enabled": True,
                "auto_memorize": True,
                "extraction_interval": 1,
                "extraction_min_content_length": 1,
                "extraction_provider_id": CHAT_PROVIDER_ID,
            },
        )
        plugin.memory_mgr = self.mgr

        event = build_internal_event(UMO_U1)
        self.assertIsInstance(event, AstrMessageEvent)
        self.assertTrue(event._is_maintenance_internal)
        # UMO private 对应 AstrBot 原生 FriendMessage 会话
        self.assertEqual(event.unified_msg_origin, "qq:FriendMessage:u1")

        run_context = ContextWrapper(
            context=None, messages=[], run_end_reason="completed"
        )
        await plugin.on_agent_begin(event, run_context)
        await plugin.on_agent_done(
            event, run_context, LLMResponse(role="assistant", completion_text="好的")
        )

        rows = await self.mgr.inbox.fetch_recent_rows_for_umo(UMO_U1)
        self.assertEqual(rows, [])
        self.assertEqual(len(await self.active_docs()), 1)


class ReviewerPerOpTests(RealStackTestCase):
    """逐条审核：单操作单调用、磁盘缓存、关联重审、预算与非法输出 fail closed。"""

    async def asyncSetUp(self):
        await self.start_stack(
            with_decision=False,
            memories=[
                ("r1", "用户在杭州做后端开发", make_memory_meta("facts://r1")),
                ("r2", "用户的常驻城市是杭州", make_memory_meta("facts://r2")),
                ("r3", "用户喜欢喝手冲咖啡", make_memory_meta("facts://r3")),
            ],
        )

    def _reviewer(self, llm: MaintenanceLLM, config: dict | None = None):
        return ReviewerAgent(
            self.context,
            self.mgr,
            llm,
            {"maintenance_model_id": CHAT_PROVIDER_ID, **(config or {})},
        )

    async def test_reviews_single_operation_per_call(self):
        llm = self.make_llm()
        ops = [
            {"type": "archive", "uri": "facts://r1", "reason": "疑似过时"},
            {
                "type": "new_link",
                "source": "facts://r2",
                "target": "facts://r3",
                "relation": "related",
                "reason": "同城偏好",
            },
        ]
        self.server.queue_chat(
            json.dumps({"verdict": "reject", "reason": "证据不足", "confidence": 0.9})
        )
        self.server.queue_chat(
            json.dumps({"verdict": "approve", "reason": "关联合理", "confidence": 0.9})
        )

        verdicts = await self._reviewer(llm).review(ops)

        self.assertEqual([v["verdict"] for v in verdicts], ["reject", "approve"])
        prompts = _chat_prompts(self.server)
        self.assertEqual(len(prompts), 2)
        # 每次 prompt 只含当前一条操作，不夹带其他操作
        self.assertIn("facts://r1", prompts[0])
        self.assertNotIn("facts://r3", prompts[0])
        self.assertIn("facts://r3", prompts[1])

    async def test_disk_cache_hit_skips_llm(self):
        llm = self.make_llm(cache_enabled=True)
        op = {"type": "archive", "uri": "facts://r1", "reason": "重复提案"}
        self.server.queue_chat(
            json.dumps({"verdict": "reject", "reason": "保留", "confidence": 0.9})
        )

        first = await self._reviewer(llm).review([op])
        second = await self._reviewer(llm).review([op])

        self.assertEqual(first[0]["verdict"], "reject")
        self.assertEqual(second[0]["verdict"], first[0]["verdict"])
        self.assertEqual(len(_chat_prompts(self.server)), 1)
        cache_key = llm.op_review_cache_key(op, CHAT_PROVIDER_ID)
        self.assertTrue(Path(llm._cache_path(cache_key)).exists())

    async def test_needs_context_retries_with_merged_related_operations(self):
        llm = self.make_llm()
        ops = [
            {
                "type": "merge",
                "uris": ["facts://r1", "facts://r2"],
                "reason": "疑似重复",
            },
            {"type": "archive", "uri": "facts://r1", "reason": "过时"},
            {"type": "archive", "uri": "facts://r2", "reason": "过时"},
        ]
        self.server.queue_chat(
            json.dumps(
                {
                    "verdict": "approve",
                    "reason": "需要关联上下文",
                    "confidence": 0.4,
                    "needs_context": True,
                }
            )
        )
        self.server.queue_chat(
            json.dumps({"verdict": "reject", "reason": "看后拒绝", "confidence": 0.9})
        )
        for _ in range(2):
            self.server.queue_chat(
                json.dumps(
                    {"verdict": "reject", "reason": "证据不足", "confidence": 0.9}
                )
            )

        verdicts = await self._reviewer(llm).review(ops)

        self.assertEqual(len(verdicts), 3)
        self.assertEqual(verdicts[0]["verdict"], "reject")
        prompts = _chat_prompts(self.server)
        self.assertEqual(len(prompts), 4)
        # 首轮只给关联提醒，不注入其他操作正文
        self.assertIn("#1 archive", prompts[0])
        self.assertNotIn('"index": 2', prompts[0])
        # 重审注入两个 URI 分组的完整操作（并集去重）
        self.assertIn('"index": 1', prompts[1])
        self.assertIn('"index": 2', prompts[1])
        self.assertIn("facts://r2", prompts[1])

    async def test_budget_exhaustion_fails_closed(self):
        llm = self.make_llm(max_calls=1)
        ops = [
            {"type": "archive", "uri": f"facts://r{index}", "reason": "预算"}
            for index in range(3)
        ]
        self.server.queue_chat(
            json.dumps({"verdict": "reject", "reason": "保留", "confidence": 0.9})
        )

        reviewer = self._reviewer(llm)
        verdicts = await reviewer.review(ops)

        self.assertEqual(len(verdicts), 1)
        self.assertEqual(reviewer.last_unresolved["budget_exhausted"], 2)
        self.assertEqual(reviewer.last_unresolved["invalid_output"], 0)
        self.assertEqual(len(_chat_prompts(self.server)), 1)

    async def test_invalid_output_yields_no_verdict(self):
        llm = self.make_llm()
        self.server.queue_chat("不是 JSON")
        reviewer = self._reviewer(llm)

        verdicts = await reviewer.review(
            [{"type": "archive", "uri": "facts://r1", "reason": "非法输出"}]
        )

        self.assertEqual(verdicts, [])
        self.assertEqual(reviewer.last_unresolved["invalid_output"], 1)


class PendingFlushAndBudgetTests(_SeededMemoryCase):
    """待审队列：跨周期去重、容量溢出、终审重提、预算尾部转人工与日志脱敏。"""

    async def asyncSetUp(self):
        await self.start_stack(with_decision=False, with_plugin=True)
        # 每个用例独立 KV 命名空间，避免跨用例串数据
        self.plugin.plugin_id = f"test/{uuid.uuid4().hex}"
        await self.seed_memory("facts://q1", "用户在杭州做后端开发")
        await self.seed_memory("facts://q2", "用户的常驻城市是杭州")
        await self.seed_memory("facts://q3", "用户喜欢喝手冲咖啡")

    def _runner(self, **config):
        return MaintenanceRunner(
            context=self.context,
            memory_mgr=self.mgr,
            llm=self.make_llm(),
            config={"maintenance_model_id": CHAT_PROVIDER_ID, **config},
            kv_put=self.plugin.put_kv_data,
            kv_get=self.plugin.get_kv_data,
        )

    async def _queue(self) -> list | None:
        return await self.plugin.get_kv_data("maintenance_pending_review", None)

    async def _overflow(self) -> list | None:
        return await self.plugin.get_kv_data(
            "maintenance_pending_review_overflow", None
        )

    async def test_dedupes_by_signature_across_cycles_and_requeues_after_approval(
        self,
    ):
        runner = self._runner()
        op = {"type": "archive", "uri": "facts://r1", "reason": "重复提案"}

        await runner._flush_pending_review([(op, {"reason": "x"})], "s1")
        await runner._flush_pending_review([(op, {"reason": "x"})], "s2")

        queue = await self._queue()
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["status"], "pending")

        # 管理员批准后条目变终态并被清理，同操作可在下周期重新提议
        queue[0]["status"] = "approved"
        await self.plugin.put_kv_data("maintenance_pending_review", queue)
        await runner._flush_pending_review([(op, {"reason": "x"})], "s3")

        queue = await self._queue()
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["status"], "pending")
        self.assertEqual(queue[0]["id"], 2)

    async def test_capacity_persists_overflow_and_drains_after_release(self):
        runner = self._runner(maintenance_pending_queue_max=2)
        items = [
            ({"type": "archive", "uri": f"facts://o{index}"}, {"reason": "x"})
            for index in range(5)
        ]

        await runner._flush_pending_review(items, "s1")

        queue = await self._queue()
        overflow = await self._overflow()
        self.assertEqual(
            [item["op"]["uri"] for item in queue], ["facts://o0", "facts://o1"]
        )
        self.assertEqual(
            [entry["op"]["uri"] for entry in overflow],
            ["facts://o2", "facts://o3", "facts://o4"],
        )

        # 释放一个待审名额后，溢出条目按序回填到容量上限
        queue[0]["status"] = "approved"
        await self.plugin.put_kv_data("maintenance_pending_review", queue)
        await runner._flush_pending_review([], "s2")

        queue = await self._queue()
        overflow = await self._overflow()
        self.assertEqual(
            [item["op"]["uri"] for item in queue], ["facts://o1", "facts://o2"]
        )
        self.assertEqual(
            [entry["op"]["uri"] for entry in overflow],
            ["facts://o3", "facts://o4"],
        )

        # 继续释放名额直到溢出队列排空
        for item in queue:
            item["status"] = "approved"
        await self.plugin.put_kv_data("maintenance_pending_review", queue)
        await runner._flush_pending_review([], "s3")

        queue = await self._queue()
        self.assertEqual(
            [item["op"]["uri"] for item in queue], ["facts://o3", "facts://o4"]
        )
        self.assertEqual(await self._overflow(), [])

        for item in queue:
            item["status"] = "approved"
        await self.plugin.put_kv_data("maintenance_pending_review", queue)
        await runner._flush_pending_review([], "s4")

        self.assertEqual(await self._queue(), [])

    async def test_capacity_counts_pending_only(self):
        runner = self._runner(maintenance_pending_queue_max=2)
        # 已有 3 条终态条目，不应占容量
        await self.plugin.put_kv_data(
            "maintenance_pending_review",
            [
                {
                    "id": index,
                    "status": status,
                    "op": {"type": "archive", "uri": f"facts://old{index}"},
                }
                for index, status in enumerate(
                    ("approved", "rejected", "failed"), start=1
                )
            ],
        )
        items = [
            ({"type": "archive", "uri": f"facts://new{index}"}, {"reason": "x"})
            for index in range(3)
        ]

        await runner._flush_pending_review(items, "s1")

        queue = await self._queue()
        overflow = await self._overflow()
        self.assertEqual(
            [item["op"]["uri"] for item in queue],
            ["facts://new0", "facts://new1"],
        )
        self.assertTrue(all(item["status"] == "pending" for item in queue))
        self.assertEqual([entry["op"]["uri"] for entry in overflow], ["facts://new2"])

    async def test_budget_tail_operations_go_to_human_review(self):
        runner = self._runner()
        runner._last_review_unresolved = {"budget_exhausted": 2, "invalid_output": 0}
        report = MaintenanceReport(
            session_id="s1",
            organizer_manifest=AgentManifest(
                agent_type="organizer",
                parsed=True,
                operations=[
                    {"type": "archive", "uri": "facts://q1", "reason": "过时"},
                    {"type": "archive", "uri": "facts://q2", "reason": "过时"},
                    {"type": "archive", "uri": "facts://q3", "reason": "过时"},
                ],
            ),
        )
        report.reviewer_verdicts = [
            {"index": 0, "verdict": "approve", "reason": "证据充分"}
        ]

        await runner._execute_operations(report)

        self.assertEqual(report.executed_ops, 1)
        self.assertEqual(report.skipped_ops, 2)
        self.assertTrue((await self.doc_meta("facts://q1"))["deprecated"])
        self.assertFalse((await self.doc_meta("facts://q2"))["deprecated"])
        self.assertFalse((await self.doc_meta("facts://q3"))["deprecated"])
        queue = await self._queue()
        self.assertEqual(
            [item["op"]["uri"] for item in queue], ["facts://q2", "facts://q3"]
        )
        self.assertTrue(all(item["status"] == "pending" for item in queue))
        self.assertEqual(queue[0]["verdict_reason"], "审核预算耗尽，转人工审核")

    async def test_execution_and_review_logs_hide_memory_bodies(self):
        runner = self._runner()
        merge_op = {
            "type": "merge",
            "uris": ["facts://q1", "facts://q2"],
            "merged_content": "private-memory-body",
            "reason": "同义合并",
        }
        disputed_op = {
            "type": "archive",
            "uri": "facts://q3",
            "reason": "private-memory-body",
        }
        report = MaintenanceReport(
            session_id="sensitive-log",
            organizer_manifest=AgentManifest(
                agent_type="organizer",
                parsed=True,
                operations=[merge_op, disputed_op],
            ),
        )
        report.reviewer_verdicts = [
            {"index": 0, "verdict": "approve", "reason": "ok"},
            {
                "index": 1,
                "verdict": "reject",
                "reason": "sensitive-review-reason",
                "needs_human_review": True,
            },
        ]

        async def failing_merge(*args, **kwargs):
            raise RuntimeError("sensitive-database-details")

        self.mgr.merge_memories = failing_merge

        with self.assertLogs("astrbot", level="INFO") as captured:
            await runner._execute_operations(report)

        output = "\n".join(captured.output)
        self.assertNotIn("private-memory-body", output)
        self.assertNotIn("sensitive-review-reason", output)
        self.assertNotIn("sensitive-database-details", output)
        self.assertIn("index=1", output)
        self.assertIn("type=archive", output)
        self.assertIn("type=merge", output)
        self.assertIn("error_type=RuntimeError", output)
        self.assertEqual(report.errors, ["op[0] merge 执行返回失败"])

        # 正文与裁决理由只落待审队列，不落日志
        queue = await self._queue()
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["op"]["uri"], "facts://q3")
        self.assertEqual(queue[0]["verdict_reason"], "sensitive-review-reason")


class ContradictionAndLinkCleanupTests(_SeededMemoryCase):
    """矛盾闭环与孤儿关联清理：全部在真实 KB 上执行。"""

    async def asyncSetUp(self):
        await self.start_stack(with_decision=False, with_plugin=True)
        self.plugin.plugin_id = f"test/{uuid.uuid4().hex}"
        await self.seed_memory(
            "facts://old", "用户的家乡是宁波", created_at="2026-01-01T00:00:00+00:00"
        )
        await self.seed_memory(
            "facts://new",
            "用户的家乡已经搬到杭州",
            created_at="2026-02-01T00:00:00+00:00",
        )
        await self.seed_memory("facts://maybe", "用户可能去过苏州")
        await self.seed_memory("facts://dep", "用户旧的工作地点是苏州", deprecated=True)

    def _runner(self):
        return MaintenanceRunner(
            context=self.context,
            memory_mgr=self.mgr,
            llm=self.make_llm(),
            config={"maintenance_model_id": CHAT_PROVIDER_ID},
            kv_put=self.plugin.put_kv_data,
            kv_get=self.plugin.get_kv_data,
        )

    async def test_approved_contradiction_waits_for_human_then_deprecates_old(self):
        runner = self._runner()
        report = MaintenanceReport(
            session_id="s1",
            analyst_manifest=AgentManifest(
                agent_type="analyst",
                parsed=True,
                operations=[
                    {
                        "type": "contradiction",
                        "old_uri": "facts://old",
                        "new_uri": "facts://new",
                        "reason": "家乡变更",
                    }
                ],
            ),
        )
        report.reviewer_verdicts = [
            {"index": 0, "verdict": "approve", "reason": "确认矛盾"}
        ]

        await runner._execute_operations(report)

        # approve 也不自动执行：进入待审队列等待管理员裁决
        self.assertEqual(report.executed_ops, 0)
        self.assertEqual(report.skipped_ops, 1)
        self.assertFalse((await self.doc_meta("facts://old"))["deprecated"])
        queue = await self.plugin.get_kv_data("maintenance_pending_review", None)
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["op"]["type"], "contradiction")
        self.assertEqual(queue[0]["status"], "pending")
        self.assertEqual(queue[0]["verdict_reason"], "矛盾待人工裁决: 确认矛盾")

        self.assertTrue(await runner.execute_approved(queue[0]["op"]))

        self.assertTrue((await self.doc_meta("facts://old"))["deprecated"])
        self.assertFalse((await self.doc_meta("facts://new"))["deprecated"])

    async def test_execute_approved_contradiction_requires_existing_old_uri(self):
        runner = self._runner()

        self.assertFalse(await runner.execute_approved({"type": "contradiction"}))
        self.assertFalse(
            await runner.execute_approved(
                {"type": "contradiction", "old_uri": "facts://missing"}
            )
        )
        self.assertFalse((await self.doc_meta("facts://old"))["deprecated"])

        # 未给出理由时使用默认裁决理由，仍然真实废弃旧记忆
        self.assertTrue(
            await runner.execute_approved(
                {"type": "contradiction", "old_uri": "facts://old"}
            )
        )
        meta = await self.doc_meta("facts://old")
        self.assertTrue(meta["deprecated"])
        self.assertEqual(meta["deprecated_reason"], "contradiction resolved")

    async def test_cleanup_removes_only_links_with_missing_endpoints(self):
        links = self.mgr.link_manager
        for source, target in (
            ("facts://old", "facts://new"),
            ("facts://old", "facts://dep"),
            ("facts://missing-source", "facts://old"),
            ("facts://old", "facts://missing-target"),
        ):
            self.assertTrue(await links.add_link(source, target, "related"))

        removed = await self.mgr.cleanup_orphan_links()
        remaining = {
            (row["source_uri"], row["target_uri"]) for row in await links.export_all()
        }
        repeated = await self.mgr.cleanup_orphan_links()

        # 指向已废弃但文档仍在的记忆不算孤儿
        self.assertEqual(removed, 2)
        self.assertEqual(
            remaining,
            {("facts://old", "facts://new"), ("facts://old", "facts://dep")},
        )
        self.assertEqual(repeated, 0)


class OrganizerMergeAndExecuteTests(_SeededMemoryCase):
    """合并通道 URI 消费互斥与真实 merge 落地（含预验证拒绝）。"""

    def _organizer(self, llm: MaintenanceLLM, config: dict | None = None):
        return OrganizerAgent(
            context=self.context,
            memory_mgr=self.mgr,
            llm=llm,
            config={"maintenance_model_id": CHAT_PROVIDER_ID, **(config or {})},
        )

    def _runner(self):
        return MaintenanceRunner(
            context=self.context,
            memory_mgr=self.mgr,
            llm=self.make_llm(),
            config={"maintenance_model_id": CHAT_PROVIDER_ID},
        )

    async def _merged_metadata(self) -> list[dict]:
        return [
            meta
            for meta in (_doc_metadata(doc) for doc in await self.active_docs())
            if meta.get("merged_from")
        ]

    async def test_merge_channel_consumes_uris_and_skips_them_in_inspection(self):
        # 三条同归属记忆使用相同向量，余弦 1.0 → 预筛出全部三对候选
        await self.start_stack(with_decision=True)
        # 三条同归属记忆使用相同向量，余弦 1.0 → 预筛出全部三对候选
        self.server.set_embedding_vectors({"住在杭州": [0.5] * 8})
        await self.seed_memory("facts://m1", "用户住在杭州")
        await self.seed_memory("facts://m2", "用户的居住地是杭州")
        await self.seed_memory("facts://m3", "用户目前住在杭州")
        llm = self.make_llm()
        self.server.queue_chat(
            json.dumps(
                {
                    "verdict": "merge",
                    "reason": "同一居住地事实",
                    "weight": 0.9,
                    "fused_text": "用户住在杭州",
                },
                ensure_ascii=False,
            )
        )
        # 巡检只应看到未被合并消费的 m3：闸门拒收 → 清洗判 keep
        self.server.set_decision_factory(
            decisions_by_request_probability([0.1, 0.05, 0.05, 0.05])
        )
        self.server.queue_chat(
            json.dumps(
                {
                    "results": [
                        {
                            "id": "facts://m3",
                            "action": "keep",
                            "reason": "证据不足",
                            "confidence": 0.9,
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )

        manifest = await self._organizer(llm).run()

        self.assertEqual(manifest["candidates_screened"], 3)
        self.assertEqual(len(manifest["merge"]), 1)
        self.assertEqual(manifest["merge"][0]["uris"], ["facts://m1", "facts://m2"])
        self.assertEqual(manifest["merge"][0]["merged_content"], "用户住在杭州")
        # 同一条 URI 不会被相邻候选重复合并
        judge_prompts = [p for p in _chat_prompts(self.server) if "向量余弦" in p]
        self.assertEqual(len(judge_prompts), 1)
        # 已被合并消费的 m1/m2 不再进入单条存量巡检
        self.assertEqual(manifest["inspected"], 1)
        self.assertEqual(manifest["inspection_kept"], 1)
        decision_requests = [
            r for r in self.server.request_log if r["kind"] == "decisions"
        ]
        self.assertEqual(len(decision_requests), 1)
        # 单条巡检只提交一个候选：闸门四头各一问
        self.assertEqual(len(decision_requests[0]["questions"]), 4)
        cleaning_prompt = _chat_prompts(self.server)[-1]
        self.assertIn("facts://m3", cleaning_prompt)
        self.assertNotIn("facts://m1", cleaning_prompt)
        self.assertNotIn("facts://m2", cleaning_prompt)

    async def test_execute_approved_merge_writes_new_memory_and_supersedes_links(
        self,
    ):
        await self.start_stack(with_decision=False)
        await self.seed_memory("facts://m1", "用户住在杭州")
        await self.seed_memory("facts://m2", "用户的居住地是杭州")
        runner = self._runner()

        self.assertTrue(
            await runner.execute_approved(
                {
                    "type": "merge",
                    "uris": ["facts://m1", "facts://m2"],
                    "merged_content": "合并后内容",
                    "reason": "同义合并",
                }
            )
        )

        merged = await self._merged_metadata()
        self.assertEqual(len(merged), 1)
        merged_meta = merged[0]
        self.assertTrue(merged_meta["uri"].startswith("facts://merged_"))
        self.assertEqual(merged_meta["memory_content"], "合并后内容")
        self.assertEqual(merged_meta["merged_from"], ["facts://m1", "facts://m2"])
        self.assertEqual(merged_meta["merged_reason"], "同义合并")
        self.assertEqual(merged_meta["created_by"], "organizer")
        self.assertFalse(merged_meta["deprecated"])

        for uri in ("facts://m1", "facts://m2"):
            self.assertTrue((await self.doc_meta(uri))["deprecated"])

        links = await self.mgr.link_manager.get_links_for_uri(
            merged_meta["uri"], injectable_only=False
        )
        self.assertEqual(
            {(row["target_uri"], row["relation_type"]) for row in links},
            {("facts://m1", "supersedes"), ("facts://m2", "supersedes")},
        )

    async def test_merge_prevalidation_rejects_missing_or_deprecated_sources(self):
        await self.start_stack(with_decision=False)
        await self.seed_memory("facts://m3", "用户喜欢喝手冲咖啡")
        await self.seed_memory("facts://m4", "用户每天早上都喝手冲咖啡")
        runner = self._runner()
        op = {
            "type": "merge",
            "merged_content": "合并后内容",
            "reason": "同义合并",
        }

        # 源 URI 不存在 → 预验证拒绝，源记忆不动、不写合并记录
        self.assertFalse(
            await runner.execute_approved(
                {**op, "uris": ["facts://missing", "facts://m3"]}
            )
        )
        self.assertFalse((await self.doc_meta("facts://m3"))["deprecated"])
        self.assertEqual(len(await self.active_docs()), 2)
        self.assertEqual(await self._merged_metadata(), [])

        # 源 URI 已 deprecated → 同样拒绝，另一条源保持活跃
        self.assertTrue(await self.mgr.deprecate_memory("facts://m3", reason="过时"))
        self.assertFalse(
            await runner.execute_approved({**op, "uris": ["facts://m3", "facts://m4"]})
        )
        self.assertFalse((await self.doc_meta("facts://m4"))["deprecated"])
        self.assertEqual(len(await self.active_docs()), 1)
        self.assertEqual(await self._merged_metadata(), [])


class PurgeDeprecatedTests(RealStackTestCase):
    """物理清理：dry-run 只统计，正式执行按宽限期删文档与关联边。"""

    async def asyncSetUp(self):
        await self.start_stack(with_decision=False)
        kb = await self.context.kb_manager.get_kb_by_name(self.kb_name)
        self.kb_id = kb.kb.kb_id
        now = datetime.now(timezone.utc)
        self.expired_uri = "facts://expired"
        self.grace_uri = "facts://grace"
        self.active_uri = "facts://active"
        self.doc_ids: dict[str, str] = {}
        for uri, deprecated, days in (
            (self.expired_uri, True, 15),
            (self.grace_uri, True, 2),
            (self.active_uri, False, 0),
        ):
            meta = make_memory_meta(uri, deprecated=deprecated)
            meta["kb_id"] = self.kb_id
            if deprecated:
                meta["deprecated_at"] = (now - timedelta(days=days)).isoformat()
            doc_id = str(uuid.uuid4())
            meta["kb_doc_id"] = doc_id
            text = f"{uri} 的记忆正文"
            await self.mgr.vec_db.insert(text, meta, doc_id)
            await self.mgr._register_kb_document(doc_id, uri, len(text))
            self.doc_ids[uri] = doc_id

        links = self.mgr.link_manager
        for source, target in (
            (self.expired_uri, self.grace_uri),
            (self.grace_uri, self.active_uri),
        ):
            self.assertTrue(await links.add_link(source, target, "related"))

    async def _docs(self, uri: str) -> list[dict]:
        return await self.mgr.vec_db.document_storage.get_documents(
            metadata_filters={"uri": uri, "is_memory_record": True}, limit=10
        )

    async def _kb_document_ids(self) -> set[str]:
        from astrbot.core.knowledge_base.models import KBDocument
        from sqlmodel import select

        kb = await self.context.kb_manager.get_kb_by_name(self.kb_name)
        async with kb.kb_db.get_db() as session:
            rows = (await session.execute(select(KBDocument))).scalars().all()
        return {row.doc_id for row in rows}

    async def test_purge_physically_removes_only_expired_deprecated_memory(self):
        preview = await self.mgr.purge_deprecated(after_days=7, dry_run=True)

        self.assertEqual(preview["candidates"], 1)
        self.assertEqual(preview["purged"], 0)
        self.assertEqual(len(await self._docs(self.expired_uri)), 1)
        self.assertEqual(await self.mgr.link_manager.count_links(), 2)

        result = await self.mgr.purge_deprecated(after_days=7)

        self.assertEqual(result["purged"], 1)
        self.assertGreaterEqual(result["links_cleaned"], 1)
        self.assertEqual(await self._docs(self.expired_uri), [])
        self.assertEqual(len(await self._docs(self.grace_uri)), 1)
        self.assertTrue((await self.doc_meta(self.grace_uri))["deprecated"])
        self.assertEqual(len(await self._docs(self.active_uri)), 1)
        self.assertFalse((await self.doc_meta(self.active_uri))["deprecated"])

        remaining = {
            (row["source_uri"], row["target_uri"])
            for row in await self.mgr.link_manager.export_all()
        }
        self.assertEqual(remaining, {(self.grace_uri, self.active_uri)})

        kb_document_ids = await self._kb_document_ids()
        self.assertNotIn(self.doc_ids[self.expired_uri], kb_document_ids)
        self.assertIn(self.doc_ids[self.grace_uri], kb_document_ids)
        self.assertIn(self.doc_ids[self.active_uri], kb_document_ids)


class DryRunZeroWriteTests(_SeededMemoryCase):
    """dry-run 全周期与后台测试项：真实管线产出提案但零持久化写入。"""

    async def test_cycle_dry_run_executes_pipeline_without_writes(self):
        uri = "facts://noise"
        await self.start_stack(
            with_decision=True,
            with_plugin=True,
            memories=[
                (
                    "n1",
                    "任务已完成，回执编号 42",
                    make_memory_meta(uri, owner="qq_u1", recall=4),
                ),
                (
                    "n2",
                    "用户在杭州做后端开发",
                    make_memory_meta(
                        "facts://n2", owner="qq_u1", created_by="memory_store"
                    ),
                ),
            ],
        )
        self.plugin.plugin_id = f"test/{uuid.uuid4().hex}"
        # Decision 未配置脚本应答 → 各阶段降级 Chat JSON
        self.server.set_decision_factory(lambda request: None)
        # merge 预筛对 → 不合并；闸门拒收回执记忆 → 清洗 archive；分析 none；审核 approve
        self.server.queue_chat(
            json.dumps(
                {
                    "verdict": "none",
                    "reason": "不同事实",
                    "weight": 0,
                    "fused_text": "",
                },
                ensure_ascii=False,
            )
        )
        self.server.queue_chat(
            _gate_chat_payload(
                [
                    _gate_result(uri, durable=0.1, risk=0.2),
                    _gate_result("facts://n2", durable=0.95, risk=0.05),
                ]
            )
        )
        self.server.queue_chat(
            json.dumps(
                {
                    "results": [
                        {
                            "id": uri,
                            "action": "archive",
                            "reason": "一次性任务回执",
                            "confidence": 0.9,
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )
        self.server.queue_chat(
            json.dumps(
                {
                    "results": [
                        {
                            "id": "0",
                            "verdict": "none",
                            "reason": "不相关",
                            "confidence": 0.9,
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )
        self.server.queue_chat(
            json.dumps(
                {"verdict": "approve", "reason": "证据充分", "confidence": 0.9},
                ensure_ascii=False,
            )
        )

        llm = self.make_llm()
        cache_dir = Path(llm._cache_dir)
        runner = MaintenanceRunner(
            context=self.context,
            memory_mgr=self.mgr,
            llm=llm,
            config={
                "maintenance_model_id": CHAT_PROVIDER_ID,
                "maintenance_max_llm_calls": 50,
                "auto_purge_enabled": False,
            },
            kv_put=self.plugin.put_kv_data,
            kv_get=self.plugin.get_kv_data,
        )

        report = await runner.run_cycle(dry_run=True)

        # 管线真实执行并产出 archive 提案，但阶段 4 全部跳过
        self.assertTrue(report.dry_run)
        organizer_ops = report.organizer_manifest.operations
        self.assertEqual(len(organizer_ops), 1)
        self.assertEqual(organizer_ops[0]["type"], "archive")
        self.assertEqual(report.organizer_manifest.metrics["inspected"], 2)
        self.assertEqual(report.organizer_manifest.metrics["op_archive"], 1)
        self.assertIn("verification_needed", report.organizer_manifest.metrics)
        self.assertEqual(report.executed_ops, 0)
        self.assertEqual(report.skipped_ops, 0)
        # 记忆未被废弃、内容不变
        self.assertFalse((await self.doc_meta(uri)).get("deprecated"))
        self.assertEqual(len(await self.active_docs()), 2)
        # KV（待审队列/提取日志）零写入，LLM 磁盘缓存目录为空
        self.assertIsNone(
            await self.plugin.get_kv_data("maintenance_pending_review", None)
        )
        self.assertIsNone(
            await self.plugin.get_kv_data("maintenance_pending_review_overflow", None)
        )
        self.assertEqual(list(cache_dir.iterdir()), [])
        # 缓冲表无新增行
        self.assertEqual(await self.mgr.inbox.status_counts(), {})

    async def test_test_stage_keeps_cache_off_and_purge_stays_dry_run(self):
        await self.start_stack(with_decision=False)
        self.server.set_embedding_vectors({"住在杭州": [0.6] * 8})
        await self.seed_memory("facts://s1", "用户住在杭州")
        await self.seed_memory("facts://s2", "用户的居住地是杭州")
        llm = self.make_llm(cache_enabled=True)
        cache_dir = Path(llm._cache_dir)
        runner = MaintenanceRunner(
            context=self.context,
            memory_mgr=self.mgr,
            llm=llm,
            config={"maintenance_model_id": CHAT_PROVIDER_ID},
        )
        self.server.queue_chat(
            json.dumps(
                {"verdict": "merge", "reason": "同一事实", "weight": 0.9},
                ensure_ascii=False,
            )
        )

        organizer_stage = await runner.run_test_stage("organizer")

        self.assertTrue(organizer_stage["dry_run"])
        self.assertEqual(organizer_stage["organizer"]["operations"], 1)
        # 后台测试禁用 LLM 磁盘缓存，结束后恢复原设置
        self.assertEqual(list(cache_dir.iterdir()), [])
        self.assertTrue(llm._cache_enabled)

        purge_stage = await runner.run_test_stage("purge")
        self.assertTrue(purge_stage["dry_run"])
        self.assertTrue(purge_stage["purge"]["dry_run"])
        self.assertEqual(purge_stage["purge"]["candidates"], 0)

        with self.assertRaisesRegex(ValueError, "合法值"):
            await runner.run_test_stage("unknown")
