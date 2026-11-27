"""后台整理真实栈测试：单条存量巡检、中文四分类、persona、核验器与 dry-run。

全部基于真实 AstrBot 4.30 组件：Context、ProviderManager、DecisionService、
KnowledgeBase/FaissVecDB、ConversationStore、AstrMessageEvent、MemoryPlugin、
tool_loop_agent 均为真实实例，LLM/Decision/工具调用由标准库本地 HTTP 服务
脚本化应答；不使用任何 AstrBot 接口替身。
"""

from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

from astrbot.api.provider import LLMResponse
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from real_stack import (
    CHAT_PROVIDER_ID,
    build_real_context,
    decisions_by_request_choices,
    start_server,
)
from test_memory_commands import MemoryManager, plugin_main

from maintenance.agents.analyst import AnalystAgent
from maintenance.agents.organizer import OrganizerAgent
from maintenance.llm import MaintenanceLLM
from maintenance.persona import PersonaSummaryProvider
from maintenance.runner import MaintenanceRunner
from maintenance.verifier import (
    VERIFICATION_BUDGET_RESERVE,
    VERIFICATION_MAX_STEPS,
    MaintenanceVerifier,
    VerificationTask,
    build_internal_event,
)

MemoryPlugin = plugin_main.MemoryPlugin

UMO_U1 = "qq:private:u1"


def _meta(
    uri: str,
    *,
    scope: str = "personal",
    owner: str = "qq_u1",
    created_at: str = "2026-11-01T00:00:00+00:00",
    updated_at: str = "",
    recall: int = 0,
    created_by: str = "maintenance_curator",
    domain: str = "facts",
    umo: str = UMO_U1,
) -> dict:
    return {
        "uri": uri,
        "is_memory_record": True,
        "deprecated": False,
        "memory_scope": scope,
        "owner_user_id": owner,
        "umo": umo,
        "domain": domain,
        "created_at": created_at,
        "updated_at": updated_at,
        "recall_count": recall,
        "created_by": created_by,
    }


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


class _RealStackCase(unittest.IsolatedAsyncioTestCase):
    """真实 Context + 真实 KB + 真实 MemoryManager 的公共装配方。"""

    def _kb_name(self) -> str:
        return f"p3-{uuid.uuid4().hex[:10]}"

    async def _stack(self, *, with_decision: bool = True, memories=()):
        self.server = start_server()
        self.kb_name = self._kb_name()
        self.context = await build_real_context(
            self.server, with_decision=with_decision, kb_name=self.kb_name
        )
        self.mgr = MemoryManager(
            kb_mgr=self.context.kb_manager, config={"kb_name": self.kb_name}
        )
        self.mgr.initialize()
        await self.mgr.connect_kb()
        kb = await self.context.kb_manager.get_kb_by_name(self.kb_name)
        for doc_id, content, meta in memories:
            await kb.vec_db.insert(content, meta, doc_id)

    async def _active_docs(self) -> list[dict]:
        docs = await self.mgr.vec_db.document_storage.get_documents(
            metadata_filters={"is_memory_record": True, "deprecated": False},
            limit=100,
            offset=0,
        )
        return docs

    async def _doc_meta(self, uri: str) -> dict:
        docs = await self.mgr.vec_db.document_storage.get_documents(
            metadata_filters={"uri": uri, "is_memory_record": True}, limit=1
        )
        raw = docs[0].get("metadata", {})
        if isinstance(raw, str):
            raw = json.loads(raw)
        return raw

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def _llm(self, *, max_calls: int = 50) -> MaintenanceLLM:
        return MaintenanceLLM(
            context=self.context,
            cache_dir=tempfile.mkdtemp(prefix="p3-llm-cache-"),
            cache_enabled=False,
            default_model_id=CHAT_PROVIDER_ID,
            max_calls_per_cycle=max_calls,
        )


class OrganizerInspectionTests(_RealStackCase):
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
            _meta(uri, scope="group", owner="qq_u1", recall=7, umo="qq:group:g1"),
        )
        await self._stack(with_decision=False, memories=[memory])
        llm = self._llm()
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
        self.assertFalse((await self._doc_meta(uri)).get("deprecated"))
        self.assertEqual(len(await self._active_docs()), 1)

    async def test_identical_rewrite_fails_closed_to_keep(self):
        uri = "facts://emotional"
        content = "我讨厌死张三了，他上次骗了我"
        await self._stack(
            with_decision=False,
            memories=[("emotional", content, _meta(uri, recall=2))],
        )
        llm = self._llm()
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
        await self._stack(
            with_decision=False,
            memories=[
                (
                    "receipt",
                    "任务已完成，回执编号 42",
                    _meta(uri, owner="qq_u1", recall=5),
                )
            ],
        )
        llm = self._llm()
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
                    _meta(uri, recall=1, created_by="memory_store"),
                )
            )
        memories.append(
            (
                "risky",
                "明天早上 9 点发中秋祝福",
                _meta("facts://risky", recall=9),
            )
        )
        await self._stack(with_decision=False, memories=memories)
        llm = self._llm()
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
            llm, {"maintenance_organizer_batch_size": 1}
        ).run()

        self.assertEqual(manifest["inspected"], 1)
        self.assertEqual(manifest["inspection_kept"], 1)
        # 选样有界：闸门脚本只含风险记忆的应答，若其余 4 条也被提交判定，
        # 它们会因结果缺失被计为不可用（fail closed 保留）
        self.assertEqual(manifest["inspection_unavailable"], 0)
        chat_requests = [r for r in self.server.request_log if r["kind"] == "chat"]
        self.assertEqual(len(chat_requests), 2)


class AnalystChineseClassificationTests(_RealStackCase):
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
            ("a1", "用户在杭州做后端开发", _meta("facts://a1", owner="qq_u1")),
            ("a2", "用户的工作地点是杭州", _meta("facts://a2", owner="qq_u1")),
            ("b1", "用户在苏州做前端开发", _meta("facts://b1", owner="qq_u2")),
            ("b2", "用户的工作地点是苏州", _meta("facts://b2", owner="qq_u2")),
        ]
        await self._stack(with_decision=True, memories=memories)
        llm = self._llm()
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
                _meta(
                    "facts://old", owner="qq_u1", created_at="2026-01-01T00:00:00+00:00"
                ),
            ),
            (
                "new",
                "用户的家乡已经搬到杭州",
                _meta(
                    "facts://new", owner="qq_u1", created_at="2026-02-01T00:00:00+00:00"
                ),
            ),
        ]
        await self._stack(with_decision=True, memories=memories)
        llm = self._llm()
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
            ("c1", "用户喜欢喝手冲咖啡", _meta("facts://c1", owner="qq_u1")),
            ("c2", "用户每天早上都喝手冲咖啡", _meta("facts://c2", owner="qq_u1")),
        ]
        await self._stack(with_decision=False, memories=memories)
        llm = self._llm()
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
            ("d1", "用户喜欢手冲咖啡", _meta("facts://d1", owner="qq_u1")),
            ("d2", "用户早上都喝手冲咖啡", _meta("facts://d2", owner="qq_u1")),
        ]
        await self._stack(with_decision=False, memories=memories)
        llm = self._llm()
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


class PersonaAutoTests(_RealStackCase):
    """persona_mode=auto 从真实 persona_manager 读取默认人格。"""

    async def test_auto_reads_real_default_persona_with_bound(self):
        await self._stack(with_decision=False)
        provider = PersonaSummaryProvider(self.context, {})

        summary = await provider.get()
        first_call = await provider.get()

        # DEFAULT_PERSONALITY 的真实 prompt 体现在摘要中，且按 UMO 缓存
        self.assertIn("helpful and friendly", summary)
        self.assertEqual(summary, first_call)
        self.assertLessEqual(len(summary), 700)

    async def test_manual_and_off_keep_configured_semantics(self):
        await self._stack(with_decision=False)
        manual = PersonaSummaryProvider(
            self.context, {"persona_mode": "manual", "persona_summary": "固定摘要"}
        )
        off = PersonaSummaryProvider(self.context, {"persona_mode": "off"})
        self.assertEqual(await manual.get(), "固定摘要")
        self.assertEqual(await off.get(), "")


class VerifierToolLoopTests(_RealStackCase):
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
        await self._stack(
            with_decision=False,
            memories=[
                (
                    "v1",
                    "用户明天要在杭州开会",
                    _meta("facts://v1", owner="qq_u1", recall=5),
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
        llm = self._llm()
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
        chat_requests = [r for r in self.server.request_log if r["kind"] == "chat"]
        self.assertEqual(len(chat_requests), 4)
        # 核验只读：记忆未被废弃
        self.assertFalse((await self._doc_meta("facts://v1")).get("deprecated"))

    async def test_out_of_scope_proposals_are_discarded(self):
        await self._seed_verification_context()
        llm = self._llm()
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
        await self._stack(
            with_decision=False,
            memories=[
                (
                    "v1",
                    "我讨厌死张三了，他上次骗了我",
                    _meta(
                        "facts://v1",
                        owner="qq_u1",
                        created_at="2026-01-01T00:00:00+00:00",
                    ),
                ),
                (
                    "v2",
                    "用户后来和张三和解了",
                    _meta(
                        "facts://v2",
                        owner="qq_u1",
                        created_at="2026-02-01T00:00:00+00:00",
                    ),
                ),
            ],
        )
        llm = self._llm()
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
        await self._stack(
            with_decision=False,
            memories=[
                (
                    "v1",
                    "用户在杭州工作",
                    _meta("facts://v1", owner="qq_u1"),
                ),
                (
                    "v2",
                    "用户做后端开发",
                    _meta("facts://v2", owner="qq_u1"),
                ),
            ],
        )
        llm = self._llm()
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
        llm = self._llm()
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
        chat_requests = [r for r in self.server.request_log if r["kind"] == "chat"]
        # 5 步工具调用 + 1 次强制收口，模型调用被 max_steps 严格封顶
        self.assertEqual(len(chat_requests), VERIFICATION_BUDGET_RESERVE)

    async def test_insufficient_budget_does_not_start_verification(self):
        await self._stack(with_decision=False)
        llm = self._llm(max_calls=VERIFICATION_BUDGET_RESERVE - 1)
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
        llm = self._llm()
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
        chat_requests = [r for r in self.server.request_log if r["kind"] == "chat"]
        self.assertEqual(len(chat_requests), 1)
        self.assertEqual(runner._verification_used, 1)

    async def test_internal_event_marker_blocks_recollection(self):
        await self._stack(
            with_decision=False,
            memories=[
                (
                    "v1",
                    "用户在杭州工作",
                    _meta("facts://v1", owner="qq_u1"),
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

        run_context = SimpleNamespace(
            context=SimpleNamespace(is_subagent=False),
            messages=[],
            run_end_reason="completed",
        )
        await plugin.on_agent_begin(event, run_context)
        await plugin.on_agent_done(
            event, run_context, LLMResponse(role="assistant", completion_text="好的")
        )

        rows = await self.mgr.inbox.fetch_recent_rows_for_umo(UMO_U1)
        self.assertEqual(rows, [])
        self.assertEqual(len(await self._active_docs()), 1)


class DryRunZeroWriteTests(_RealStackCase):
    """dry-run 全周期：真实管线产出提案但零持久化写入。"""

    async def test_cycle_dry_run_executes_pipeline_without_writes(self):
        uri = "facts://noise"
        await self._stack(
            with_decision=True,
            memories=[
                (
                    "n1",
                    "任务已完成，回执编号 42",
                    _meta(uri, owner="qq_u1", recall=4),
                ),
                (
                    "n2",
                    "用户在杭州做后端开发",
                    _meta("facts://n2", owner="qq_u1", created_by="memory_store"),
                ),
            ],
        )
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
        kv_writes: list[tuple] = []

        async def kv_put(key, value):
            kv_writes.append((key, value))

        async def kv_get(key, default=None):
            return default

        llm = self._llm()
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
            kv_put=kv_put,
            kv_get=kv_get,
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
        self.assertFalse((await self._doc_meta(uri)).get("deprecated"))
        self.assertEqual(len(await self._active_docs()), 2)
        # KV（待审队列/提取日志）零写入，LLM 磁盘缓存目录为空
        self.assertEqual(kv_writes, [])
        self.assertEqual(list(cache_dir.iterdir()), [])
        # 缓冲表无新增行
        self.assertEqual(await self.mgr.inbox.status_counts(), {})
