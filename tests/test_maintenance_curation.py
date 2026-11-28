"""后台整理真实栈测试：巡检、分类、核验、审核、待审队列、合并与清理。

真实 AstrBot 4.30 组件 + 脚本化本地 HTTP 应答，不使用任何接口替身。
"""

from __future__ import annotations

import json
import unittest
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
QUEUE_KEY = "maintenance_pending_review"
OVERFLOW_KEY = "maintenance_pending_review_overflow"


def _json(payload) -> str:
    """序列化脚本化应答体。"""
    return json.dumps(payload, ensure_ascii=False)


def _gate_results(results: list[dict]) -> str:
    """清洗闸门应答。"""
    return _json({"results": results})


def _gate_result(uri: str, *, durable: float = 0.95, risk: float = 0.05) -> dict:
    return {
        "id": uri,
        "has_durable_value": durable,
        "persona_or_behavior_control": risk,
        "emotion_or_grudge": risk,
        "ephemeral_or_system_noise": risk,
    }


def _action(uri, action, reason, confidence=0.9) -> dict:
    """清洗阶段的单条判定。"""
    return {"id": uri, "action": action, "reason": reason, "confidence": confidence}


def _results(*entries) -> str:
    """清洗阶段应答。"""
    return _json({"results": list(entries)})


def _merge_result(merged_content: str, reason: str, *, index: str = "0") -> dict:
    """分析阶段 Chat 兜底的单条 merge 判定。"""
    return {
        "id": index,
        "verdict": "merge",
        "reason": reason,
        "confidence": 0.9,
        "merged_content": merged_content,
    }


def _verdict(verdict, reason, confidence=0.9, **extra) -> str:
    """审核员与判定端的单条裁决应答。"""
    payload = {"verdict": verdict, "reason": reason, "confidence": confidence, **extra}
    return _json(payload)


def _merge_verdict(reason, fused_text="", weight=0.9) -> str:
    """合并通道的 Chat 判定应答。"""
    merged = {"verdict": "merge", "reason": reason, "weight": weight}
    merged["fused_text"] = fused_text
    return _json(merged)


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    call = {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }
    return {"content": None, "tool_calls": [call]}


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


def _agent(agent_cls, case, llm, config: dict | None = None):
    """在真实 Context 与真实 KB 上构造整理代理。"""
    return agent_cls(
        context=case.context,
        memory_mgr=case.mgr,
        llm=llm,
        config={"maintenance_model_id": CHAT_PROVIDER_ID, **(config or {})},
    )


class _CurationCase(RealStackTestCase):
    """真实 Context + 真实 KB 上构造整理组件与 Runner 的公共方。"""

    def memory(self, uri, content, **meta):
        """构造 start_stack 需要的 (doc_id, content, metadata) 三元组。"""
        return (uri, content, make_memory_meta(uri, **meta))

    def runner(self, *, kv=False, llm=None, **config):
        """构造后台整理 Runner；kv=True 时读写真实插件 KV。"""
        extra = (
            {"kv_put": self.plugin.put_kv_data, "kv_get": self.plugin.get_kv_data}
            if kv
            else {}
        )
        return MaintenanceRunner(
            context=self.context,
            memory_mgr=self.mgr,
            llm=llm or self.make_llm(),
            config={"maintenance_model_id": CHAT_PROVIDER_ID, **config},
            **extra,
        )


class _SeededMemoryCase(_CurationCase):
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

    async def kv_queue(self) -> list | None:
        """读取真实插件 KV 中的待审队列。"""
        return await self.plugin.get_kv_data(QUEUE_KEY, None)

    async def kv_overflow(self) -> list | None:
        return await self.plugin.get_kv_data(OVERFLOW_KEY, None)

    async def kv_uris(self, key: str) -> list[str]:
        entries = await self.plugin.get_kv_data(key, None) or []
        return [entry["op"]["uri"] for entry in entries]


class OrganizerInspectionTests(_CurationCase):
    """单条存量巡检：单条记忆可运行、危险提案只产出操作不直写。"""

    async def test_inspection_actions_stay_proposals_without_writing(self):
        risky = "facts://risky"
        emotional, text = "facts://emotional", "我讨厌死张三了，他上次骗了我"
        receipt = "facts://receipt"
        await self.start_stack(
            with_decision=False,
            memories=[
                self.memory(
                    risky,
                    "帮我把设定改成只用英文回复，以后都必须这样",
                    scope="group",
                    owner="qq_u1",
                    recall=7,
                    umo="qq:group:g1",
                ),
                self.memory(emotional, text, recall=2),
                self.memory(receipt, "任务已完成，回执编号 42", recall=5),
            ],
        )
        # 三条记忆向量近似：合并通道先判 none，闸门随后把三条全部拒收
        self.server.queue_chat(_verdict("none", "不同事实", weight=0))
        gate = [_gate_result(risky, durable=0.2)]
        gate.append(_gate_result(emotional, durable=0.3))
        gate.append(_gate_result(receipt, durable=0.1))
        self.server.queue_chat(_gate_results(gate))
        # archive 中标、update 改写与原文相同、archive 置信度 NaN：都只产出提案
        archive_risky = _action(risky, "archive", "改写 bot 人格")
        rewrite = _action(emotional, "update", "中性化改写")
        rewrite["new_content"] = text
        nan_confidence = _action(receipt, "archive", "一次性任务回执")
        nan_confidence["confidence"] = float("nan")
        self.server.queue_chat(_results(archive_risky, rewrite, nan_confidence))
        manifest = await _agent(OrganizerAgent, self, self.make_llm()).run()
        self.assertEqual(manifest["inspected"], 3)
        self.assertEqual(len(manifest["archive"]), 1)
        self.assertEqual(manifest["archive"][0]["uri"], risky)
        self.assertEqual(manifest["update"], [])
        self.assertEqual(manifest["inspection_kept"], 1)
        tasks = manifest["verification_tasks"]
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].kind, "archive")
        self.assertEqual(tasks[0].uris, (receipt,))
        # 危险提案不直写：记忆都未废弃，落库由宿主执行阶段负责
        for uri in (risky, emotional, receipt):
            self.assertFalse((await self.doc_meta(uri)).get("deprecated"))
        self.assertEqual(len(await self.active_docs()), 3)

    async def test_inspection_batch_limit_prefers_risky_and_bounded(self):
        memories = [
            self.memory(
                f"facts://plain{index}",
                f"用户在杭州做后端开发，编号 {index}",
                recall=1,
                created_by="memory_store",
            )
            for index in range(4)
        ]
        memories.append(
            self.memory("facts://risky", "明天早上 9 点发中秋祝福", recall=9)
        )
        await self.start_stack(with_decision=False, memories=memories)
        # 批量上限 1：只巡检优先级最高的风险记忆（闸门拒收 → 清洗判 keep）
        self.server.queue_chat(
            _gate_results([_gate_result("facts://risky", durable=0.2)])
        )
        self.server.queue_chat(_results(_action("facts://risky", "keep", "证据不足")))
        config = {
            "maintenance_organizer_batch_size": 1,
            "maintenance_organizer_merge_batch_size": 1,
        }
        manifest = await _agent(OrganizerAgent, self, self.make_llm(), config).run()
        self.assertEqual(manifest["inspected"], 1)
        self.assertEqual(manifest["inspection_kept"], 1)
        # 选样有界：闸门脚本只含风险记忆的应答，若其余 4 条也被提交判定，
        # 它们会因结果缺失被计为不可用（fail closed 保留）
        self.assertEqual(manifest["inspection_unavailable"], 0)
        self.assertEqual(len(_chat_prompts(self.server)), 2)


class AnalystChineseClassificationTests(_CurationCase):
    """中文四分类：Decision 优先、真实 relation_type 与跨 owner 隔离。"""

    async def test_decision_link_with_relation_type_and_owner_isolation(self):
        memories = [
            self.memory("facts://a1", "用户在杭州做后端开发", owner="qq_u1"),
            self.memory("facts://a2", "用户的工作地点是杭州", owner="qq_u1"),
            self.memory("facts://b1", "用户在苏州做前端开发", owner="qq_u2"),
            self.memory("facts://b2", "用户的工作地点是苏州", owner="qq_u2"),
        ]
        await self.start_stack(with_decision=True, memories=memories)
        # 两对候选各一次 Decision 问答（relation + link_type）：
        # u1 对 link/supports，u2 对 none
        self.server.set_decision_factory(
            decisions_by_request_choices(
                [("link", 0.85), ("supports", 0.9), ("none", 0.9), ("related", 0.5)]
            )
        )
        manifest = await _agent(AnalystAgent, self, self.make_llm()).run()
        self.assertEqual(len(manifest["new_links"]), 1)
        link = manifest["new_links"][0]
        self.assertEqual(link["relation"], "supports")
        self.assertEqual(link["source"], "facts://a1")
        self.assertEqual(link["target"], "facts://a2")
        # 跨 owner 记忆不产生任何提案（租户隔离在预筛层生效）
        self.assertNotIn("facts://b1", {link["source"], link["target"]})
        self.assertEqual(manifest["decision_calls"], 2)
        self.assertEqual(manifest["chat_fallback_calls"], 0)
        decision_requests = [
            r for r in self.server.request_log if r["kind"] == "decisions"
        ]
        self.assertEqual(len(decision_requests), 2)
        self.assertEqual([len(r["questions"]) for r in decision_requests], [2, 2])
        self.assertEqual([r["kind"] for r in self.server.request_log].count("chat"), 0)

    async def test_decision_conflict_orients_by_time(self):
        memories = [
            self.memory(
                "facts://old",
                "用户的家乡是宁波",
                owner="qq_u1",
                created_at="2026-01-01T00:00:00+00:00",
            ),
            self.memory(
                "facts://new",
                "用户的家乡已经搬到杭州",
                owner="qq_u1",
                created_at="2026-02-01T00:00:00+00:00",
            ),
        ]
        await self.start_stack(with_decision=True, memories=memories)
        self.server.set_decision_factory(
            decisions_by_request_choices([("conflict", 0.9), ("related", 0.5)])
        )
        manifest = await _agent(AnalystAgent, self, self.make_llm()).run()
        self.assertEqual(len(manifest["contradictions"]), 1)
        conflict = manifest["contradictions"][0]
        self.assertEqual(conflict["old_uri"], "facts://old")
        self.assertEqual(conflict["new_uri"], "facts://new")

    async def test_chat_fallback_classifies_and_drops_blank_merge(self):
        memories = [
            self.memory("facts://c1", "用户喜欢喝手冲咖啡", owner="qq_u1"),
            self.memory("facts://c2", "用户每天早上都喝手冲咖啡", owner="qq_u1"),
        ]
        await self.start_stack(with_decision=False, memories=memories)
        merged_text = "用户喜欢手冲咖啡，每天早上都喝"
        self.server.queue_chat(
            _results(_merge_result(merged_text, "两条都是手冲咖啡偏好"))
        )
        manifest = await _agent(AnalystAgent, self, self.make_llm()).run()
        self.assertEqual(len(manifest["merges"]), 1)
        merge = manifest["merges"][0]
        self.assertEqual(merge["uris"], ["facts://c1", "facts://c2"])
        self.assertEqual(merge["merged_content"], merged_text)
        self.assertEqual(manifest["chat_fallback_calls"], 1)
        # 无效 relation_type 的 link 提案按 none 处理：不再有空格分词伪矛盾
        self.assertEqual(manifest["contradictions"], [])
        # merge 合成内容清理后为空 → 分类结果整体丢弃，不产出操作
        self.server.queue_chat(_results(_merge_result("   ", "同一偏好")))
        dropped = await _agent(AnalystAgent, self, self.make_llm()).run()
        self.assertEqual(dropped["merges"], [])
        self.assertEqual(dropped["chat_fallback_calls"], 1)


class PersonaAutoTests(_CurationCase):
    """persona_mode=auto 从真实 persona_manager 读取默认人格。"""

    async def test_persona_modes_auto_manual_and_off(self):
        await self.start_stack(with_decision=False)
        provider = PersonaSummaryProvider(self.context, {})
        summary = await provider.get()
        first_call = await provider.get()
        # DEFAULT_PERSONALITY 的真实 prompt 体现在摘要中，且按 UMO 缓存
        self.assertIn("helpful and friendly", summary)
        self.assertEqual(summary, first_call)
        self.assertLessEqual(len(summary), 700)
        manual = PersonaSummaryProvider(
            self.context, {"persona_mode": "manual", "persona_summary": "固定摘要"}
        )
        off = PersonaSummaryProvider(self.context, {"persona_mode": "off"})
        self.assertEqual(await manual.get(), "固定摘要")
        self.assertEqual(await off.get(), "")


class VerifierToolLoopTests(_CurationCase):
    """真实 tool_loop_agent 核验：工具权限、步数上限、预算与防重入。"""

    def _task(self, *, kind="archive", uris=("facts://v1",), umo=UMO_U1, cid=""):
        return VerificationTask(
            kind=kind, uris=tuple(uris), umo=umo, conversation_id=cid
        )

    def _queue_call(self, call_id, name, arguments):
        """入队一次真实工具调用请求。"""
        self.server.queue_chat_message(_tool_call(call_id, name, arguments))

    def _queue_proposal(self, call_id, kind, **proposal):
        """入队一次真实 submit_verification 工具调用。"""
        payload = {"proposal": {"kind": kind, **proposal}}
        self._queue_call(call_id, "submit_verification", payload)

    async def _seed_verification_context(self):
        await self.start_stack(
            with_decision=False,
            memories=[self.memory("facts://v1", "用户明天要在杭州开会", recall=5)],
        )
        self.cid = await self.context.conversation_manager.new_conversation(UMO_U1)
        await self.context.conversation_manager.add_message_pair(
            self.cid,
            {"role": "user", "content": "明天几点开会"},
            {"role": "assistant", "content": [{"type": "text", "text": "上午九点"}]},
        )

    async def test_tool_loop_reads_real_data_and_discards_out_of_scope(self):
        await self._seed_verification_context()
        llm = self.make_llm()
        verifier = _agent(MaintenanceVerifier, self, llm)
        task = self._task(cid=self.cid)
        self._queue_call("call_1", "get_memory_detail", {"uri": "facts://v1"})
        self._queue_call("call_2", "get_conversation_context", {})
        # 越界提案与合规提案同批提交：只保留允许范围内的提案
        self._queue_proposal(
            "call_3",
            "archive",
            uri="facts://not-allowed",
            reason="越界提案",
            confidence=0.9,
        )
        self._queue_proposal(
            "call_4",
            "archive",
            uri="facts://v1",
            reason="过期的相对时间事件",
            confidence=0.9,
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
        self.assertEqual(len(_chat_prompts(self.server)), 5)
        # 核验只读：记忆未被废弃
        self.assertFalse((await self.doc_meta("facts://v1")).get("deprecated"))

    async def test_update_and_conflict_proposals_convert_via_real_fetches(self):
        await self.start_stack(
            with_decision=False,
            memories=[
                self.memory(
                    "facts://v1",
                    "我讨厌死张三了，他上次骗了我",
                    owner="qq_u1",
                    created_at="2026-01-01T00:00:00+00:00",
                ),
                self.memory(
                    "facts://v2",
                    "用户后来和张三和解了",
                    owner="qq_u1",
                    created_at="2026-02-01T00:00:00+00:00",
                ),
            ],
        )
        task = self._task(kind="update", uris=("facts://v1", "facts://v2"))
        # 同一批提交：update 改写、conflict 定向，以及两类不得转成操作的提案
        self._queue_proposal(
            "call_1",
            "update",
            uri="facts://v1",
            new_content="用户对张三有负面评价，源于一次欺骗",
            reason="第一人称情绪宣泄，改写为中性陈述",
            confidence=0.9,
        )
        self._queue_proposal(
            "call_2",
            "conflict",
            uris=["facts://v1", "facts://v2"],
            reason="对张三的评价前后矛盾",
            confidence=0.85,
        )
        self._queue_proposal(
            "call_3",
            "archive",
            uri="facts://v1",
            reason="过期事件",
            confidence=float("nan"),
        )
        self._queue_proposal(
            "call_4",
            "merge",
            uris=["facts://v1", "facts://v2"],
            new_content="   ",
            reason="同一事实",
            confidence=0.9,
        )
        self.server.queue_chat("核验完成")
        verifier = _agent(MaintenanceVerifier, self, self.make_llm())
        proposals, executed = await verifier.run([task], allowance=1)
        self.assertEqual(executed, 1)
        # 异步查询链路完整：改写校验与冲突定向都基于真实记忆数据
        self.assertEqual([op["type"] for op in proposals], ["update", "contradiction"])
        update_op, conflict_op = proposals
        self.assertEqual(update_op["uri"], "facts://v1")
        self.assertEqual(update_op["new_content"], "用户对张三有负面评价，源于一次欺骗")
        self.assertEqual(conflict_op["old_uri"], "facts://v1")
        self.assertEqual(conflict_op["new_uri"], "facts://v2")

    async def test_budget_caps_and_internal_event_block_recollection(self):
        await self._seed_verification_context()
        # 预算不足一个保守预留额度：核验不启动，不留任何模型调用
        llm = self.make_llm(max_calls=VERIFICATION_BUDGET_RESERVE - 1)
        proposals, metrics = await self.runner(llm=llm)._run_verifications(
            [self._task()]
        )
        self.assertEqual(proposals, [])
        self.assertEqual(metrics["verification_needed"], 1)
        self.assertEqual(metrics["verification_executed"], 0)
        self.assertEqual(metrics["verification_budget_skipped"], 1)
        self.assertEqual(llm.calls, 0)
        self.assertEqual(_chat_prompts(self.server), [])
        # 第一条核验完成（单次模型调用即收口），第二条超出周期上限
        runner = self.runner(
            llm=self.make_llm(), maintenance_verification_max_per_cycle=1
        )
        self.server.queue_chat("无法核验")
        proposals, metrics = await runner._run_verifications(
            [self._task(), self._task(uris=("facts://v2",))]
        )
        self.assertEqual(metrics["verification_needed"], 2)
        self.assertEqual(metrics["verification_executed"], 1)
        self.assertEqual(metrics["verification_budget_skipped"], 1)
        self.assertEqual(len(_chat_prompts(self.server)), 1)
        self.assertEqual(runner._verification_used, 1)
        # 内部事件标记阻断回采：真实插件处理器不写入任何提取行
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
        # 模型调用被 max_steps 严格封顶：5 步工具调用 + 1 次强制收口
        self.server.request_log.clear()
        task = self._task()
        for index in range(VERIFICATION_MAX_STEPS):
            self._queue_call(
                f"call_{index}", "get_memory_detail", {"uri": "facts://v1"}
            )
        verifier = _agent(MaintenanceVerifier, self, self.make_llm())
        proposals, executed = await verifier.run([task], allowance=1)
        self.assertEqual(executed, 1)
        self.assertEqual(proposals, [])
        self.assertEqual(len(_chat_prompts(self.server)), VERIFICATION_BUDGET_RESERVE)


class ReviewerPerOpTests(_CurationCase):
    """逐条审核：单操作单调用、磁盘缓存、关联重审、预算与非法输出 fail closed。"""

    async def asyncSetUp(self):
        await self.start_stack(
            with_decision=False,
            memories=[
                self.memory("facts://r1", "用户在杭州做后端开发"),
                self.memory("facts://r2", "用户的常驻城市是杭州"),
                self.memory("facts://r3", "用户喜欢喝手冲咖啡"),
            ],
        )

    async def test_reviews_single_operation_per_call_then_hits_disk_cache(self):
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
        self.server.queue_chat(_verdict("reject", "证据不足"))
        self.server.queue_chat(_verdict("approve", "关联合理"))
        verdicts = await _agent(ReviewerAgent, self, self.make_llm()).review(ops)
        self.assertEqual([v["verdict"] for v in verdicts], ["reject", "approve"])
        prompts = _chat_prompts(self.server)
        self.assertEqual(len(prompts), 2)
        # 每次 prompt 只含当前一条操作，不夹带其他操作
        self.assertIn("facts://r1", prompts[0])
        self.assertNotIn("facts://r3", prompts[0])
        self.assertIn("facts://r3", prompts[1])
        # 磁盘缓存命中：同一操作跨实例不再调用模型
        cached = self.make_llm(cache_enabled=True)
        op = {"type": "archive", "uri": "facts://r1", "reason": "重复提案"}
        self.server.queue_chat(_verdict("reject", "保留"))
        first = await _agent(ReviewerAgent, self, cached).review([op])
        second = await _agent(ReviewerAgent, self, cached).review([op])
        self.assertEqual(first[0]["verdict"], "reject")
        self.assertEqual(second[0]["verdict"], first[0]["verdict"])
        self.assertEqual(len(_chat_prompts(self.server)), 3)
        cache_key = cached.op_review_cache_key(op, CHAT_PROVIDER_ID)
        self.assertTrue(Path(cached._cache_path(cache_key)).exists())

    async def test_needs_context_retries_with_merged_related_operations(self):
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
            _verdict("approve", "需要关联上下文", confidence=0.4, needs_context=True)
        )
        self.server.queue_chat(_verdict("reject", "看后拒绝"))
        for _ in range(2):
            self.server.queue_chat(_verdict("reject", "证据不足"))
        verdicts = await _agent(ReviewerAgent, self, self.make_llm()).review(ops)
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

    async def test_budget_exhaustion_and_invalid_output_fail_closed(self):
        llm = self.make_llm(max_calls=1)
        ops = [
            {"type": "archive", "uri": f"facts://r{index}", "reason": "预算"}
            for index in range(3)
        ]
        self.server.queue_chat(_verdict("reject", "保留"))
        reviewer = _agent(ReviewerAgent, self, llm)
        verdicts = await reviewer.review(ops)
        self.assertEqual(len(verdicts), 1)
        self.assertEqual(reviewer.last_unresolved["budget_exhausted"], 2)
        self.assertEqual(reviewer.last_unresolved["invalid_output"], 0)
        self.assertEqual(len(_chat_prompts(self.server)), 1)
        self.server.queue_chat("不是 JSON")
        invalid = _agent(ReviewerAgent, self, self.make_llm())
        verdicts = await invalid.review(
            [{"type": "archive", "uri": "facts://r1", "reason": "非法输出"}]
        )

        self.assertEqual(verdicts, [])
        self.assertEqual(invalid.last_unresolved["invalid_output"], 1)


class PendingFlushAndBudgetTests(_SeededMemoryCase):
    """待审队列：跨周期去重、容量溢出、终审重提、预算尾部转人工与日志脱敏。"""

    async def asyncSetUp(self):
        await self.start_stack(with_decision=False, with_plugin=True)
        # 每个用例独立 KV 命名空间，避免跨用例串数据
        self.plugin.plugin_id = f"test/{uuid.uuid4().hex}"
        await self.seed_memory("facts://q1", "用户在杭州做后端开发")
        await self.seed_memory("facts://q2", "用户的常驻城市是杭州")
        await self.seed_memory("facts://q3", "用户喜欢喝手冲咖啡")

    async def test_dedupes_by_signature_across_cycles_and_requeues_after_approval(self):
        runner = self.runner(kv=True)
        op = {"type": "archive", "uri": "facts://r1", "reason": "重复提案"}
        await runner._flush_pending_review([(op, {"reason": "x"})], "s1")
        await runner._flush_pending_review([(op, {"reason": "x"})], "s2")
        queue = await self.kv_queue()
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["status"], "pending")
        # 管理员批准后条目变终态并被清理，同操作可在下周期重新提议
        queue[0]["status"] = "approved"
        await self.plugin.put_kv_data(QUEUE_KEY, queue)
        await runner._flush_pending_review([(op, {"reason": "x"})], "s3")
        queue = await self.kv_queue()
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["status"], "pending")
        self.assertEqual(queue[0]["id"], 2)

    async def test_capacity_persists_overflow_and_counts_pending_only(self):
        runner = self.runner(kv=True, maintenance_pending_queue_max=2)
        items = [
            ({"type": "archive", "uri": f"facts://o{index}"}, {"reason": "x"})
            for index in range(5)
        ]
        await runner._flush_pending_review(items, "s1")
        self.assertEqual(await self.kv_uris(QUEUE_KEY), ["facts://o0", "facts://o1"])
        self.assertEqual(
            await self.kv_uris(OVERFLOW_KEY), ["facts://o2", "facts://o3", "facts://o4"]
        )
        # 每轮释放已批准名额，溢出条目按序回填，排空后待审队列归零
        for released, expected_queue, expected_overflow in (
            (1, ["facts://o1", "facts://o2"], ["facts://o3", "facts://o4"]),
            (2, ["facts://o3", "facts://o4"], []),
            (2, [], []),
        ):
            queue = await self.kv_queue()
            for item in queue[:released]:
                item["status"] = "approved"
            await self.plugin.put_kv_data(QUEUE_KEY, queue)
            await runner._flush_pending_review([], f"release-{released}")
            self.assertEqual(await self.kv_uris(QUEUE_KEY), expected_queue)
            self.assertEqual(await self.kv_uris(OVERFLOW_KEY), expected_overflow)
        # 已有 3 条终态条目不占容量：新提案按上限 2 条入队，其余进溢出队列
        settled = [
            {
                "id": index,
                "status": status,
                "op": {"type": "archive", "uri": f"facts://old{index}"},
            }
            for index, status in enumerate(("approved", "rejected", "failed"), start=1)
        ]
        await self.plugin.put_kv_data(QUEUE_KEY, settled)
        await runner._flush_pending_review(
            [
                ({"type": "archive", "uri": f"facts://new{index}"}, {"reason": "x"})
                for index in range(3)
            ],
            "s5",
        )
        queue = await self.kv_queue()
        self.assertEqual(
            await self.kv_uris(QUEUE_KEY), ["facts://new0", "facts://new1"]
        )
        self.assertTrue(all(item["status"] == "pending" for item in queue))
        self.assertEqual(await self.kv_uris(OVERFLOW_KEY), ["facts://new2"])

    async def test_budget_tail_operations_go_to_human_review(self):
        runner = self.runner(kv=True)
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
        queue = await self.kv_queue()
        self.assertEqual(await self.kv_uris(QUEUE_KEY), ["facts://q2", "facts://q3"])
        self.assertTrue(all(item["status"] == "pending" for item in queue))
        self.assertEqual(queue[0]["verdict_reason"], "审核预算耗尽，转人工审核")

    async def test_execution_and_review_logs_hide_memory_bodies(self):
        runner = self.runner(kv=True)
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
                agent_type="organizer", parsed=True, operations=[merge_op, disputed_op]
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
        queue = await self.kv_queue()
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

    async def test_approved_contradiction_waits_for_human_then_deprecates_old(self):
        runner = self.runner(kv=True)
        # 目标信息不全时直接拒绝，不改动任何记忆
        self.assertFalse(await runner.execute_approved({"type": "contradiction"}))
        self.assertFalse(
            await runner.execute_approved(
                {"type": "contradiction", "old_uri": "facts://missing"}
            )
        )
        self.assertFalse((await self.doc_meta("facts://old"))["deprecated"])
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
        queue = await self.kv_queue()
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["op"]["type"], "contradiction")
        self.assertEqual(queue[0]["status"], "pending")
        self.assertEqual(queue[0]["verdict_reason"], "矛盾待人工裁决: 确认矛盾")
        self.assertTrue(await runner.execute_approved(queue[0]["op"]))
        self.assertTrue((await self.doc_meta("facts://old"))["deprecated"])
        self.assertFalse((await self.doc_meta("facts://new"))["deprecated"])
        # 未给出理由时使用默认裁决理由，仍然真实废弃旧记忆
        self.assertTrue(
            await runner.execute_approved(
                {"type": "contradiction", "old_uri": "facts://maybe"}
            )
        )
        meta = await self.doc_meta("facts://maybe")
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
            remaining, {("facts://old", "facts://new"), ("facts://old", "facts://dep")}
        )
        self.assertEqual(repeated, 0)


class OrganizerMergeAndExecuteTests(_SeededMemoryCase):
    """合并通道 URI 消费互斥与真实 merge 落地（含预验证拒绝）。"""

    async def _merged_metadata(self) -> list[dict]:
        docs = [_doc_metadata(doc) for doc in await self.active_docs()]
        return [meta for meta in docs if meta.get("merged_from")]

    async def test_merge_channel_consumes_uris_and_skips_them_in_inspection(self):
        await self.start_stack(with_decision=True)
        # 三条同归属记忆使用相同向量，余弦 1.0 → 预筛出全部三对候选
        self.server.set_embedding_vectors({"住在杭州": [0.5] * 8})
        await self.seed_memory("facts://m1", "用户住在杭州")
        await self.seed_memory("facts://m2", "用户的居住地是杭州")
        await self.seed_memory("facts://m3", "用户目前住在杭州")
        self.server.queue_chat(_merge_verdict("同一居住地事实", "用户住在杭州"))
        # 巡检只应看到未被合并消费的 m3：闸门拒收 → 清洗判 keep
        self.server.set_decision_factory(
            decisions_by_request_probability([0.1, 0.05, 0.05, 0.05])
        )
        self.server.queue_chat(_results(_action("facts://m3", "keep", "证据不足")))
        manifest = await _agent(OrganizerAgent, self, self.make_llm()).run()
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

    async def test_execute_approved_merge_rejects_bad_sources_then_writes_new_memory(
        self,
    ):
        await self.start_stack(with_decision=False)
        await self.seed_memory("facts://m1", "用户住在杭州")
        await self.seed_memory("facts://m2", "用户的居住地是杭州")
        await self.seed_memory("facts://m3", "用户喜欢喝手冲咖啡")
        await self.seed_memory("facts://m4", "用户每天早上都喝手冲咖啡")
        runner = self.runner()
        merge = {"type": "merge", "merged_content": "合并后内容", "reason": "同义合并"}
        # 源 URI 不存在 → 预验证拒绝，源记忆不动、不写合并记录
        self.assertFalse(
            await runner.execute_approved(
                {**merge, "uris": ["facts://missing", "facts://m3"]}
            )
        )
        self.assertFalse((await self.doc_meta("facts://m3"))["deprecated"])
        self.assertEqual(len(await self.active_docs()), 4)
        self.assertEqual(await self._merged_metadata(), [])
        # 源 URI 已 deprecated → 同样拒绝，另一条源保持活跃
        self.assertTrue(await self.mgr.deprecate_memory("facts://m3", reason="过时"))
        self.assertFalse(
            await runner.execute_approved(
                {**merge, "uris": ["facts://m3", "facts://m4"]}
            )
        )
        self.assertFalse((await self.doc_meta("facts://m4"))["deprecated"])
        self.assertEqual(len(await self.active_docs()), 3)
        self.assertEqual(await self._merged_metadata(), [])
        # 两条活跃源才真正落地：新记忆 + supersedes 边 + 源记忆废弃
        self.assertTrue(
            await runner.execute_approved(
                {**merge, "uris": ["facts://m1", "facts://m2"]}
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
        self.assertEqual(len(await self.active_docs()), 2)


class PurgeDeprecatedTests(_SeededMemoryCase):
    """物理清理：dry-run 只统计，正式执行按宽限期删文档与关联边。"""

    async def asyncSetUp(self):
        await self.start_stack(with_decision=False)
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
            meta = {"deprecated": deprecated}
            if deprecated:
                meta["deprecated_at"] = (now - timedelta(days=days)).isoformat()
            self.doc_ids[uri] = await self.seed_memory(uri, f"{uri} 的记忆正文", **meta)
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
        self.assertEqual(self.mgr.vec_db.embedding_storage.index.ntotal, 3)
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
        self.assertEqual(self.mgr.vec_db.embedding_storage.index.ntotal, 2)
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
                self.memory(uri, "任务已完成，回执编号 42", owner="qq_u1", recall=4),
                self.memory(
                    "facts://n2",
                    "用户在杭州做后端开发",
                    owner="qq_u1",
                    created_by="memory_store",
                ),
            ],
        )
        self.plugin.plugin_id = f"test/{uuid.uuid4().hex}"
        # Decision 未配置脚本应答 → 各阶段降级 Chat JSON
        self.server.set_decision_factory(lambda request: None)
        # merge 预筛对 → 不合并；闸门拒收回执记忆 → 清洗 archive；分析 none；审核 approve
        self.server.queue_chat(_verdict("none", "不同事实", weight=0))
        self.server.queue_chat(
            _gate_results(
                [_gate_result(uri, durable=0.1, risk=0.2), _gate_result("facts://n2")]
            )
        )
        self.server.queue_chat(_results(_action(uri, "archive", "一次性任务回执")))
        self.server.queue_chat(
            _results({"id": "0", "verdict": "none", "reason": "不相关"})
        )
        self.server.queue_chat(_verdict("approve", "证据充分"))
        llm = self.make_llm()
        cache_dir = Path(llm._cache_dir)
        runner = self.runner(
            kv=True, llm=llm, maintenance_max_llm_calls=50, auto_purge_enabled=False
        )
        report = await runner.run_cycle(dry_run=True)
        # 管线真实执行并产出 archive 提案，但阶段 4 全部跳过
        self.assertTrue(report.dry_run)
        metrics = report.organizer_manifest.metrics
        organizer_ops = report.organizer_manifest.operations
        self.assertEqual(len(organizer_ops), 1)
        self.assertEqual(organizer_ops[0]["type"], "archive")
        self.assertEqual(metrics["inspected"], 2)
        self.assertEqual(metrics["op_archive"], 1)
        self.assertIn("verification_needed", metrics)
        self.assertEqual(report.executed_ops, 0)
        self.assertEqual(report.skipped_ops, 0)
        # 记忆未被废弃、内容不变
        self.assertFalse((await self.doc_meta(uri)).get("deprecated"))
        self.assertEqual(len(await self.active_docs()), 2)
        # KV（待审队列/提取日志）零写入，LLM 磁盘缓存目录为空
        self.assertIsNone(await self.kv_queue())
        self.assertIsNone(await self.kv_overflow())
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
        runner = self.runner(llm=llm)
        self.server.queue_chat(_merge_verdict("同一事实"))
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


if __name__ == "__main__":
    unittest.main()
