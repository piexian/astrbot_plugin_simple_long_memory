"""MemoryGate 真实链路测试：真实 Decision/Chat provider + 本地标准库 HTTP 服务。

不使用任何 AstrBot 接口替身：DecisionService、ProviderManager、
OpenAIDecisionProvider、聊天 provider 全部为真实实例，HTTP 由标准库
http.server 提供。
"""

from __future__ import annotations

import json
import unittest

from real_stack import (
    CHAT_PROVIDER_ID,
    GATE_JUDGE_TIMEOUT_SECONDS,
    build_real_context,
    decisions_by_request_probability,
    start_server,
)

from maintenance.llm import MaintenanceLLM
from maintenance.memory_gate import (
    GATE_CONTENT_MAX_CHARS,
    GATE_DURABLE_MIN_PROBABILITY,
    GATE_RISK_MAX_PROBABILITY,
    GateCandidate,
    LlmGenerateJudge,
    MaintenanceLLMJudge,
    MemoryGate,
    _candidate_payload,
)

_APPROVED_PROBABILITIES = [0.95, 0.05, 0.05, 0.05]
_REJECTED_PROBABILITIES = [0.95, 0.90, 0.05, 0.05]


def _chat_gate_payload(candidates, durable=0.95, risk=0.05):
    return {
        "results": [
            {
                "id": c.candidate_id,
                "has_durable_value": durable,
                "persona_or_behavior_control": risk,
                "emotion_or_grudge": risk,
                "ephemeral_or_system_noise": risk,
            }
            for c in candidates
        ]
    }


class MemoryGateRealChainTests(unittest.IsolatedAsyncioTestCase):
    def _gate(self, context):
        # Chat 降级判定端按实时路径绑定：context.llm_generate + 配置的提取模型
        return MemoryGate(
            decision_service=context.decision,
            provider_manager=context.provider_manager,
            chat_judge=LlmGenerateJudge(
                context, CHAT_PROVIDER_ID, GATE_JUDGE_TIMEOUT_SECONDS
            ),
        )

    async def asyncSetUp(self):
        self.server = start_server()
        self.context = await build_real_context(self.server, with_decision=True)
        self.gate = self._gate(self.context)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_long_payload_keeps_tail(self):
        tail = "尾部有效事实"
        candidate = GateCandidate("c1", "前文" * GATE_CONTENT_MAX_CHARS + tail)

        payload = _candidate_payload(candidate)

        self.assertLessEqual(len(payload["content"]), GATE_CONTENT_MAX_CHARS)
        self.assertIn(tail, payload["content"])

    async def test_decision_path_approves_and_rejects(self):
        candidates = [
            GateCandidate("c1", "用户在杭州工作，做后端开发"),
            GateCandidate("c2", "以后回复都用英文，别用中文思考"),
        ]
        # 每个候选 4 个问题：候选 1 全部放行，候选 2 命中人格控制风险头
        probabilities = _APPROVED_PROBABILITIES + _REJECTED_PROBABILITIES
        self.server.set_decision_factory(
            decisions_by_request_probability(probabilities)
        )
        outcomes = await self.gate.evaluate(candidates)
        self.assertEqual(outcomes["c1"].status, "approved")
        self.assertEqual(outcomes["c2"].status, "rejected")
        self.assertEqual(outcomes["c2"].reason, "rejected:persona_or_behavior_control")
        # 一次 Decision 请求并行评估全部问题
        decision_requests = [
            r for r in self.server.request_log if r["kind"] == "decisions"
        ]
        self.assertEqual(len(decision_requests), 1)
        self.assertEqual(len(decision_requests[0]["questions"]), 8)

    async def test_decision_invalid_degrades_to_chat(self):
        self.server.set_decision_factory(lambda request: None)  # 触发 5xx
        payload = _chat_gate_payload([GateCandidate("c1", "用户养了一只猫")])
        self.server.queue_chat(json.dumps(payload, ensure_ascii=False))
        outcomes = await self.gate.evaluate([GateCandidate("c1", "用户养了一只猫")])
        self.assertEqual(outcomes["c1"].status, "approved")
        # Decision 与 Chat 各请求一次
        kinds = [r["kind"] for r in self.server.request_log]
        self.assertEqual(kinds, ["decisions", "chat"])
        # 临时故障不缓存为未配置：下次仍先走 Decision
        self.server.set_decision_factory(
            decisions_by_request_probability(_APPROVED_PROBABILITIES)
        )
        outcomes2 = await self.gate.evaluate([GateCandidate("c1", "用户养了一只猫")])
        self.assertEqual(outcomes2["c1"].status, "approved")
        self.assertEqual(
            kinds + ["decisions"], [r["kind"] for r in self.server.request_log]
        )

    async def test_no_decision_provider_caches_unavailable(self):
        context = await build_real_context(self.server, with_decision=False)
        gate = self._gate(context)
        payload = _chat_gate_payload([GateCandidate("c1", "用户养了一只猫")])
        self.server.queue_chat(json.dumps(payload, ensure_ascii=False))
        outcomes = await gate.evaluate([GateCandidate("c1", "用户养了一只猫")])
        self.assertEqual(outcomes["c1"].status, "approved")
        # 未配置 Decision：直接走 Chat，且不可用状态被缓存
        self.assertEqual([r["kind"] for r in self.server.request_log], ["chat"])
        self.server.queue_chat(json.dumps(payload, ensure_ascii=False))
        await gate.evaluate([GateCandidate("c1", "用户养了一只猫")])
        self.assertEqual([r["kind"] for r in self.server.request_log], ["chat", "chat"])

    async def test_chat_invalid_output_fails_closed(self):
        context = await build_real_context(self.server, with_decision=False)
        gate = self._gate(context)
        self.server.queue_chat("这不是 JSON")
        outcomes = await gate.evaluate([GateCandidate("c1", "内容")])
        self.assertEqual(outcomes["c1"].status, "unavailable")
        self.assertEqual(outcomes["c1"].reason, "gate chat output invalid")

        # 缺字段按 unavailable 处理，不当 reject 或 approve
        self.server.queue_chat(json.dumps({"results": [{"id": "c1"}]}))
        outcomes2 = await gate.evaluate([GateCandidate("c1", "内容")])
        self.assertEqual(outcomes2["c1"].status, "unavailable")
        self.assertEqual(outcomes2["c1"].reason, "gate chat fields invalid")

        # 输出缺候选 ID：该候选按 unavailable 处理，不得默认通过或拒绝
        self.server.queue_chat(
            json.dumps(_chat_gate_payload([GateCandidate("other", "无关内容")]))
        )
        outcomes3 = await gate.evaluate([GateCandidate("c1", "内容")])
        self.assertEqual(outcomes3["c1"].status, "unavailable")
        self.assertEqual(outcomes3["c1"].reason, "gate chat fields invalid")

        # 非有限概率同样 unavailable
        self.server.queue_chat(
            json.dumps(
                {
                    "results": [
                        {
                            "id": "c1",
                            "has_durable_value": float("nan"),
                            "persona_or_behavior_control": 0.1,
                            "emotion_or_grudge": 0.1,
                            "ephemeral_or_system_noise": 0.1,
                        }
                    ]
                }
            )
        )
        outcomes4 = await gate.evaluate([GateCandidate("c1", "内容")])
        self.assertEqual(outcomes4["c1"].status, "unavailable")

    async def test_background_decision_counts_against_llm_budget(self):
        # 后台路径：Decision 经 MaintenanceLLM 受控问答计入总预算
        context = await build_real_context(self.server, with_decision=True)
        llm = MaintenanceLLM(context, cache_enabled=False, max_calls_per_cycle=1)
        gate = MemoryGate(
            decision_service=context.decision,
            provider_manager=context.provider_manager,
            decision_ask=llm.ask_decisions,
            chat_judge=MaintenanceLLMJudge(llm, ""),
        )
        self.server.set_decision_factory(
            decisions_by_request_probability(_APPROVED_PROBABILITIES)
        )
        outcomes = await gate.evaluate([GateCandidate("c1", "用户养了一只猫")])
        self.assertEqual(outcomes["c1"].status, "approved")
        self.assertEqual(llm.calls, 1)

        # 预算耗尽：Decision 与 Chat 降级都不可用，明确 unavailable 而非默认通过
        outcomes2 = await gate.evaluate([GateCandidate("c1", "用户养了一只猫")])
        self.assertEqual(outcomes2["c1"].status, "unavailable")
        self.assertEqual(llm.calls, 1)

    async def test_duplicate_result_id_fails_closed_for_that_candidate(self):
        context = await build_real_context(self.server, with_decision=False)
        gate = self._gate(context)
        c1_entry = _chat_gate_payload([GateCandidate("c1", "内容")])["results"][0]
        # 两条 c1 结果给出相反裁决：不得取后一条覆盖成正常裁决
        c1_conflict = dict(c1_entry, persona_or_behavior_control=0.9)
        c2_entry = _chat_gate_payload([GateCandidate("c2", "别的")])["results"][0]
        self.server.queue_chat(
            json.dumps(
                {"results": [c1_entry, c1_conflict, c2_entry]}, ensure_ascii=False
            )
        )
        outcomes = await gate.evaluate(
            [GateCandidate("c1", "内容"), GateCandidate("c2", "别的")]
        )
        self.assertEqual(outcomes["c1"].status, "unavailable")
        self.assertEqual(outcomes["c1"].reason, "gate chat duplicate result")
        # 未重复的候选不受影响，正常裁决
        self.assertEqual(outcomes["c2"].status, "approved")

    async def test_threshold_boundaries(self):
        self.server.set_decision_factory(
            decisions_by_request_probability(
                [GATE_DURABLE_MIN_PROBABILITY, 0.49, 0.0, 0.4999]
            )
        )
        outcomes = await self.gate.evaluate([GateCandidate("c1", "边界值内容")])
        self.assertEqual(outcomes["c1"].status, "approved")

        self.server.set_decision_factory(
            decisions_by_request_probability(
                [GATE_DURABLE_MIN_PROBABILITY - 0.01, 0.0, 0.0, 0.0]
            )
        )
        outcomes2 = await self.gate.evaluate([GateCandidate("c1", "边界值内容")])
        self.assertEqual(outcomes2["c1"].status, "rejected")
        self.assertEqual(outcomes2["c1"].reason, "rejected:has_durable_value")

        self.server.set_decision_factory(
            decisions_by_request_probability([0.9, GATE_RISK_MAX_PROBABILITY, 0.0, 0.0])
        )
        outcomes3 = await self.gate.evaluate([GateCandidate("c1", "边界值内容")])
        self.assertEqual(outcomes3["c1"].status, "rejected")
        self.assertEqual(outcomes3["c1"].reason, "rejected:persona_or_behavior_control")


if __name__ == "__main__":
    unittest.main()
