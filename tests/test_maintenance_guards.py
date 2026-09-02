"""后台整理防失控护栏测试：矛盾检测上限、审核员逐条审核、待审队列去重/容量。"""

import json
import unittest

from maintenance.agents.analyst import AnalystAgent
from maintenance.agents.reviewer import ReviewerAgent
from maintenance.runner import AgentManifest, MaintenanceReport, MaintenanceRunner


def _mem(
    uri: str, content: str, created_at: str, scope: str = "personal", owner: str = "u1"
) -> dict:
    return {
        "uri": uri,
        "content": content,
        "metadata": {
            "memory_scope": scope,
            "owner_user_id": owner,
            "created_at": created_at,
        },
    }


class _FakeLLM:
    """模拟 MaintenanceLLM 的最小接口。"""

    def __init__(self, max_calls: int = 50, responses: list | None = None):
        self.max_calls_per_cycle = max_calls
        self.calls = 0
        self._cache_enabled = True
        self._responses = list(responses or [])
        self._cache: dict[str, dict] = {}
        self.prompts: list[str] = []

    @property
    def remaining_calls(self) -> int:
        return self.max_calls_per_cycle - self.calls

    def _resolve_model_id(self, model_id: str) -> str:
        return model_id or "default-model"

    def op_review_cache_key(self, op: dict, model_id: str) -> str:
        return json.dumps(
            {"model": model_id, "op": op}, ensure_ascii=False, sort_keys=True
        )

    def _cache_get(self, key: str):
        return self._cache.get(key)

    def _cache_put(self, key: str, value: dict) -> None:
        self._cache[key] = value

    def _parse_json(self, raw: str):
        try:
            return json.loads(raw)
        except Exception:
            return None

    async def _chat(self, system_prompt: str, user_prompt: str, model_id: str):
        self.calls += 1
        self.prompts.append(user_prompt)
        if self._responses:
            return self._responses.pop(0)
        return json.dumps({"verdict": "approve", "reason": "ok", "confidence": 0.9})

    def reset_cycle_stats(self) -> None:
        self.calls = 0

    def stats(self) -> dict[str, int]:
        return {"calls": self.calls, "cache_hits": 0, "errors": 0}


class _FakeMemoryMgr:
    def __init__(self, memories: dict | None = None):
        self._memories = memories or {}
        self.fetch_calls: list[str] = []

    async def _get_memory_by_uri(self, uri: str):
        self.fetch_calls.append(uri)
        return self._memories.get(uri)


class _FakeKV:
    def __init__(self):
        self.store: dict = {}

    async def get_kv_data(self, key, default=None):
        return self.store.get(key, default)

    async def put_kv_data(self, key, value):
        self.store[key] = value


class AnalystContradictionTests(unittest.TestCase):
    def _analyst(self, config: dict | None = None) -> AnalystAgent:
        return AnalystAgent(None, None, None, config or {})

    def test_respects_max_contradictions(self):
        analyst = self._analyst({"maintenance_analyst_max_contradictions": 3})
        pairs = [
            (
                _mem(f"u://{i}a", f"用户喜欢 苹果 香蕉 橙子 {i}", "2026-01-01"),
                _mem(f"u://{i}b", f"用户喜欢 苹果 香蕉 西瓜 {i}", "2026-02-01"),
                0.8,
            )
            for i in range(50)
        ]
        result = analyst._detect_contradictions(pairs)
        self.assertEqual(len(result), 3)

    def test_excludes_linked_pairs_both_directions(self):
        analyst = self._analyst()
        a = _mem("u://a", "用户 喜欢 苹果 香蕉 橙子", "2026-01-01")
        b = _mem("u://b", "用户 喜欢 苹果 香蕉 西瓜", "2026-02-01")
        result = analyst._detect_contradictions(
            [(a, b, 0.8)], exclude_pairs={("u://b", "u://a")}
        )
        self.assertEqual(result, [])

    def test_old_new_ordering_by_time(self):
        analyst = self._analyst()
        a = _mem("u://a", "用户 喜欢 苹果 香蕉 橙子", "2026-02-01")
        b = _mem("u://b", "用户 喜欢 苹果 香蕉 西瓜", "2026-01-01")
        result = analyst._detect_contradictions([(a, b, 0.8)])
        self.assertEqual(result[0]["old_uri"], "u://b")
        self.assertEqual(result[0]["new_uri"], "u://a")

    def test_same_scope_group_tenant_isolation(self):
        analyst = self._analyst()
        p1 = _mem("u://1", "a", "t", "personal", "u1")
        p2 = _mem("u://2", "b", "t", "personal", "u2")
        g1 = _mem("u://3", "c", "t", "group", "u1")
        glob = _mem("u://4", "d", "t", "global", "")
        self.assertFalse(analyst._same_scope_group(p1, p2))
        self.assertFalse(analyst._same_scope_group(p1, g1))
        self.assertTrue(analyst._same_scope_group(p1, _mem("u://5", "e", "t")))
        self.assertTrue(
            analyst._same_scope_group(glob, _mem("u://6", "f", "t", "global", ""))
        )


class ReviewerPerOpTests(unittest.IsolatedAsyncioTestCase):
    def _reviewer(
        self,
        llm: _FakeLLM,
        config: dict | None = None,
        memory_mgr: _FakeMemoryMgr | None = None,
    ) -> ReviewerAgent:
        return ReviewerAgent(None, memory_mgr or _FakeMemoryMgr(), llm, config or {})

    async def test_reviews_ops_one_by_one(self):
        llm = _FakeLLM()
        reviewer = self._reviewer(llm)
        ops = [
            {"type": "archive", "uri": "u://1", "reason": "r1"},
            {"type": "archive", "uri": "u://2", "reason": "r2"},
        ]
        verdicts = await reviewer.review(ops)
        self.assertEqual(len(verdicts), 2)
        self.assertEqual(llm.calls, 2)
        # 每次调用 prompt 只含单条操作
        for prompt in llm.prompts:
            self.assertIn("u://", prompt)
        self.assertNotIn("u://2", llm.prompts[0])

    async def test_cache_hit_skips_llm_call(self):
        llm = _FakeLLM()
        reviewer = self._reviewer(llm)
        op = {"type": "archive", "uri": "u://1", "reason": "r1"}
        key = llm.op_review_cache_key(op, "default-model")
        llm._cache[key] = {"verdict": "reject", "reason": "cached", "confidence": 0.8}
        verdicts = await reviewer.review([op])
        self.assertEqual(llm.calls, 0)
        self.assertEqual(verdicts[0]["verdict"], "reject")

    async def test_needs_context_triggers_related_retry(self):
        responses = [
            json.dumps(
                {"verdict": "approve", "confidence": 0.4, "needs_context": True}
            ),
            json.dumps({"verdict": "reject", "reason": "看后拒绝", "confidence": 0.9}),
        ]
        llm = _FakeLLM(responses=responses)
        reviewer = self._reviewer(llm)
        ops = [
            {"type": "merge", "uris": ["u://1", "u://2"], "reason": "r1"},
            {"type": "archive", "uri": "u://1", "reason": "r2"},
        ]
        verdicts = await reviewer.review(ops)
        self.assertEqual(llm.calls, 3)  # 第 1 条 2 次（补关联重审）+ 第 2 条 1 次
        self.assertEqual(verdicts[0]["verdict"], "reject")
        # 重审 prompt 注入了关联操作完整内容
        self.assertIn('"type": "archive"', llm.prompts[1])

    async def test_budget_exhausted_fails_closed(self):
        llm = _FakeLLM(max_calls=1)
        reviewer = self._reviewer(llm)
        ops = [{"type": "archive", "uri": f"u://{i}", "reason": "r"} for i in range(5)]
        verdicts = await reviewer.review(ops)
        self.assertEqual(len(verdicts), 1)  # 预算只够 1 条，其余 fail closed
        self.assertEqual(reviewer.last_unresolved["budget_exhausted"], 4)
        self.assertEqual(reviewer.last_unresolved["invalid_output"], 0)

    async def test_invalid_output_returns_no_verdict(self):
        llm = _FakeLLM(responses=["not json at all"])
        reviewer = self._reviewer(llm)
        verdicts = await reviewer.review([{"type": "archive", "uri": "u://1"}])
        self.assertEqual(verdicts, [])
        self.assertEqual(reviewer.last_unresolved["invalid_output"], 1)

    async def test_source_cache_shared_across_ops(self):
        mgr = _FakeMemoryMgr({"u://1": {"content": "记忆一", "metadata": {}}})
        llm = _FakeLLM()
        reviewer = self._reviewer(llm, memory_mgr=mgr)
        ops = [
            {"type": "merge", "uris": ["u://1", "u://2"], "reason": "r1"},
            {"type": "archive", "uri": "u://1", "reason": "r2"},
        ]
        verdicts = await reviewer.review(ops)
        self.assertEqual(len(verdicts), 2)
        # 两条操作都引用 u://1，但单次运行内只拉取一次
        self.assertEqual(mgr.fetch_calls.count("u://1"), 1)

    def test_related_map_by_shared_uri(self):
        ops = [
            {"type": "merge", "uris": ["u://1", "u://2"]},
            {"type": "archive", "uri": "u://1"},
            {"type": "new_link", "source": "u://9", "target": "u://10"},
        ]
        related = ReviewerAgent._build_related_map(ops)
        self.assertEqual(related[0], [1])
        self.assertEqual(related[1], [0])
        self.assertNotIn(2, related)

    def test_related_map_unions_across_uri_groups(self):
        # op0 触及 u://1 和 u://2，两个 URI 分组的关联必须合并而不是覆盖
        ops = [
            {"type": "merge", "uris": ["u://1", "u://2"]},
            {"type": "archive", "uri": "u://1"},
            {"type": "archive", "uri": "u://2"},
        ]
        related = ReviewerAgent._build_related_map(ops)
        self.assertEqual(related[0], [1, 2])
        self.assertEqual(related[1], [0])
        self.assertEqual(related[2], [0])

    def test_related_map_duplicate_uri_in_one_op(self):
        # 同一操作内重复 URI 不产生自关联，也不清掉真实关联
        ops = [
            {"type": "merge", "uris": ["u://1", "u://1"]},
            {"type": "archive", "uri": "u://1"},
        ]
        related = ReviewerAgent._build_related_map(ops)
        self.assertEqual(related[0], [1])
        self.assertEqual(related[1], [0])
        # 单独一条含重复 URI 的操作无任何关联
        self.assertEqual(ReviewerAgent._build_related_map(ops[:1]), {})


class PendingFlushTests(unittest.IsolatedAsyncioTestCase):
    def _runner(self, kv: _FakeKV, config: dict | None = None) -> MaintenanceRunner:
        return MaintenanceRunner(
            context=None,
            memory_mgr=_FakeMemoryMgr(),
            llm=_FakeLLM(),
            config=config or {},
            kv_put=kv.put_kv_data,
            kv_get=kv.get_kv_data,
        )

    async def test_dedupe_by_signature_across_flushes(self):
        kv = _FakeKV()
        runner = self._runner(kv)
        op = {"type": "archive", "uri": "u://1", "reason": "r"}
        await runner._flush_pending_review([(op, {"reason": "x"})], "s1")
        await runner._flush_pending_review([(op, {"reason": "x"})], "s2")
        queue = kv.store["maintenance_pending_review"]
        self.assertEqual(len(queue), 1)

    async def test_queue_max_drops_overflow(self):
        kv = _FakeKV()
        runner = self._runner(kv, {"maintenance_pending_queue_max": 2})
        items = [
            ({"type": "archive", "uri": f"u://{i}"}, {"reason": "x"}) for i in range(5)
        ]
        await runner._flush_pending_review(items, "s1")
        queue = kv.store["maintenance_pending_review"]
        self.assertEqual(len(queue), 2)

    async def test_capacity_counts_pending_only(self):
        kv = _FakeKV()
        # 已有 3 条终态条目，不应占容量
        kv.store["maintenance_pending_review"] = [
            {"id": i, "status": s, "op": {"type": "archive", "uri": f"u://old{i}"}}
            for i, s in enumerate(("approved", "rejected", "failed"), start=1)
        ]
        runner = self._runner(kv, {"maintenance_pending_queue_max": 2})
        items = [
            ({"type": "archive", "uri": f"u://new{i}"}, {"reason": "x"})
            for i in range(3)
        ]
        await runner._flush_pending_review(items, "s1")
        queue = kv.store["maintenance_pending_review"]
        # 终态被清理，新条目按 pending 容量只入 2 条
        self.assertEqual(len(queue), 2)
        self.assertTrue(all(it["status"] == "pending" for it in queue))

    async def test_id_monotonic_after_prune(self):
        kv = _FakeKV()
        # pending id 1、终态最大 id 5：新条目 id 必须 > 5，不能复用
        kv.store["maintenance_pending_review"] = [
            {
                "id": 1,
                "status": "pending",
                "op": {"type": "archive", "uri": "u://p1"},
            },
            {"id": 5, "status": "approved", "op": {"type": "archive", "uri": "u://t5"}},
        ]
        runner = self._runner(kv)
        await runner._flush_pending_review(
            [({"type": "archive", "uri": "u://new"}, {"reason": "x"})], "s1"
        )
        queue = kv.store["maintenance_pending_review"]
        ids = [it["id"] for it in queue]
        self.assertEqual(ids, [1, 6])
        self.assertEqual(kv.store["maintenance_pending_review_seq"], 6)

    async def test_approved_op_can_be_reproposed(self):
        kv = _FakeKV()
        op = {"type": "archive", "uri": "u://1", "reason": "r"}
        runner = self._runner(kv)
        await runner._flush_pending_review([(op, {"reason": "x"})], "s1")
        # 管理员批准执行后变为终态
        kv.store["maintenance_pending_review"][0]["status"] = "approved"
        # 同操作下周期重新提议：终态已清理，可再次入队
        await runner._flush_pending_review([(op, {"reason": "x"})], "s2")
        queue = kv.store["maintenance_pending_review"]
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["status"], "pending")
        self.assertEqual(queue[0]["id"], 2)

    async def test_single_kv_write_per_cycle(self):
        kv = _FakeKV()
        runner = self._runner(kv)
        writes = []
        orig_put = kv.put_kv_data

        async def counting_put(key, value):
            writes.append(key)
            await orig_put(key, value)

        runner._kv_put = counting_put
        items = [
            ({"type": "archive", "uri": f"u://{i}"}, {"reason": "x"}) for i in range(10)
        ]
        await runner._flush_pending_review(items, "s1")
        # 队列本体每周期只写一次（另有一次单调 id 计数器写入）
        self.assertEqual(writes.count("maintenance_pending_review"), 1)
        self.assertEqual(writes.count("maintenance_pending_review_seq"), 1)

    def test_op_signature_stable(self):
        op1 = {"type": "archive", "uri": "u://1", "reason": "r"}
        op2 = {"uri": "u://1", "reason": "r", "type": "archive"}
        self.assertEqual(
            MaintenanceRunner._op_signature(op1),
            MaintenanceRunner._op_signature(op2),
        )


class DryRunMaintenanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_cycle_dry_run_calls_purge_preview_and_skips_host_execution(self):
        memory_mgr = _FakeMemoryMgr()
        purge_calls: list[bool] = []
        executed = False

        async def purge_deprecated(*, after_days: int, dry_run: bool):
            purge_calls.append(dry_run)
            return {
                "candidates": 2,
                "purged": 0,
                "links_cleaned": 0,
                "dry_run": dry_run,
            }

        async def organizer():
            self.assertFalse(runner._llm._cache_enabled)
            return AgentManifest(
                agent_type="organizer", parsed=True, operations=[{"type": "archive"}]
            )

        async def analyst():
            return AgentManifest(
                agent_type="analyst", parsed=True, operations=[{"type": "new_link"}]
            )

        async def reviewer(organizer_manifest, analyst_manifest):
            self.assertTrue(organizer_manifest.parsed)
            self.assertTrue(analyst_manifest.parsed)
            return [{"index": 0, "verdict": "approve"}]

        async def execute_operations(report):
            nonlocal executed
            executed = True

        memory_mgr.purge_deprecated = purge_deprecated
        runner = MaintenanceRunner(None, memory_mgr, _FakeLLM(), {}, None, None)
        runner._run_organizer = organizer
        runner._run_analyst = analyst
        runner._run_reviewer = reviewer
        runner._execute_operations = execute_operations

        report = await runner.run_cycle(dry_run=True)

        self.assertTrue(report.dry_run)
        self.assertEqual(purge_calls, [True])
        self.assertEqual(report.purge_result["candidates"], 2)
        self.assertFalse(executed)
        self.assertEqual(report.executed_ops, 0)
        self.assertTrue(runner._llm._cache_enabled)

    async def test_agent_stages_disable_and_restore_llm_cache(self):
        memory_mgr = _FakeMemoryMgr()
        llm = _FakeLLM()
        runner = MaintenanceRunner(None, memory_mgr, llm, {}, None, None)
        cache_states: list[tuple[str, bool]] = []

        async def organizer():
            cache_states.append(("organizer", llm._cache_enabled))
            return AgentManifest(agent_type="organizer", parsed=True)

        async def analyst():
            cache_states.append(("analyst", llm._cache_enabled))
            return AgentManifest(agent_type="analyst", parsed=True)

        async def reviewer(organizer_manifest, analyst_manifest):
            self.assertTrue(organizer_manifest.parsed)
            self.assertTrue(analyst_manifest.parsed)
            cache_states.append(("reviewer", llm._cache_enabled))
            return [{"index": 0, "verdict": "approve"}]

        runner._run_organizer = organizer
        runner._run_analyst = analyst
        runner._run_reviewer = reviewer

        for stage in ("organizer", "analyst", "reviewer"):
            result = await runner.run_test_stage(stage)
            self.assertTrue(result["dry_run"])
            self.assertTrue(llm._cache_enabled)

        self.assertTrue(cache_states)
        self.assertTrue(all(not cache_enabled for _, cache_enabled in cache_states))

    async def test_purge_stage_uses_dry_run_and_rejects_unknown_stage(self):
        memory_mgr = _FakeMemoryMgr()
        purge_calls: list[bool] = []

        async def purge_deprecated(*, after_days: int, dry_run: bool):
            purge_calls.append(dry_run)
            self.assertFalse(runner._llm._cache_enabled)
            return {"candidates": 3, "purged": 0, "dry_run": dry_run}

        memory_mgr.purge_deprecated = purge_deprecated
        runner = MaintenanceRunner(None, memory_mgr, _FakeLLM(), {}, None, None)

        result = await runner.run_test_stage("purge")

        self.assertTrue(result["dry_run"])
        self.assertEqual(result["purge"]["candidates"], 3)
        self.assertEqual(purge_calls, [True])
        self.assertTrue(runner._llm._cache_enabled)
        with self.assertRaisesRegex(ValueError, "合法值"):
            await runner.run_test_stage("unknown")


class _DeprecatingMemoryMgr(_FakeMemoryMgr):
    """带 deprecate_memory 记录的 fake，用于矛盾执行路径。"""

    def __init__(self):
        super().__init__()
        self.deprecate_calls: list[tuple[str, str]] = []

    async def deprecate_memory(self, uri: str, reason: str = ""):
        self.deprecate_calls.append((uri, reason))
        return True


class ContradictionExecutionTests(unittest.IsolatedAsyncioTestCase):
    """矛盾闭环：approve 不自动执行转人工；管理员批准后真正废弃旧记忆。"""

    def _runner(self, memory_mgr, kv: _FakeKV) -> MaintenanceRunner:
        return MaintenanceRunner(
            context=None,
            memory_mgr=memory_mgr,
            llm=_FakeLLM(),
            config={},
            kv_put=kv.put_kv_data,
            kv_get=kv.get_kv_data,
        )

    def _report_with_approved_contradiction(self) -> MaintenanceReport:
        report = MaintenanceReport(session_id="s1")
        report.analyst_manifest = AgentManifest(
            agent_type="analyst",
            parsed=True,
            operations=[
                {
                    "type": "contradiction",
                    "old_uri": "u://old",
                    "new_uri": "u://new",
                    "reason": "偏好变更",
                }
            ],
        )
        report.reviewer_verdicts = [
            {"index": 0, "verdict": "approve", "reason": "确认矛盾"}
        ]
        return report

    async def test_approved_contradiction_goes_pending_without_execution(self):
        kv = _FakeKV()
        mgr = _DeprecatingMemoryMgr()
        runner = self._runner(mgr, kv)

        report = self._report_with_approved_contradiction()
        await runner._execute_operations(report)

        self.assertEqual(mgr.deprecate_calls, [])
        self.assertEqual(report.executed_ops, 0)
        self.assertEqual(report.skipped_ops, 1)
        queue = kv.store["maintenance_pending_review"]
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["op"]["type"], "contradiction")
        self.assertEqual(queue[0]["status"], "pending")
        self.assertEqual(queue[0]["verdict_reason"], "矛盾待人工裁决: 确认矛盾")

    async def test_execute_approved_contradiction_deprecates_old_uri(self):
        mgr = _DeprecatingMemoryMgr()
        runner = self._runner(mgr, _FakeKV())
        op = {
            "type": "contradiction",
            "old_uri": "u://old",
            "new_uri": "u://new",
            "reason": "偏好变更",
        }

        self.assertTrue(await runner.execute_approved(op))
        self.assertEqual(mgr.deprecate_calls, [("u://old", "偏好变更")])

    async def test_execute_approved_contradiction_requires_old_uri_and_defaults_reason(
        self,
    ):
        mgr = _DeprecatingMemoryMgr()
        runner = self._runner(mgr, _FakeKV())

        self.assertFalse(await runner.execute_approved({"type": "contradiction"}))
        self.assertEqual(mgr.deprecate_calls, [])

        self.assertTrue(
            await runner.execute_approved(
                {"type": "contradiction", "old_uri": "u://old"}
            )
        )
        self.assertEqual(mgr.deprecate_calls, [("u://old", "contradiction resolved")])


if __name__ == "__main__":
    unittest.main()
