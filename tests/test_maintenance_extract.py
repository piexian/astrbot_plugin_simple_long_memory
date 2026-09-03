"""夜间对话提取测试：整理师三态、runner 提取集成、游标提交、create 执行、过期 carry。"""

import copy
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from test_maintenance_segmenter import (
    _KV,
    INCOMPLETE,
    _config,
    _row,
    _StubSegmenter,
)
from test_maintenance_segmenter import (
    _FakeLLM as _SegFakeLLM,
)
from test_memory_commands import plugin_main

import maintenance.runner as runner_module
from maintenance.agents.curator import CuratorAgent
from maintenance.agents.reviewer import ReviewerAgent
from maintenance.agents.segmenter import ConversationBlock
from maintenance.runner import MaintenanceRunner


def _block(
    text="这是一段有信息量的对话内容，讨论了咖啡偏好并得出结论。",
    conv_key="qq:qq:private:u1",
    umo="qq:private:u1",
    first_id=1,
    last_id=10,
    truncated=False,
):
    return ConversationBlock(
        conv_key=conv_key,
        platform_id="qq",
        umo=umo,
        text=text,
        message_count=3,
        char_count=len(text),
        truncated=truncated,
        first_id=first_id,
        last_id=last_id,
    )


class _FakeRetrieveResult:
    def __init__(self, text, metadata):
        self.data = {"text": text, "metadata": metadata}
        self.similarity = 0.9


class _FakeVecDB:
    def __init__(self, results=None, fail=False):
        self._results = list(results or [])
        self._fail = fail
        self.calls = []

    async def retrieve(self, query, k, rerank, metadata_filters):
        self.calls.append(
            {
                "query": query,
                "k": k,
                "rerank": rerank,
                "metadata_filters": metadata_filters,
            }
        )
        if self._fail:
            raise RuntimeError("retrieve boom")
        return list(self._results)


class _CuratorMemoryMgr:
    def __init__(self, vec_db=None):
        self.vec_db = vec_db or _FakeVecDB()


class _CuratorLLM:
    """预设 chat_json 返回序列的 fake llm。"""

    def __init__(self, responses=None, max_calls=1000):
        self._responses = list(responses or [])
        self._max = max_calls
        self.calls = 0
        self.model_ids = []

    @property
    def remaining_calls(self):
        return self._max - self.calls

    async def chat_json(self, system, user, model_id=""):
        self.calls += 1
        self.model_ids.append(model_id)
        if self._responses:
            return self._responses.pop(0)
        return {"memories": [], "updates": [], "notes": ""}


def _curator(llm, vec_db=None, config=None):
    return CuratorAgent(
        context=None,
        memory_mgr=_CuratorMemoryMgr(vec_db),
        llm=llm,
        config=config or {},
    )


class CuratorAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_three_outcomes_create_update_nothing(self):
        llm = _CuratorLLM(
            responses=[
                {
                    "memories": [
                        {
                            "scope": "personal",
                            "type": "preference",
                            "content": "用户喜欢美式咖啡",
                            "subject": "u1",
                            "subjects": ["u1"],
                            "entities": ["咖啡"],
                            "topics": ["饮食"],
                            "disclosure": "讨论咖啡时",
                            "importance": 4,
                        }
                    ],
                    "updates": [],
                    "notes": "n1",
                },
                {
                    "memories": [],
                    "updates": [
                        {
                            "uri": "facts://old1",
                            "new_content": "用户现在改喝拿铁",
                            "reason": "偏好变更",
                        }
                    ],
                    "notes": "",
                },
                {"memories": [], "updates": [], "notes": ""},
            ]
        )
        vec_db = _FakeVecDB(
            results=[
                _FakeRetrieveResult(
                    "memory: 用户喜欢美式咖啡",
                    {"uri": "facts://old1", "memory_content": "用户喜欢美式咖啡"},
                )
            ]
        )
        curator = _curator(llm, vec_db)
        blocks = [
            _block(last_id=10),
            _block(last_id=20),
            _block(last_id=30),
        ]
        result = await curator.run(blocks, 10)

        self.assertEqual(
            [o for _, o in result["outcomes"]], ["created", "updated", "nothing"]
        )
        self.assertEqual(result["blocks_processed"], 3)
        self.assertEqual(result["blocks_nothing"], 1)
        self.assertEqual(result["llm_calls"], 3)
        self.assertEqual(result["notes"], "n1")

        create = result["create"][0]
        self.assertEqual(create["type"], "create")
        self.assertEqual(create["content"], "用户喜欢美式咖啡")
        self.assertEqual(create["domain"], "preference")
        self.assertEqual(create["scope"], "personal")
        self.assertEqual(create["subject"], "u1")
        self.assertEqual(create["subjects"], ["u1"])
        self.assertEqual(create["entities"], ["咖啡"])
        self.assertEqual(create["topics"], ["饮食"])
        self.assertEqual(create["disclosure"], "讨论咖啡时")
        self.assertEqual(create["importance"], 4)
        self.assertEqual(create["umo"], "qq:private:u1")
        self.assertFalse(create["truncated_block"])
        self.assertLessEqual(len(create["source_excerpt"]), 200)

        update = result["update"][0]
        self.assertEqual(
            update,
            {
                "type": "update",
                "uri": "facts://old1",
                "new_content": "用户现在改喝拿铁",
                "reason": "偏好变更",
            },
        )

    async def test_retrieve_filters_and_failure_fallback(self):
        llm = _CuratorLLM()
        vec_db = _FakeVecDB()
        curator = _curator(llm, vec_db)
        block = _block(conv_key="qq:qq:group:g1", umo="qq:group:g1")
        await curator.run([block], 5)

        call = vec_db.calls[0]
        self.assertEqual(call["query"], block.text[:500])
        self.assertEqual(call["k"], 5)
        self.assertEqual(call["rerank"], False)
        self.assertEqual(
            call["metadata_filters"],
            {
                "is_memory_record": True,
                "deprecated": False,
                "owner_session_id": "qq_g1",
            },
        )

        # 检索失败不阻断提取，按无旧记忆继续
        llm2 = _CuratorLLM(
            responses=[
                {
                    "memories": [
                        {"scope": "group", "type": "fact", "content": "群规一二三"}
                    ],
                    "updates": [],
                }
            ]
        )
        curator2 = _curator(llm2, _FakeVecDB(fail=True))
        result = await curator2.run([block], 5)
        self.assertEqual(len(result["create"]), 1)
        self.assertEqual([o for _, o in result["outcomes"]], ["created"])

    async def test_update_uri_not_in_candidates_dropped(self):
        llm = _CuratorLLM(
            responses=[
                {
                    "memories": [],
                    "updates": [
                        {
                            "uri": "facts://ghost",
                            "new_content": "幻觉引用",
                            "reason": "x",
                        }
                    ],
                }
            ]
        )
        vec_db = _FakeVecDB(results=[_FakeRetrieveResult("t", {"uri": "facts://old1"})])
        curator = _curator(llm, vec_db)
        result = await curator.run([_block()], 5)
        self.assertEqual(result["update"], [])
        self.assertEqual([o for _, o in result["outcomes"]], ["nothing"])

    async def test_global_scope_downgraded_to_personal(self):
        llm = _CuratorLLM(
            responses=[
                {
                    "memories": [
                        {"scope": "global", "type": "fact", "content": "试图写全局"}
                    ],
                    "updates": [],
                }
            ]
        )
        curator = _curator(llm)
        result = await curator.run([_block()], 5)
        self.assertEqual(result["create"][0]["scope"], "personal")

    async def test_smalltalk_produces_nothing(self):
        llm = _CuratorLLM(responses=[{"memories": [], "updates": [], "notes": ""}])
        curator = _curator(llm)
        result = await curator.run([_block(text="嗯嗯 好的 哈哈")], 5)
        self.assertEqual(result["create"], [])
        self.assertEqual(result["update"], [])
        self.assertEqual(result["blocks_nothing"], 1)
        self.assertEqual([o for _, o in result["outcomes"]], ["nothing"])

    async def test_source_excerpt_collapsed_and_capped(self):
        text = ("  多  空   格\n换行\t" + "聊" * 400) * 2
        llm = _CuratorLLM(
            responses=[
                {
                    "memories": [{"scope": "personal", "type": "fact", "content": "c"}],
                    "updates": [],
                }
            ]
        )
        curator = _curator(llm)
        result = await curator.run([_block(text=text)], 5)
        excerpt = result["create"][0]["source_excerpt"]
        self.assertLessEqual(len(excerpt), 200)
        self.assertNotIn("\n", excerpt)
        self.assertNotIn("  ", excerpt)

    async def test_group_personal_without_subject_dropped(self):
        llm = _CuratorLLM(
            responses=[
                {
                    "memories": [
                        {"scope": "personal", "type": "fact", "content": "群友某事"}
                    ],
                    "updates": [],
                }
            ]
        )
        curator = _curator(llm)
        block = _block(conv_key="qq:qq:group:g1", umo="qq:group:g1")
        result = await curator.run([block], 5)
        self.assertEqual(result["create"], [])
        self.assertEqual([o for _, o in result["outcomes"]], ["nothing"])

    async def test_budget_exhaustion_marks_skipped(self):
        llm = _CuratorLLM()
        curator = _curator(llm)
        blocks = [_block(last_id=10), _block(last_id=20)]
        result = await curator.run(blocks, 0)
        self.assertEqual(
            [o for _, o in result["outcomes"]], ["skipped_budget", "skipped_budget"]
        )
        self.assertEqual(llm.calls, 0)
        self.assertEqual(result["blocks_processed"], 0)

        # 全局 remaining_calls 约束同样生效
        llm2 = _CuratorLLM(max_calls=0)
        curator2 = _curator(llm2)
        result2 = await curator2.run([_block()], 10)
        self.assertEqual([o for _, o in result2["outcomes"]], ["skipped_budget"])
        self.assertEqual(llm2.calls, 0)

    async def test_llm_failure_counts_as_nothing(self):
        llm = _CuratorLLM(responses=[None])
        curator = _curator(llm)
        result = await curator.run([_block()], 5)
        self.assertEqual([o for _, o in result["outcomes"]], ["nothing"])
        self.assertEqual(result["blocks_nothing"], 1)

    async def test_curator_model_id_from_config(self):
        llm = _CuratorLLM()
        curator = _curator(llm, config={"maintenance_curator_model_id": "m-curator"})
        await curator.run([_block()], 5)
        self.assertEqual(llm.model_ids, ["m-curator"])


class _RunnerFakeLLM:
    def __init__(self, max_calls=50):
        self._max = max_calls
        self.calls = 0
        self._cache_enabled = True

    @property
    def remaining_calls(self):
        return self._max - self.calls

    def reset_cycle_stats(self):
        self.calls = 0

    def stats(self):
        return {"calls": self.calls, "cache_hits": 0, "errors": 0}


class _StoreMemoryMgr:
    def __init__(self):
        self.vec_db = _FakeVecDB()
        self.stored = []

    async def store_memory(self, **kwargs):
        self.stored.append(kwargs)
        return "facts://created-1"


class _KVLog:
    def __init__(self, initial=None):
        self.store = copy.deepcopy(initial or {})
        self.put_log = []

    async def get_kv_data(self, key, default=None):
        return self.store.get(key, default)

    async def put_kv_data(self, key, value):
        self.put_log.append((key, copy.deepcopy(value)))
        self.store[key] = value


class _FakeSegmenter:
    def __init__(self, result):
        self._result = result
        self.budgets = []

    async def collect_blocks(self, max_llm_calls):
        self.budgets.append(max_llm_calls)
        return self._result


class _FakeCurator:
    def __init__(self, create=None, update=None, llm_calls=0, outcomes=None):
        self._create = create or []
        self._update = update or []
        self._llm_calls = llm_calls
        self._outcomes = outcomes
        self.budgets = []

    async def run(self, blocks, max_llm_calls):
        self.budgets.append(max_llm_calls)
        outcomes = self._outcomes
        if outcomes is None:
            outcomes = [(b, "created") for b in blocks]
        return {
            "create": self._create,
            "update": self._update,
            "notes": "",
            "blocks_processed": len(blocks),
            "blocks_nothing": 0,
            "llm_calls": self._llm_calls,
            "outcomes": outcomes,
        }


def _seg_result(blocks, llm_calls=0, cursor_updates=None, carry_updates=None):
    return {
        "blocks": blocks,
        "cursor_updates": cursor_updates or {},
        "carry_updates": carry_updates or {},
        "stats": {"llm_calls": llm_calls, "blocks_emitted": len(blocks)},
    }


class RunnerExtractionTests(unittest.IsolatedAsyncioTestCase):
    def _runner(self, memory_mgr=None, kv=None, config=None):
        cfg = {
            "auto_purge_enabled": False,
            "maintenance_organizer_enabled": False,
            "maintenance_analyst_enabled": False,
            "maintenance_reviewer_enabled": False,
            "maintenance_extract_enabled": True,
            "maintenance_max_llm_calls": 50,
        }
        cfg.update(config or {})
        return MaintenanceRunner(
            context=None,
            memory_mgr=memory_mgr or _StoreMemoryMgr(),
            llm=_RunnerFakeLLM(),
            config=cfg,
            kv_put=kv.put_kv_data if kv else None,
            kv_get=kv.get_kv_data if kv else None,
        )

    def _patched(self, seg, cur):
        return [
            patch.object(runner_module, "SegmenterAgent", lambda **kw: seg),
            patch.object(runner_module, "CuratorAgent", lambda **kw: cur),
        ]

    async def test_budget_split_60_40(self):
        blocks = [_block()]
        seg = _FakeSegmenter(_seg_result(blocks, llm_calls=2))
        cur = _FakeCurator()
        runner = self._runner()
        for p in self._patched(seg, cur):
            p.start()
        try:
            manifest = await runner._run_extraction(dry_run=True)
        finally:
            for p in self._patched(seg, cur):
                p.stop()
        self.assertEqual(seg.budgets, [30])  # int(50 * 0.6)
        self.assertEqual(cur.budgets, [28])  # 30 - 分段员实际 2
        self.assertTrue(manifest.parsed)
        self.assertEqual(manifest.metrics["blocks"], 1)
        self.assertEqual(manifest.metrics["llm_calls"], 2)

    async def test_cursor_commit_per_block_and_carry_clear(self):
        b1 = _block(conv_key="qq:a", umo="qq:private:a", last_id=10)
        b2 = _block(conv_key="qq:a", umo="qq:private:a", first_id=11, last_id=20)
        b3 = _block(conv_key="qq:b", umo="qq:private:b", last_id=5)
        kv = _KVLog(
            {
                "maintenance_extract_carry": {
                    "qq:a": {"start_id": 1, "cycles": 1, "fail_count": 0},
                    "qq:z": {"start_id": 3, "cycles": 0, "fail_count": 1},
                }
            }
        )
        seg = _FakeSegmenter(
            _seg_result(
                [b1, b2, b3],
                cursor_updates={"qq:a": 20, "qq:b": 5, "qq:c": 7},
                carry_updates={"qq:a": None},
            )
        )
        cur = _FakeCurator(
            outcomes=[(b1, "created"), (b2, "nothing"), (b3, "skipped_budget")]
        )
        runner = self._runner(kv=kv)
        for p in self._patched(seg, cur):
            p.start()
        try:
            await runner._run_extraction(dry_run=False)
        finally:
            for p in self._patched(seg, cur):
                p.stop()

        cursor_writes = [v for k, v in kv.put_log if k == "maintenance_extract_cursors"]
        # 逐块提交：a 先推进到 10 再到 20；b 因 skipped_budget 不推进；
        # c 无产出块，直接采用分段员游标 7
        self.assertEqual(cursor_writes[0]["qq:a"]["id"], 10)
        self.assertEqual(cursor_writes[1]["qq:a"]["id"], 20)
        self.assertNotIn("qq:b", kv.store["maintenance_extract_cursors"])
        self.assertEqual(kv.store["maintenance_extract_cursors"]["qq:c"]["id"], 7)
        self.assertEqual(len(cursor_writes), 3)
        # carry None 清除 a，未提及的 z 保留
        carry = kv.store["maintenance_extract_carry"]
        self.assertNotIn("qq:a", carry)
        self.assertIn("qq:z", carry)

    async def test_dry_run_zero_kv_zero_store(self):
        blocks = [_block()]
        seg = _FakeSegmenter(
            _seg_result(blocks, cursor_updates={"qq:qq:private:u1": 10})
        )
        cur = _FakeCurator(
            create=[{"type": "create", "content": "x", "umo": "qq:private:u1"}]
        )
        kv = _KVLog()
        mgr = _StoreMemoryMgr()
        runner = self._runner(memory_mgr=mgr, kv=kv)
        for p in self._patched(seg, cur):
            p.start()
        try:
            report = await runner.run_cycle(dry_run=True)
        finally:
            for p in self._patched(seg, cur):
                p.stop()
        self.assertIsNotNone(report.extract_manifest)
        self.assertEqual(len(report.extract_manifest.operations), 1)
        self.assertEqual(kv.put_log, [])
        self.assertEqual(mgr.stored, [])

    async def test_extract_disabled_skips_stage(self):
        runner = self._runner(config={"maintenance_extract_enabled": False})
        report = await runner.run_cycle(dry_run=True)
        self.assertIsNone(report.extract_manifest)

    async def test_create_op_executed_when_reviewer_disabled(self):
        blocks = [_block()]
        seg = _FakeSegmenter(_seg_result(blocks))
        create_op = {
            "type": "create",
            "content": "用户喜欢美式咖啡",
            "domain": "preference",
            "scope": "personal",
            "subject": "u1",
            "subjects": ["u1"],
            "entities": [],
            "topics": [],
            "disclosure": "",
            "importance": 3,
            "umo": "qq:private:u1",
            "source_excerpt": "用户说喜欢美式咖啡",
            "truncated_block": False,
        }
        cur = _FakeCurator(create=[create_op])
        kv = _KVLog()
        mgr = _StoreMemoryMgr()
        runner = self._runner(memory_mgr=mgr, kv=kv)
        for p in self._patched(seg, cur):
            p.start()
        try:
            report = await runner.run_cycle(dry_run=False)
        finally:
            for p in self._patched(seg, cur):
                p.stop()
        self.assertEqual(report.executed_ops, 1)
        self.assertEqual(len(mgr.stored), 1)
        kw = mgr.stored[0]
        self.assertEqual(kw["event"].unified_msg_origin, "qq:private:u1")
        self.assertEqual(kw["event"].get_sender_id(), "u1")
        self.assertEqual(kw["memory_scope"], "personal")
        self.assertEqual(kw["owner_sender_id"], "u1")
        self.assertEqual(kw["extra_metadata"], {"created_by": "maintenance_curator"})
        # 游标已提交
        self.assertIn(
            "qq:qq:private:u1", kv.store.get("maintenance_extract_cursors", {})
        )

    async def test_run_test_stage_extract(self):
        blocks = [_block()]
        seg = _FakeSegmenter(_seg_result(blocks, llm_calls=1))
        cur = _FakeCurator(create=[{"type": "create", "content": "x"}])
        kv = _KVLog()
        runner = self._runner(kv=kv)
        for p in self._patched(seg, cur):
            p.start()
        try:
            result = await runner.run_test_stage("extract")
        finally:
            for p in self._patched(seg, cur):
                p.stop()
        self.assertEqual(result["stage"], "extract")
        self.assertTrue(result["extract"]["parsed"])
        self.assertEqual(result["extract"]["operations"], 1)
        self.assertEqual(kv.put_log, [])  # dry-run 不提交游标


class ExecuteCreateTests(unittest.IsolatedAsyncioTestCase):
    def _runner(self, mgr):
        return MaintenanceRunner(None, mgr, _RunnerFakeLLM(), {}, None, None)

    def _op(self, **over):
        op = {
            "type": "create",
            "content": "用户喜欢美式咖啡",
            "domain": "preference",
            "scope": "personal",
            "subject": "u1",
            "subjects": ["u1"],
            "entities": ["咖啡"],
            "topics": ["饮食"],
            "disclosure": "讨论咖啡时",
            "importance": 4,
            "umo": "qq:private:u1",
            "source_excerpt": " excerpt ",
            "truncated_block": False,
        }
        op.update(over)
        return op

    async def test_execute_create_fields(self):
        mgr = _StoreMemoryMgr()
        runner = self._runner(mgr)
        self.assertTrue(await runner._execute_create(self._op()))
        kw = mgr.stored[0]
        self.assertEqual(kw["event"].unified_msg_origin, "qq:private:u1")
        self.assertEqual(kw["event"].get_sender_id(), "u1")
        self.assertEqual(kw["content"], "用户喜欢美式咖啡")
        self.assertEqual(kw["domain"], "preference")
        self.assertEqual(kw["memory_scope"], "personal")
        self.assertEqual(kw["subject"], "u1")
        self.assertEqual(kw["owner_sender_id"], "u1")
        self.assertEqual(kw["owner_sender_ids"], ["u1"])
        self.assertEqual(kw["importance"], 4)
        self.assertEqual(kw["extra_metadata"], {"created_by": "maintenance_curator"})

    async def test_execute_create_global_downgraded(self):
        mgr = _StoreMemoryMgr()
        runner = self._runner(mgr)
        await runner._execute_create(self._op(scope="global"))
        self.assertEqual(mgr.stored[0]["memory_scope"], "personal")

    async def test_execute_create_group_scope_sender_fallback(self):
        mgr = _StoreMemoryMgr()
        runner = self._runner(mgr)
        op = self._op(scope="group", subject="", subjects=[], umo="qq:group:g1")
        self.assertTrue(await runner._execute_create(op))
        kw = mgr.stored[0]
        self.assertEqual(kw["memory_scope"], "group")
        self.assertIsNone(kw["owner_sender_id"])
        # 群聊无 subject 时 sender 兜底为 session_id，避免空 owner
        self.assertEqual(kw["event"].get_sender_id(), "g1")

    async def test_execute_create_rejects_empty_content_or_umo(self):
        mgr = _StoreMemoryMgr()
        runner = self._runner(mgr)
        self.assertFalse(await runner._execute_create(self._op(content="")))
        self.assertFalse(await runner._execute_create(self._op(umo="")))
        self.assertEqual(mgr.stored, [])


class ReviewerCreateSourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_op_source_data(self):
        class _NoFetchMgr:
            async def _get_memory_by_uri(self, uri):
                raise AssertionError("create 操作不应拉取源记忆")

        reviewer = ReviewerAgent(None, _NoFetchMgr(), _RunnerFakeLLM(), {})
        data = await reviewer._get_op_source_data(
            {
                "type": "create",
                "content": "提案内容",
                "scope": "personal",
                "source_excerpt": "来源摘要",
            },
            {},
        )
        self.assertEqual(
            data,
            {
                "memories": {},
                "proposed_create": {
                    "content": "提案内容",
                    "scope": "personal",
                    "source_excerpt": "来源摘要",
                },
            },
        )


class SegmenterStaleCarryTests(unittest.IsolatedAsyncioTestCase):
    async def test_stale_carry_dropped_and_cleared(self):
        now = datetime.now(timezone.utc)
        rows = [_row(1, "聊" * 120, now - timedelta(hours=3))]
        kv = _KV(
            {
                "maintenance_extract_carry": {
                    "qq:g1": {"start_id": 99, "cycles": 2, "fail_count": 0}
                }
            }
        )
        agent = _StubSegmenter(rows, [("qq", "g1")], _SegFakeLLM(), _config(), kv)
        result = await agent.collect_blocks(10)
        # 过期 carry（start_id=99 ≠ 候选块 first_id=1）被丢弃并清除
        self.assertEqual(result["carry_updates"], {"qq:g1": None})
        self.assertEqual(len(result["blocks"]), 1)

    async def test_stale_carry_cycles_not_inherited(self):
        now = datetime.now(timezone.utc)
        # 块尾距今 5 分钟未超 gap → 判定不完整时挂起
        rows = [_row(1, "聊" * 120, now - timedelta(minutes=5))]
        kv = _KV(
            {
                "maintenance_extract_carry": {
                    "qq:g1": {"start_id": 99, "cycles": 2, "fail_count": 0}
                }
            }
        )
        agent = _StubSegmenter(
            rows, [("qq", "g1")], _SegFakeLLM(responses=[INCOMPLETE]), _config(), kv
        )
        result = await agent.collect_blocks(10)
        # cycles 从 0 重计为 1，而非继承过期的 2（否则将立刻触发强制截断）
        self.assertEqual(
            result["carry_updates"],
            {"qq:g1": {"start_id": 1, "cycles": 1, "fail_count": 0}},
        )

    async def test_matching_carry_still_inherited(self):
        now = datetime.now(timezone.utc)
        rows = [_row(1, "聊" * 120, now - timedelta(minutes=5))]
        kv = _KV(
            {
                "maintenance_extract_carry": {
                    "qq:g1": {"start_id": 1, "cycles": 1, "fail_count": 0}
                }
            }
        )
        agent = _StubSegmenter(
            rows, [("qq", "g1")], _SegFakeLLM(responses=[INCOMPLETE]), _config(), kv
        )
        result = await agent.collect_blocks(10)
        # start_id 匹配时正常继承 cycles
        self.assertEqual(
            result["carry_updates"],
            {"qq:g1": {"start_id": 1, "cycles": 2, "fail_count": 0}},
        )


class MainAliasTests(unittest.TestCase):
    """main.py 薄封装行为不变：共享解析经别名委派。"""

    def test_parse_extracted_memories_delegates(self):
        text = (
            '```json\n[{"scope": "global", "type": "fact", '
            '"content": "测试内容", "importance": 9}]\n```'
        )
        result = plugin_main.MemoryPlugin._parse_extracted_memories(
            None, text, "private"
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["scope"], "personal")  # global 降级
        self.assertEqual(result[0]["importance"], 5)  # clamp 到 1-5

    def test_strip_json_fence_delegates(self):
        self.assertEqual(
            plugin_main.MemoryPlugin._strip_json_fence(None, "```json\n{}\n```"),
            "{}",
        )
        self.assertEqual(plugin_main._normalize_subject_id("user:u1"), "u1")
        self.assertEqual(plugin_main._normalize_subject_ids("u1, u2"), ["u1", "u2"])


if __name__ == "__main__":
    unittest.main()
