"""后台提取链路测试：整理师三态、Inbox 提取集成、状态提交、create 执行。"""

import copy
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from astrbot.core.db.vec_db.faiss_impl.document_storage import DocumentStorage
from test_memory_commands import plugin_main  # noqa: F401

import maintenance.runner as runner_module
from maintenance.agents.curator import SOURCE_EXCERPT_MAX_CHARS, CuratorAgent
from maintenance.agents.reviewer import ReviewerAgent
from maintenance.extraction_journal import JOURNAL_KV_KEY
from maintenance.inbox import InboxRecord, MaintenanceInbox
from maintenance.memory_gate import GateOutcome
from maintenance.runner import MaintenanceRunner


def _block(
    text="这是一段有信息量的对话内容，讨论了咖啡偏好并得出结论。",
    conv_key="qq:qq:private:u1",
    umo="qq:private:u1",
    first_id=1,
    last_id=10,
    truncated=False,
    realtime_extracted=False,
    row_ids=None,
):
    from maintenance.inbox import ConversationBlock

    return ConversationBlock(
        conv_key=conv_key,
        platform_id="qq",
        umo=umo,
        conversation_id="c1",
        session_type="private",
        session_id="u1",
        text=text,
        message_count=3,
        char_count=len(text),
        first_id=first_id,
        last_id=last_id,
        row_ids=row_ids if row_ids is not None else list(range(first_id, last_id + 1)),
        sender_ids=["u1"],
        user_chars=len(text),
        truncated=truncated,
        realtime_extracted=realtime_extracted,
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
        self.prompts = []

    @property
    def remaining_calls(self):
        return self._max - self.calls

    async def chat_json(self, system, user, model_id=""):
        self.calls += 1
        self.model_ids.append(model_id)
        self.prompts.append(f"{system}\n\n{user}")
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


async def _real_inbox(case):
    """每个用例独立的真实 AstrBot DocumentStorage + 缓冲表，结束自动关闭。"""
    tmp = tempfile.mkdtemp(prefix="inbox-runner-")
    case.addCleanup(shutil.rmtree, tmp, True)
    storage = DocumentStorage(f"{tmp}/doc.db")
    await storage.initialize()
    case.addAsyncCleanup(storage.close)
    inbox = MaintenanceInbox(SimpleNamespace(document_storage=storage))
    return inbox, storage


async def _seed_turn(
    inbox,
    *,
    text="聊" * 160,
    cid="c1",
    umo="qq:private:u1",
    sender_id="u1",
    minutes_ago=60,
    brief="助手简短回复",
):
    record = InboxRecord(
        umo=umo,
        platform_id="qq",
        session_type="private" if ":private:" in umo else "group",
        session_id=umo.rsplit(":", 1)[-1],
        conversation_id=cid,
        sender_id=sender_id,
        sender_name="测试用户",
        user_text=text,
        assistant_brief=brief,
        created_at=(
            datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
        ).isoformat(),
    )
    return await inbox.append(record)


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
            _block(first_id=1, last_id=10),
            _block(first_id=11, last_id=20),
            _block(first_id=21, last_id=30),
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
        self.assertLessEqual(len(create["source_excerpt"]), 2000)

        update = result["update"][0]
        self.assertEqual(
            update,
            {
                "type": "update",
                "uri": "facts://old1",
                "new_content": "用户现在改喝拿铁",
                "reason": "偏好变更",
                "_extract_block": blocks[1].key,
            },
        )

    async def test_realtime_extracted_block_prompts_no_duplicate_create(self):
        llm = _CuratorLLM()
        curator = _curator(llm)
        block = _block(realtime_extracted=True)
        await curator.run([block], 5)
        self.assertIn("realtime extractor", llm.prompts[0])

        llm_plain = _CuratorLLM()
        await _curator(llm_plain).run([_block()], 5)
        self.assertNotIn("realtime extractor", llm_plain.prompts[0])

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
                "umo": "qq:group:g1",
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
                            "uri": "facts://hallucinated",
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
                        {
                            "scope": "global",
                            "type": "fact",
                            "content": "试图写全局",
                            "subject": "u1",
                        }
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
        text = ("  多  空   格\n换行\t" + "聊" * 1500) * 2
        llm = _CuratorLLM(
            responses=[
                {
                    "memories": [
                        {
                            "scope": "personal",
                            "type": "fact",
                            "content": "c",
                            "subject": "u1",
                        }
                    ],
                    "updates": [],
                }
            ]
        )
        curator = _curator(llm)
        result = await curator.run([_block(text=text)], 5)
        excerpt = result["create"][0]["source_excerpt"]
        self.assertGreater(len(" ".join(text.split())), SOURCE_EXCERPT_MAX_CHARS)
        self.assertEqual(len(excerpt), SOURCE_EXCERPT_MAX_CHARS)
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
        blocks = [_block(last_id=10), _block(first_id=11, last_id=20)]
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

    async def test_llm_failure_is_retryable(self):
        llm = _CuratorLLM(responses=[None])
        curator = _curator(llm)
        result = await curator.run([_block()], 5)
        self.assertEqual([o for _, o in result["outcomes"]], ["failed"])
        self.assertEqual(result["blocks_nothing"], 0)

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

    async def ask_decisions(self, questions, state, *, timeout_seconds=30):
        # 镜像真实 MaintenanceLLM.ask_decisions：测试桩不做真实问答
        return None

    def stats(self):
        return {"calls": self.calls, "cache_hits": 0, "errors": 0}


class _StoreMemoryMgr:
    def __init__(self, inbox=None):
        self.vec_db = _FakeVecDB()
        self.inbox = inbox
        self.stored = []

    async def store_memory(self, **kwargs):
        self.stored.append(kwargs)
        return "facts://created-1"


class _KVLog:
    def __init__(self, initial=None):
        self.store = copy.deepcopy(initial or {})
        self.put_log = []

    async def get_kv_data(self, key, default=None):
        return copy.deepcopy(self.store.get(key, default))

    async def put_kv_data(self, key, value):
        self.put_log.append((key, copy.deepcopy(value)))
        self.store[key] = copy.deepcopy(value)


class _MarkerGate:
    """按候选内容标记决定结论的插件内测试替身。"""

    @staticmethod
    def _decide(candidate):
        if "吵架" in candidate.content:
            return GateOutcome("rejected", "rejected:emotion_or_grudge")
        if "闲聊" in candidate.content:
            return GateOutcome("unavailable", "gate judge unavailable")
        return GateOutcome("approved", "approved")

    async def evaluate(self, candidates, *, chat_judge=None):
        return {c.candidate_id: self._decide(c) for c in candidates}


class _AllUnavailableGate:
    async def evaluate(self, candidates, *, chat_judge=None):
        return {
            c.candidate_id: GateOutcome("unavailable", "gate judge unavailable")
            for c in candidates
        }


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


def _create_op(block, content="用户喜欢咖啡"):
    return {
        "type": "create",
        "content": content,
        "domain": "preference",
        "scope": "personal",
        "subject": "u1",
        "subjects": ["u1"],
        "umo": block.umo,
        "sender_ids": ["u1"],
        "_extract_block": block.key,
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
            memory_mgr=memory_mgr,
            llm=_RunnerFakeLLM(),
            config=cfg,
            kv_put=kv.put_kv_data if kv else None,
            kv_get=kv.get_kv_data if kv else None,
        )

    def _patches(self, gate, curator):
        return [
            patch.object(
                runner_module.MaintenanceRunner, "_get_gate", lambda self: gate
            ),
            patch.object(runner_module, "CuratorAgent", lambda **kw: curator),
        ]

    def _context(self, patches):
        for p in patches:
            p.start()

    def _exit(self, patches):
        for p in patches:
            p.stop()

    async def test_gate_outcomes_drive_row_status(self):
        inbox, _storage = await _real_inbox(self)
        await inbox.ensure_table()
        await _seed_turn(inbox, text="聊咖啡偏好" * 40, minutes_ago=90)
        await _seed_turn(
            inbox, text="吵架记账别存我" * 30, umo="qq:private:u2", minutes_ago=60
        )
        await _seed_turn(
            inbox, text="闲聊无增量内容" * 25, umo="qq:private:u3", minutes_ago=30
        )
        kv = _KVLog()

        budgets = []

        async def cur_run(blocks, max_llm_calls):
            budgets.append(max_llm_calls)
            return {
                "create": [_create_op(b) for b in blocks],
                "update": [],
                "notes": "",
                "blocks_processed": len(blocks),
                "blocks_nothing": 0,
                "llm_calls": 1,
                "outcomes": [(b, "created") for b in blocks],
            }

        curator = SimpleNamespace(run=cur_run)
        gate = _MarkerGate()
        runner = self._runner(memory_mgr=_StoreMemoryMgr(inbox=inbox), kv=kv)
        patches = self._patches(gate, curator)
        self._context(patches)
        try:
            manifest = await runner._run_extraction(dry_run=False)
        finally:
            self._exit(patches)

        self.assertTrue(manifest.parsed)
        self.assertEqual(manifest.metrics["blocks"], 3)
        self.assertEqual(manifest.metrics["gate_approved"], 1)
        self.assertEqual(manifest.metrics["gate_unavailable"], 1)
        self.assertEqual(manifest.metrics["gate_rejected"], 1)
        self.assertEqual(budgets, [30])  # 全部提取预算归整理师
        counts = await inbox.status_counts()
        # created 块行保持 pending（等待操作执行完成），拒收块行 skipped
        self.assertEqual(counts.get("pending"), 2)
        self.assertEqual(counts.get("skipped"), 1)
        self.assertEqual(len(manifest.operations), 1)
        journal_state = kv.store[JOURNAL_KV_KEY]
        self.assertEqual(len(journal_state["operations"]), 1)
        self.assertEqual(len(journal_state["block_rows"]), 1)

    async def test_nothing_outcome_marks_rows_processed(self):
        inbox, _storage = await _real_inbox(self)
        await inbox.ensure_table()
        await _seed_turn(inbox, text="聊无增量内容" * 30, minutes_ago=90)
        kv = _KVLog()

        async def cur_run(blocks, max_llm_calls):
            return {
                "create": [],
                "update": [],
                "notes": "",
                "blocks_processed": len(blocks),
                "blocks_nothing": len(blocks),
                "llm_calls": 1,
                "outcomes": [(b, "nothing") for b in blocks],
            }

        curator = SimpleNamespace(run=cur_run)
        gate = _MarkerGate()
        runner = self._runner(memory_mgr=_StoreMemoryMgr(inbox=inbox), kv=kv)
        patches = self._patches(gate, curator)
        self._context(patches)
        try:
            manifest = await runner._run_extraction(dry_run=False)
        finally:
            self._exit(patches)
        self.assertEqual(manifest.operations, [])
        counts = await inbox.status_counts()
        self.assertEqual(counts.get("processed"), 1)
        self.assertNotIn(JOURNAL_KV_KEY, kv.store)  # 无操作不写日志

    async def test_gate_unavailable_keeps_rows_pending(self):
        inbox, _storage = await _real_inbox(self)
        await inbox.ensure_table()
        await _seed_turn(inbox, minutes_ago=90)
        kv = _KVLog()
        curator = _FakeCurator()
        runner = self._runner(memory_mgr=_StoreMemoryMgr(inbox=inbox), kv=kv)
        patches = self._patches(_AllUnavailableGate(), curator)
        self._context(patches)
        try:
            manifest = await runner._run_extraction(dry_run=False)
        finally:
            self._exit(patches)

        self.assertEqual(manifest.metrics["gate_unavailable"], 1)
        self.assertEqual(manifest.metrics["gate_approved"], 0)
        self.assertEqual(manifest.operations, [])
        self.assertEqual(curator.budgets, [])  # 无块进入整理师
        counts = await inbox.status_counts()
        self.assertEqual(counts.get("pending"), 1)
        self.assertNotIn(JOURNAL_KV_KEY, kv.store)

    async def test_curator_budget_exhaustion_keeps_rows_pending(self):
        inbox, _storage = await _real_inbox(self)
        await inbox.ensure_table()
        await _seed_turn(inbox, minutes_ago=90)
        kv = _KVLog()

        async def cur_run(blocks, max_llm_calls):
            self.assertEqual(max_llm_calls, 30)  # 闸门未耗预算时整理师独享提取额度
            return {
                "create": [],
                "update": [],
                "notes": "",
                "blocks_processed": 0,
                "blocks_nothing": 0,
                "llm_calls": 0,
                "outcomes": [(b, "skipped_budget") for b in blocks],
            }

        curator = SimpleNamespace(run=cur_run)
        gate = _MarkerGate()
        runner = self._runner(memory_mgr=_StoreMemoryMgr(inbox=inbox), kv=kv)
        patches = self._patches(gate, curator)
        self._context(patches)
        try:
            manifest = await runner._run_extraction(dry_run=False)
        finally:
            self._exit(patches)

        self.assertEqual(manifest.operations, [])
        # 预算耗尽的块不落终态、不写日志，行留待下周期
        counts = await inbox.status_counts()
        self.assertEqual(counts.get("pending"), 1)
        self.assertNotIn(JOURNAL_KV_KEY, kv.store)

    async def test_settle_failure_keeps_journal_until_retry(self):
        inbox, storage = await _real_inbox(self)
        await inbox.ensure_table()
        row_id = await _seed_turn(inbox, minutes_ago=90)
        kv = _KVLog()
        runner = self._runner(memory_mgr=_StoreMemoryMgr(inbox=inbox), kv=kv)
        journal = runner._extract_journal
        op = _create_op(_block(), "用户喜欢咖啡")
        await journal.prepare(
            operations=[op],
            block_rows={op["_extract_block"]: [row_id]},
            metrics={},
            notes="",
        )
        await journal.acknowledge(journal.state["operations"][0])
        self.assertTrue(journal.settled_block_rows())

        # 真实存储故障：行状态提交失败时日志保留，避免下周期重复提取
        await storage.close()
        with self.assertLogs("astrbot", level="WARNING") as caught:
            await runner._settle_inbox_blocks()
        self.assertTrue(any("提取日志保留" in str(entry) for entry in caught.records))
        # 日志保留：操作未销账重放，行状态待存储恢复后重试
        self.assertTrue(kv.store[JOURNAL_KV_KEY]["operations"])

        # 存储恢复后重试成功，日志才清理，行落 processed
        await storage.initialize()
        await runner._settle_inbox_blocks()
        self.assertFalse(kv.store.get(JOURNAL_KV_KEY))
        counts = await inbox.status_counts()
        self.assertEqual(counts.get("processed"), 1)

    async def test_settle_without_inbox_keeps_journal(self):
        kv = _KVLog()
        runner = self._runner(memory_mgr=_StoreMemoryMgr(inbox=None), kv=kv)
        journal = runner._extract_journal
        op = _create_op(_block(), "用户喜欢咖啡")
        await journal.prepare(
            operations=[op],
            block_rows={op["_extract_block"]: [999]},
            metrics={},
            notes="",
        )
        await journal.acknowledge(journal.state["operations"][0])
        with self.assertLogs("astrbot", level="WARNING") as caught:
            await runner._settle_inbox_blocks()
        self.assertTrue(any("缓冲表不可用" in str(entry) for entry in caught.records))
        # 日志保留：下周期缓冲表可用时重试标 processed
        self.assertTrue(kv.store[JOURNAL_KV_KEY]["operations"])

    async def test_gate_is_lazy_and_reused(self):
        runner = self._runner(memory_mgr=_StoreMemoryMgr(), kv=_KVLog())
        gate = runner._get_gate()
        self.assertIs(runner._get_gate(), gate)
        # 后台 Decision 走受控问答计入总预算
        self.assertEqual(gate._decision_ask, runner._llm.ask_decisions)

    async def test_dry_run_commits_no_inbox_status(self):
        inbox, _storage = await _real_inbox(self)
        await inbox.ensure_table()
        await _seed_turn(inbox, minutes_ago=90)
        gate = _MarkerGate()
        curator = _FakeCurator()
        runner = self._runner(memory_mgr=_StoreMemoryMgr(inbox=inbox), kv=_KVLog())
        patches = self._patches(gate, curator)
        self._context(patches)
        try:
            manifest = await runner._run_extraction(dry_run=True)
        finally:
            self._exit(patches)
        counts = await inbox.status_counts()
        self.assertEqual(counts.get("pending"), 1)
        self.assertEqual(manifest.metrics["blocks"], 1)

    async def test_journal_short_circuits_new_extraction(self):
        inbox, _storage = await _real_inbox(self)
        await inbox.ensure_table()
        await _seed_turn(inbox, minutes_ago=90)
        kv = _KVLog(
            {
                JOURNAL_KV_KEY: {
                    "operations": [
                        {
                            "type": "create",
                            "content": "旧提案",
                            "umo": "qq:private:u1",
                            "_extract_id": "old-op",
                            "_extract_block": "old-block",
                        }
                    ],
                    "completed": [],
                    "block_rows": {"old-block": [999]},
                    "metrics": {"blocks": 1},
                    "notes": "",
                }
            }
        )
        gate = _MarkerGate()
        curator = _FakeCurator()
        runner = self._runner(memory_mgr=_StoreMemoryMgr(inbox=inbox), kv=kv)
        patches = self._patches(gate, curator)
        self._context(patches)
        try:
            manifest = await runner._run_extraction(dry_run=False)
        finally:
            self._exit(patches)
        self.assertEqual(len(manifest.operations), 1)
        self.assertEqual(manifest.operations[0]["content"], "旧提案")
        self.assertEqual(curator.budgets, [])  # 不新提取
        counts = await inbox.status_counts()
        self.assertEqual(counts.get("pending"), 1)  # 新行不受旧日志影响

    async def test_failed_block_marks_retry(self):
        inbox, _storage = await _real_inbox(self)
        await inbox.ensure_table()
        await _seed_turn(inbox, minutes_ago=90)
        kv = _KVLog()

        async def cur_run(blocks, max_llm_calls):
            return {
                "create": [],
                "update": [],
                "notes": "",
                "blocks_processed": 0,
                "blocks_nothing": 0,
                "llm_calls": 1,
                "outcomes": [(b, "failed") for b in blocks],
            }

        curator = SimpleNamespace(run=cur_run)
        gate = _MarkerGate()
        runner = self._runner(memory_mgr=_StoreMemoryMgr(inbox=inbox), kv=kv)
        patches = self._patches(gate, curator)
        self._context(patches)
        try:
            await runner._run_extraction(dry_run=False)
        finally:
            self._exit(patches)
        counts = await inbox.status_counts()
        self.assertEqual(counts.get("failed"), 1)

    async def test_cycle_settles_rows_after_execution(self):
        inbox, _storage = await _real_inbox(self)
        await inbox.ensure_table()
        await _seed_turn(inbox, text="聊咖啡偏好" * 40, minutes_ago=90)
        kv = _KVLog()
        gate = _MarkerGate()
        mgr = _StoreMemoryMgr(inbox=inbox)
        runner = self._runner(memory_mgr=mgr, kv=kv)

        async def cur_run(blocks, max_llm_calls):
            return {
                "create": [_create_op(b) for b in blocks],
                "update": [],
                "notes": "",
                "blocks_processed": len(blocks),
                "blocks_nothing": 0,
                "llm_calls": 1,
                "outcomes": [(b, "created") for b in blocks],
            }

        curator = SimpleNamespace(run=cur_run)
        patches = self._patches(gate, curator)
        self._context(patches)
        try:
            report = await runner.run_cycle(dry_run=False)
        finally:
            self._exit(patches)
        self.assertEqual(report.executed_ops, 1)
        counts = await inbox.status_counts()
        self.assertEqual(counts.get("processed"), 1)
        self.assertEqual(kv.store.get(JOURNAL_KV_KEY), {})  # 日志已清空

    async def test_run_test_stage_extract(self):
        inbox, _storage = await _real_inbox(self)
        await inbox.ensure_table()
        await _seed_turn(inbox, minutes_ago=90)
        gate = _MarkerGate()
        curator = _FakeCurator()
        runner = self._runner(memory_mgr=_StoreMemoryMgr(inbox=inbox), kv=_KVLog())
        patches = self._patches(gate, curator)
        self._context(patches)
        try:
            result = await runner.run_test_stage("extract")
        finally:
            self._exit(patches)
        self.assertEqual(result["stage"], "extract")
        self.assertTrue(result["extract"]["parsed"])
        counts = await inbox.status_counts()
        self.assertEqual(counts.get("pending"), 1)  # dry-run 不写状态

    async def test_extract_disabled_skips_stage(self):
        inbox, _storage = await _real_inbox(self)
        await inbox.ensure_table()
        await _seed_turn(inbox, minutes_ago=90)
        runner = self._runner(
            memory_mgr=_StoreMemoryMgr(inbox=inbox),
            kv=_KVLog(),
            config={"maintenance_extract_enabled": False},
        )
        report = await runner.run_cycle(dry_run=True)
        self.assertIsNone(report.extract_manifest)

    async def test_inbox_unavailable_returns_error_manifest(self):
        runner = self._runner(memory_mgr=_StoreMemoryMgr(inbox=None), kv=_KVLog())
        manifest = await runner._run_extraction(dry_run=False)
        self.assertFalse(manifest.parsed)
        self.assertEqual(manifest.error, "maintenance inbox unavailable")


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
                raise AssertionError("create 操作不应按 URI 拉取源记忆")

        reviewer = ReviewerAgent(
            context=None,
            memory_mgr=_NoFetchMgr(),
            llm=_RunnerFakeLLM(),
            config={},
        )
        op = {
            "type": "create",
            "content": "用户喜欢咖啡",
            "scope": "personal",
            "source_excerpt": "[11-27 10:00] 用户(sender_id=u1): 我喜欢咖啡",
        }
        data = await reviewer._get_op_source_data(op, {})
        self.assertEqual(data["memories"], {})
        self.assertEqual(data["proposed_create"]["content"], "用户喜欢咖啡")
        self.assertIn("source_excerpt", data["proposed_create"])


class MainAliasTests(unittest.TestCase):
    def test_parse_extracted_memories_delegates(self):
        plugin_cls = plugin_main.MemoryPlugin
        text = (
            '[{"scope": "personal", "type": "fact", "content": "c", "subject": "u1"}]'
        )
        parsed = plugin_cls._parse_extracted_memories(
            SimpleNamespace(), text, "private"
        )
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["content"], "c")

    def test_strip_json_fence_delegates(self):
        plugin_cls = plugin_main.MemoryPlugin
        self.assertEqual(
            plugin_cls._strip_json_fence(SimpleNamespace(), "```json\n[1]\n```"),
            "[1]",
        )
