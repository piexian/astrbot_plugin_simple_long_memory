import asyncio
import copy
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import test_maintenance_extract as extract_tests
from test_maintenance_extract import (
    _block,
    _CuratorLLM,
    _FakeCurator,
    _FakeSegmenter,
    _KVLog,
    _seg_result,
    _StoreMemoryMgr,
)
from test_maintenance_segmenter import _config, _FakeLLM, _row, _StubSegmenter
from test_maintenance_segmenter_conv2 import _agent, _conv2_conv
from test_memory_commands import MemoryManager

import maintenance.agents.segmenter as segmenter_module
import maintenance.runner as runner_module
from maintenance.agents.curator import CuratorAgent
from maintenance.agents.segmenter import CURSOR_KV_KEY, SegmenterAgent
from maintenance.extraction_journal import JOURNAL_KV_KEY


class DurableKV(_KVLog):
    fail_ack = False

    async def get_kv_data(self, key, default=None):
        return copy.deepcopy(self.store.get(key, default))

    async def put_kv_data(self, key, value):
        if self.fail_ack and key == JOURNAL_KV_KEY and value.get("completed"):
            self.fail_ack = False
            raise RuntimeError("journal write failed")
        await super().put_kv_data(key, copy.deepcopy(value))


class VectorStore:
    def __init__(self):
        self.docs = {}
        self.document_storage = self
        self.insert_count = 0
        self.crash_after_insert = False

    async def get_documents(self, metadata_filters, limit=1):
        return [
            copy.deepcopy(doc)
            for doc in self.docs.values()
            if all(doc["metadata"].get(k) == v for k, v in metadata_filters.items())
        ][:limit]

    async def insert(self, content, metadata, id):
        self.docs[id] = {"id": id, "text": content, "metadata": copy.deepcopy(metadata)}
        self.insert_count += 1
        if self.crash_after_insert:
            self.crash_after_insert = False
            raise asyncio.CancelledError()

    async def delete_documents(self, metadata_filters):
        for key, doc in list(self.docs.items()):
            if all(doc["metadata"].get(k) == v for k, v in metadata_filters.items()):
                del self.docs[key]

    async def retrieve(self, **kwargs):
        return []


def real_manager(kv):
    mgr = MemoryManager.__new__(MemoryManager)
    mgr.config = {}
    mgr._rebuilding, mgr._pending_writes = False, []
    mgr._kb_helper = SimpleNamespace(
        kb=SimpleNamespace(kb_id="kb-test", kb_name="test"), vec_db=VectorStore()
    )
    mgr._kv_put, mgr._kv_get, mgr._kv_delete = (
        kv.put_kv_data,
        kv.get_kv_data,
        AsyncMock(),
    )
    mgr._register_kb_document, mgr._unregister_kb_documents = AsyncMock(), AsyncMock()
    mgr._sync_kb_stats = AsyncMock()
    mgr._ensure_kb_document = AsyncMock(return_value=True)
    mgr._ensure_extraction_vector = AsyncMock()
    mgr._link_manager = None
    return mgr


class ExtractionReliabilityTests(unittest.IsolatedAsyncioTestCase):
    def runner(self, mgr, kv, **config):
        return extract_tests.RunnerExtractionTests()._runner(
            memory_mgr=mgr, kv=kv, config=config
        )

    async def test_failed_store_survives_runner_restart(self):
        kv, mgr = _KVLog(), _StoreMemoryMgr()
        mgr.store_memory = AsyncMock(side_effect=RuntimeError("write failed"))
        block = _block()
        op = {"type": "create", "content": "用户喜欢咖啡", "umo": block.umo}
        with (
            patch.object(
                runner_module,
                "SegmenterAgent",
                return_value=_FakeSegmenter(_seg_result([block])),
            ),
            patch.object(
                runner_module, "CuratorAgent", return_value=_FakeCurator(create=[op])
            ),
        ):
            await self.runner(mgr, kv).run_cycle()
        self.assertNotIn(block.conv_key, kv.store.get(CURSOR_KV_KEY, {}))
        mgr.store_memory = AsyncMock(return_value="facts://stored")
        with patch.object(
            runner_module,
            "SegmenterAgent",
            side_effect=AssertionError("must replay saved proposals"),
        ):
            await self.runner(mgr, kv).run_cycle()
        mgr.store_memory.assert_awaited_once()
        self.assertEqual(kv.store[CURSOR_KV_KEY][block.conv_key]["id"], 10)

    async def test_llm_failure_retains_cursor(self):
        kv, mgr = _KVLog(), _StoreMemoryMgr()
        block = _block()
        curator = CuratorAgent(None, mgr, _CuratorLLM([None]), {})
        with (
            patch.object(
                runner_module,
                "SegmenterAgent",
                return_value=_FakeSegmenter(_seg_result([block])),
            ),
            patch.object(runner_module, "CuratorAgent", return_value=curator),
        ):
            await self.runner(mgr, kv).run_cycle()
        self.assertNotIn(block.conv_key, kv.store.get(CURSOR_KV_KEY, {}))

    async def test_budget_leaves_calls_for_curator(self):
        kv, mgr = _KVLog(), _StoreMemoryMgr()
        now = datetime.now(timezone.utc)
        rows = [
            _row(i + 1, "聊" * 200, now - timedelta(hours=10 - i)) for i in range(10)
        ]
        seg = _StubSegmenter(rows, [("qq", "g1")], _FakeLLM(), _config(), kv)
        curator = CuratorAgent(None, mgr, _CuratorLLM(), {})
        with (
            patch.object(runner_module, "SegmenterAgent", return_value=seg),
            patch.object(runner_module, "CuratorAgent", return_value=curator),
        ):
            report = await self.runner(
                mgr, kv, maintenance_max_llm_calls=10
            ).run_cycle()
        self.assertGreater(report.extract_manifest.metrics["blocks_processed"], 0)

    def test_group_lines_include_sender_id(self):
        seg = SegmenterAgent(None, _FakeLLM(), {}, None, None)
        line, _ = seg._flatten_row(
            _row(
                1,
                "我喜欢咖啡",
                datetime.now(timezone.utc),
                sender_id="123456",
                sender_name="小明",
            )
        )
        self.assertIn("123456", line)
        self.assertIn("小明", line)

    def test_nonpositive_chunk_size_is_safe(self):
        for size in (0, -1):
            seg = SegmenterAgent(
                None,
                _FakeLLM(),
                {"maintenance_extract_conv2_chunk_messages": size},
                None,
                None,
            )
            items = seg._flatten_conv2_history(
                [{"role": "user", "content": "值得记住的信息"}] * 4, "q:u"
            )
            self.assertTrue(seg._pre_split_conv2(items, 0, datetime.now(timezone.utc)))

    def test_initial_segments_obey_character_limit(self):
        seg = SegmenterAgent(
            None, _FakeLLM(), {"maintenance_segment_max_chars": 120}, None, None
        )
        rows = [_row(i + 1, "聊" * 200, datetime.now(timezone.utc)) for i in range(3)]
        segments = seg._pre_split(rows)
        self.assertTrue(all(len(s.text) <= 120 for s in segments))
        self.assertEqual(sum(s.body_chars for s in segments), 600)

    async def test_create_retries_after_insert_without_duplication(self):
        kv = DurableKV()
        mgr = real_manager(kv)
        mgr.vec_db.crash_after_insert = True
        block = _block()
        op = {"type": "create", "content": "用户喜欢咖啡", "umo": block.umo}
        with (
            patch.object(
                runner_module,
                "SegmenterAgent",
                return_value=_FakeSegmenter(_seg_result([block])),
            ),
            patch.object(
                runner_module, "CuratorAgent", return_value=_FakeCurator(create=[op])
            ),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await self.runner(mgr, kv).run_cycle()
        await self.runner(mgr, kv).run_cycle()
        self.assertEqual(mgr.vec_db.insert_count, 1)
        self.assertEqual(kv.store[CURSOR_KV_KEY][block.conv_key]["id"], 10)
        self.assertFalse(kv.store[JOURNAL_KV_KEY])

    async def test_ack_failure_replays_idempotently(self):
        kv = DurableKV()
        kv.fail_ack = True
        mgr = real_manager(kv)
        block = _block()
        op = {"type": "create", "content": "用户喜欢咖啡", "umo": block.umo}
        with (
            patch.object(
                runner_module,
                "SegmenterAgent",
                return_value=_FakeSegmenter(_seg_result([block])),
            ),
            patch.object(
                runner_module, "CuratorAgent", return_value=_FakeCurator(create=[op])
            ),
        ):
            await self.runner(mgr, kv).run_cycle()
        self.assertNotIn(block.conv_key, kv.store.get(CURSOR_KV_KEY, {}))
        await self.runner(mgr, kv).run_cycle()
        self.assertEqual(mgr.vec_db.insert_count, 1)
        self.assertEqual(kv.store[CURSOR_KV_KEY][block.conv_key]["id"], 10)

    async def test_retry_repairs_partial_faiss_write(self):
        from astrbot.core.db.vec_db.faiss_impl.vec_db import FaissVecDB

        for operation in ("create", "update"):
            for failure_point in ("insert", "save_index"):
                with self.subTest(operation=operation, failure_point=failure_point):
                    with tempfile.TemporaryDirectory() as directory:
                        provider = SimpleNamespace(
                            get_dim=lambda: 2,
                            get_embedding=AsyncMock(return_value=[0.25, 0.75]),
                        )
                        vec_db = FaissVecDB(
                            str(Path(directory) / "documents.db"),
                            str(Path(directory) / "vectors.faiss"),
                            provider,
                        )
                        await vec_db.initialize()
                        self.addAsyncCleanup(vec_db.close)
                        kv = DurableKV()
                        mgr = real_manager(kv)
                        mgr._kb_helper.vec_db = vec_db
                        del mgr._ensure_extraction_vector
                        event = runner_module._MaintenanceEvent("qq:private:u1", "u1")
                        op = {
                            "type": operation,
                            "content": "喜欢咖啡",
                            "new_content": "喜欢咖啡",
                            "umo": event.unified_msg_origin,
                            "_extract_id": "partial-write",
                        }
                        if operation == "update":
                            op["uri"] = await mgr.store_memory(event, "旧偏好", "facts")
                        with patch.object(
                            vec_db.embedding_storage,
                            failure_point,
                            side_effect=asyncio.CancelledError,
                        ):
                            with self.assertRaises(asyncio.CancelledError):
                                await self.runner(mgr, kv).execute_approved(op)
                        self.assertTrue(await self.runner(mgr, kv).execute_approved(op))
                        docs = await vec_db.document_storage.get_documents(
                            metadata_filters={"is_memory_record": True}, limit=None
                        )
                        self.assertEqual(len(docs), 1)
                        self.assertEqual(vec_db.embedding_storage.index.ntotal, 1)
                        persisted = vec_db.embedding_storage._read_index(
                            vec_db.embedding_storage.path
                        )
                        vector, status, _ = mgr._reconstruct_faiss_vector(
                            persisted,
                            docs[0]["id"],
                            mgr._prepare_faiss_index(persisted),
                        )
                        self.assertEqual(status, "ok")
                        self.assertEqual(vector, [0.25, 0.75])
                        await vec_db.close()

    async def test_update_retries_partial_replacement(self):
        kv = DurableKV()
        mgr = real_manager(kv)
        r = self.runner(mgr, kv)
        event = runner_module._MaintenanceEvent("qq:private:u1", "u1")
        old_uri = await mgr.store_memory(event, "旧偏好", "preference")
        op = {
            "type": "update",
            "uri": old_uri,
            "new_content": "新偏好",
            "_extract_id": "update-1",
        }
        mgr.vec_db.crash_after_insert = True
        with self.assertRaises(asyncio.CancelledError):
            await r.execute_approved(op)
        self.assertEqual(len(mgr.vec_db.docs), 2)
        self.assertTrue(await self.runner(mgr, kv).execute_approved(op))
        self.assertEqual(mgr.vec_db.insert_count, 2)
        self.assertEqual(len(mgr.vec_db.docs), 1)
        self.assertTrue(await self.runner(mgr, kv).execute_approved(op))
        self.assertEqual(mgr.vec_db.insert_count, 2)

    async def test_operation_limit_replays_remainder(self):
        kv, mgr = DurableKV(), _StoreMemoryMgr()
        block = _block()
        ops = [
            {"type": "create", "content": text, "umo": block.umo}
            for text in ("喜欢咖啡", "喜欢跑步")
        ]
        with (
            patch.object(
                runner_module,
                "SegmenterAgent",
                return_value=_FakeSegmenter(_seg_result([block])),
            ),
            patch.object(
                runner_module, "CuratorAgent", return_value=_FakeCurator(create=ops)
            ),
        ):
            await self.runner(mgr, kv, maintenance_max_ops_per_cycle=1).run_cycle()
        self.assertEqual(len(mgr.stored), 1)
        self.assertNotIn(block.conv_key, kv.store.get(CURSOR_KV_KEY, {}))
        await self.runner(mgr, kv, maintenance_max_ops_per_cycle=1).run_cycle()
        self.assertEqual(len(mgr.stored), 2)
        self.assertEqual(kv.store[CURSOR_KV_KEY][block.conv_key]["id"], 10)

    async def test_full_pending_queue_keeps_proposal_for_retry(self):
        kv, mgr = DurableKV(), _StoreMemoryMgr()
        kv.store["maintenance_pending_review"] = [
            {"id": 1, "status": "pending", "op": {"type": "archive", "uri": "old"}}
        ]
        block = _block()
        op = {"type": "create", "content": "用户喜欢咖啡", "umo": block.umo}
        r = self.runner(
            mgr, kv, maintenance_reviewer_enabled=True, maintenance_pending_queue_max=1
        )
        r._run_reviewer = AsyncMock(return_value=[])
        with (
            patch.object(
                runner_module,
                "SegmenterAgent",
                return_value=_FakeSegmenter(_seg_result([block])),
            ),
            patch.object(
                runner_module, "CuratorAgent", return_value=_FakeCurator(create=[op])
            ),
        ):
            await r.run_cycle()
        self.assertNotIn(block.conv_key, kv.store.get(CURSOR_KV_KEY, {}))
        self.assertTrue(kv.store[JOURNAL_KV_KEY]["operations"])
        kv.store["maintenance_pending_review"][0]["status"] = "rejected"
        await r.run_cycle()
        queue = kv.store["maintenance_pending_review"]
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["op"]["content"], op["content"])
        self.assertEqual(kv.store[CURSOR_KV_KEY][block.conv_key]["id"], 10)
        self.assertFalse(mgr.stored)

    async def test_group_ownership_uses_verified_id(self):
        kv = DurableKV()
        mgr = real_manager(kv)
        block = _block(umo="qq:group:g1")
        block.sender_ids = ["123456"]
        curator = CuratorAgent(
            None,
            mgr,
            _CuratorLLM(
                [
                    {
                        "memories": [
                            {
                                "content": "喜欢咖啡",
                                "scope": "personal",
                                "subjects": ["123456"],
                            }
                        ],
                        "updates": [],
                    }
                ]
            ),
            {},
        )
        result = await curator.run([block], 1)
        self.assertTrue(await self.runner(mgr, kv)._execute_create(result["create"][0]))
        meta = next(iter(mgr.vec_db.docs.values()))["metadata"]
        self.assertEqual(meta["owner_user_id"], "qq_123456")
        bad = {**result["create"][0], "subject": "小明", "subjects": ["小明"]}
        self.assertFalse(await self.runner(mgr, kv)._execute_create(bad))

    async def test_rebuild_preserves_provenance_and_idempotency(self):
        kv = DurableKV()
        mgr = real_manager(kv)
        mgr._rebuilding = True
        event = runner_module._MaintenanceEvent("qq:private:u1", "u1")
        extra = {"created_by": "maintenance_curator", "extraction_id": "create-1"}
        uri = await mgr.store_memory(
            event, "喜欢咖啡", "preference", extra_metadata=extra
        )
        self.assertEqual(
            await mgr.store_memory(
                event, "喜欢咖啡", "preference", extra_metadata=extra
            ),
            uri,
        )
        self.assertEqual(len(mgr._pending_writes), 1)
        self.assertEqual(await mgr._flush_pending_writes(), 1)
        meta = next(iter(mgr.vec_db.docs.values()))["metadata"]
        self.assertEqual(meta["created_by"], "maintenance_curator")
        self.assertEqual(meta["extraction_id"], "create-1")

    async def test_llm_failure_blocks_later_blocks_in_same_conversation(self):
        first, second = _block(), _block(first_id=11, last_id=20)
        llm = _CuratorLLM([None, {"memories": [], "updates": []}])
        curator = CuratorAgent(None, _StoreMemoryMgr(), llm, {})
        result = await curator.run([first, second], 10)
        self.assertEqual(
            [value for _, value in result["outcomes"]], ["failed", "skipped_budget"]
        )
        self.assertEqual(llm.calls, 1)

    async def test_long_messages_resume_without_losing_text(self):
        for source in ("pmh", "conv2"):
            with self.subTest(source=source):
                kv = DurableKV()
                text = "".join(str(i % 10) for i in range(1100))
                now = datetime.now(timezone.utc)
                cfg = _config(
                    maintenance_segment_max_chars=120,
                    maintenance_extract_max_blocks_per_cycle=1,
                    extraction_min_content_length=1000,
                )
                if source == "pmh":
                    agent = _StubSegmenter(
                        [_row(1, text, now)], [("qq", "g1")], _FakeLLM(), cfg, kv
                    )
                else:
                    agent = _agent(
                        [],
                        [],
                        [
                            _conv2_conv(
                                "qq",
                                "g1",
                                [{"role": "user", "content": text}],
                                int(now.timestamp()),
                            )
                        ],
                        _FakeLLM(),
                        cfg,
                        kv,
                    )
                recovered = ""
                for _ in range(30):
                    result = await agent.collect_blocks(10)
                    if not result["blocks"]:
                        break
                    block = result["blocks"][0]
                    self.assertLessEqual(len(block.text), 120)
                    recovered += block.text.split(": ", 1)[1]
                    await self.runner(_StoreMemoryMgr(), kv)._commit_extract_progress(
                        result, [(block, "nothing")]
                    )
                self.assertEqual(recovered, text)

    async def test_old_backlog_and_unscanned_conversations_resume(self):
        class Clock(datetime):
            value = datetime(2026, 9, 4, 3, tzinfo=timezone.utc)

            @classmethod
            def now(cls, tz=None):
                return cls.value

        class WindowSegmenter(_StubSegmenter):
            async def _list_active_conversations(self, since):
                return list(
                    dict.fromkeys(
                        (r.platform_id, r.user_id)
                        for r in self._rows
                        if r.created_at >= since
                    )
                )

        kv = DurableKV()
        rows = [
            _row(1, "聊" * 200, Clock.value - timedelta(hours=4)),
            _row(2, "聊" * 200, Clock.value - timedelta(hours=2)),
            _row(3, "聊" * 200, Clock.value - timedelta(hours=1), umo="g2"),
        ]
        agent = WindowSegmenter(
            rows,
            [],
            _FakeLLM(),
            _config(maintenance_extract_max_blocks_per_cycle=1),
            kv,
        )
        r = self.runner(_StoreMemoryMgr(), kv)
        with (
            patch.object(segmenter_module, "datetime", Clock),
            patch.object(runner_module, "datetime", Clock),
        ):
            first = await agent.collect_blocks(10)
            await r._commit_extract_progress(first, [(first["blocks"][0], "nothing")])
            Clock.value += timedelta(days=2)
            second = await agent.collect_blocks(10)
            self.assertEqual(second["blocks"][0].last_id, 2)
            await r._commit_extract_progress(second, [(second["blocks"][0], "nothing")])
            third = await agent.collect_blocks(10)
            self.assertEqual(third["blocks"][0].last_id, 3)

    async def test_completed_blocks_advance_before_failed_later_block(self):
        kv, mgr = DurableKV(), _StoreMemoryMgr()
        first, second = _block(), _block(first_id=11, last_id=20)
        ops = [
            {
                "type": "create",
                "content": text,
                "umo": block.umo,
                "_extract_block": block.key,
            }
            for block, text in ((first, "先"), (second, "后"))
        ]
        store = mgr.store_memory

        async def write(**kwargs):
            if kwargs["content"] == "后":
                raise RuntimeError("write failed")
            return await store(**kwargs)

        mgr.store_memory = write
        with (
            patch.object(
                runner_module,
                "SegmenterAgent",
                return_value=_FakeSegmenter(_seg_result([first, second])),
            ),
            patch.object(
                runner_module, "CuratorAgent", return_value=_FakeCurator(create=ops)
            ),
        ):
            await self.runner(mgr, kv).run_cycle()
        self.assertEqual(kv.store[CURSOR_KV_KEY][first.conv_key]["id"], 10)
        mgr.store_memory = store
        await self.runner(mgr, kv).run_cycle()
        self.assertEqual([op["content"] for op in mgr.stored], ["先", "后"])
        self.assertEqual(kv.store[CURSOR_KV_KEY][first.conv_key]["id"], 20)

    async def test_pending_ack_failure_respects_human_rejection(self):
        kv, mgr = DurableKV(), _StoreMemoryMgr()
        kv.fail_ack = True
        block = _block()
        op = {"type": "create", "content": "待审", "umo": block.umo}
        r = self.runner(mgr, kv, maintenance_reviewer_enabled=True)
        r._run_reviewer = AsyncMock(return_value=[])
        with (
            patch.object(
                runner_module,
                "SegmenterAgent",
                return_value=_FakeSegmenter(_seg_result([block])),
            ),
            patch.object(
                runner_module, "CuratorAgent", return_value=_FakeCurator(create=[op])
            ),
        ):
            await r.run_cycle()
        self.assertTrue(kv.store[JOURNAL_KV_KEY])
        kv.store["maintenance_pending_review"][0]["status"] = "rejected"
        await r.run_cycle()
        self.assertFalse(kv.store["maintenance_pending_review"])
        self.assertFalse(kv.store[JOURNAL_KV_KEY])
        self.assertFalse(mgr.stored)
        self.assertEqual(kv.store[CURSOR_KV_KEY][block.conv_key]["id"], 10)

    async def test_cursor_failure_retries_without_reexecuting_operations(self):
        class CursorFailureKV(DurableKV):
            fail_cursor = True

            async def put_kv_data(self, key, value):
                if key == CURSOR_KV_KEY and self.fail_cursor:
                    self.fail_cursor = False
                    raise RuntimeError("cursor write failed")
                await super().put_kv_data(key, value)

        kv, mgr = CursorFailureKV(), _StoreMemoryMgr()
        block = _block()
        op = {"type": "create", "content": "已写入", "umo": block.umo}
        with (
            patch.object(
                runner_module,
                "SegmenterAgent",
                return_value=_FakeSegmenter(_seg_result([block])),
            ),
            patch.object(
                runner_module, "CuratorAgent", return_value=_FakeCurator(create=[op])
            ),
        ):
            await self.runner(mgr, kv).run_cycle()
        self.assertEqual(len(mgr.stored), 1)
        self.assertTrue(kv.store[JOURNAL_KV_KEY])
        await self.runner(mgr, kv).run_cycle()
        self.assertEqual(len(mgr.stored), 1)
        self.assertEqual(kv.store[CURSOR_KV_KEY][block.conv_key]["id"], 10)

    async def test_journal_write_failure_prevents_memory_writes(self):
        kv, mgr = DurableKV(), _StoreMemoryMgr()
        kv.put_kv_data = AsyncMock(side_effect=RuntimeError("KV unavailable"))
        block = _block()
        with (
            patch.object(
                runner_module,
                "SegmenterAgent",
                return_value=_FakeSegmenter(_seg_result([block])),
            ),
            patch.object(
                runner_module,
                "CuratorAgent",
                return_value=_FakeCurator(
                    create=[{"type": "create", "content": "提案", "umo": block.umo}]
                ),
            ),
        ):
            report = await self.runner(mgr, kv).run_cycle()
        self.assertFalse(mgr.stored)
        self.assertTrue(report.errors)

    async def test_dry_run_with_pending_journal_has_no_writes(self):
        kv, mgr = DurableKV(), _StoreMemoryMgr()
        mgr.store_memory = AsyncMock(side_effect=RuntimeError("store unavailable"))
        block = _block()
        with (
            patch.object(
                runner_module,
                "SegmenterAgent",
                return_value=_FakeSegmenter(_seg_result([block])),
            ),
            patch.object(
                runner_module,
                "CuratorAgent",
                return_value=_FakeCurator(
                    create=[{"type": "create", "content": "提案", "umo": block.umo}]
                ),
            ),
        ):
            await self.runner(mgr, kv).run_cycle()
        state, writes = copy.deepcopy(kv.store), len(kv.put_log)
        mgr.store_memory.reset_mock()
        await self.runner(mgr, kv).run_cycle(dry_run=True)
        self.assertEqual(kv.store, state)
        self.assertEqual(len(kv.put_log), writes)
        mgr.store_memory.assert_not_awaited()
