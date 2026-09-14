"""召回预算、局部降级、单次长度重试与取消传播。"""

import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from test_memory_commands import MemoryManager, plugin_main
from test_memory_sharing import Event, memory

from recall_utils import is_query_length_error, query_max_chars, recall_timeout


class QueryBudgetTests(unittest.TestCase):
    def test_current_question_has_priority_over_history_and_tools(self):
        contexts = [
            {"role": "tool", "content": "TOOL" * 5000},
            {"role": "user", "content": "OLD" * 5000},
            {"role": "assistant", "content": "RECENT" * 5000},
        ]
        query = plugin_main._build_recall_query("CURRENT QUESTION", contexts, 128)
        self.assertTrue(query.startswith("CURRENT QUESTION"))
        self.assertLessEqual(len(query), 128)
        self.assertNotIn("TOOL", query)
        self.assertIn("RECENT", query)
        self.assertEqual(
            plugin_main._build_recall_query("a" * 1000, contexts, 128), "a" * 128
        )

    def test_invalid_settings_are_bounded(self):
        for value in (None, "invalid", -100, 0, 10**9):
            self.assertGreaterEqual(
                query_max_chars({"recall_query_max_chars": value}), 128
            )
            self.assertLessEqual(
                query_max_chars({"recall_query_max_chars": value}), 2048
            )
            self.assertGreaterEqual(
                recall_timeout({"recall_timeout_seconds": value}), 2
            )
            self.assertLessEqual(recall_timeout({"recall_timeout_seconds": value}), 30)

    def test_only_explicit_length_errors_are_retryable(self):
        self.assertTrue(
            is_query_length_error(
                RuntimeError("Input id 0 has 13170 tokens, exceeding max 8192 tokens")
            )
        )
        for message in (
            "Connection error",
            "Request timed out",
            "401 Unauthorized",
            "invalid Gemini function call history name does not match",
            "429 rate limited",
        ):
            self.assertFalse(is_query_length_error(RuntimeError(message)))
        error = RuntimeError("context_length_exceeded")
        error.status_code = 401
        self.assertFalse(is_query_length_error(error))


class RecallResilienceTests(unittest.IsolatedAsyncioTestCase):
    def manager(self, error=None):
        mgr = MemoryManager(None, {})
        mgr._kb_helper = SimpleNamespace(
            vec_db=SimpleNamespace(
                retrieve=AsyncMock(side_effect=error, return_value=[]),
                rerank_provider=None,
            )
        )
        mgr._sparse_retrieve = AsyncMock(return_value=[memory()])
        mgr._disclosure_retrieve = AsyncMock(return_value=[])
        return mgr

    async def test_embedding_failure_keeps_sparse_hits_without_network_retry(self):
        mgr = self.manager(RuntimeError("Connection error"))
        result = await mgr._retrieve_with_filter("coffee", 5, {}, event=Event())
        self.assertEqual(len(result), 1)
        mgr.vec_db.retrieve.assert_awaited_once()
        mgr._sparse_retrieve.assert_awaited_once()

    async def test_all_scope_channels_share_one_length_retry(self):
        mgr = self.manager(
            RuntimeError("Input id 0 has 13170 tokens, exceeding max 8192 tokens")
        )
        budget = [1]
        results = await asyncio.gather(
            *[
                mgr._retrieve_with_filter(
                    "x" * 1000, 5, {}, event=Event(), retry_budget=budget
                )
                for _ in range(2)
            ]
        )
        self.assertTrue(all(results))
        self.assertEqual(mgr.vec_db.retrieve.await_count, 3)
        lengths = [
            len(call.kwargs["query"]) for call in mgr.vec_db.retrieve.await_args_list
        ]
        self.assertEqual(lengths.count(500), 1)
        self.assertEqual(budget, [0])

    async def test_dense_timeout_reserves_time_for_sparse_results(self):
        mgr = self.manager()
        cancelled = asyncio.Event()

        async def blocked(**kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        mgr.vec_db.retrieve.side_effect = blocked
        result = await mgr._retrieve_with_filter(
            "coffee", 5, {}, event=Event(), deadline=time.monotonic() + 0.04
        )
        self.assertEqual(len(result), 1)
        self.assertTrue(cancelled.is_set())

    async def test_cancellation_is_not_swallowed(self):
        mgr = self.manager(asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            await mgr._retrieve_with_filter("coffee", 5, {}, event=Event())
        mgr._sparse_retrieve.assert_not_awaited()

    async def test_reranker_does_not_receive_foreign_private_memory(self):
        mgr = self.manager()
        mgr._sparse_retrieve.return_value = [
            memory(owner="sl_alice", uri="facts://foreign"),
            memory(),
            memory(uri="facts://two"),
        ]
        mgr.vec_db.rerank_provider = SimpleNamespace(
            rerank=AsyncMock(return_value=[SimpleNamespace(index=0, relevance_score=1)])
        )
        await mgr._retrieve_with_filter("coffee", 5, {}, event=Event(), final_top_k=1)
        docs = mgr.vec_db.rerank_provider.rerank.await_args.args[1]
        self.assertEqual(len(docs), 2)

    async def test_common_entry_bounds_tool_queries_and_bumps_once(self):
        mgr = MemoryManager(None, {"recall_query_max_chars": 128})
        mgr._kb_helper = SimpleNamespace()
        mgr.has_any_active_memory = AsyncMock(return_value=True)
        mgr._retrieve_with_filter = AsyncMock(return_value=[memory()])
        mgr._retrieve_shared_personal = AsyncMock(return_value=[])
        mgr._bump_recall_stats = AsyncMock()
        result = await mgr.recall_memories(Event(), "x" * 10000, bump=True)
        self.assertEqual(len(result), 1)
        self.assertTrue(
            all(
                len(call.args[0]) == 128
                for call in mgr._retrieve_with_filter.await_args_list
            )
        )
        mgr._bump_recall_stats.assert_awaited_once()

    async def test_one_failed_scope_does_not_discard_successful_scopes(self):
        mgr = self.manager()
        mgr.has_any_active_memory = AsyncMock(return_value=True)

        async def retrieve(query, top_k, filters, **kwargs):
            if filters["memory_scope"] == "global":
                raise RuntimeError("storage failure")
            return [memory()]

        mgr._retrieve_with_filter = retrieve
        mgr._retrieve_shared_personal = AsyncMock(return_value=[])
        result = await mgr.recall_memories(Event(), "coffee")
        self.assertEqual(len(result), 1)
