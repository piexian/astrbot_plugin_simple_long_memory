"""召回预算、空池短路、rerank 短路与召回韧性（真实 AstrBot 栈）。

全部基于真实 AstrBot 组件：MemoryPlugin、MemoryManager、真实 KB/FaissVecDB、
真实 VLLMRerankProvider 与 AstrMessageEvent，Embedding/Chat/Rerank 请求由
real_stack 的本地 HTTP 测试服务应答。
"""

from __future__ import annotations

import asyncio
import json
import unittest

from astrbot.api.provider import ProviderRequest
from real_stack import (
    CHAT_PROVIDER_ID,
    RERANK_PROVIDER_ID,
    RealStackTestCase,
    make_memory_meta,
    plugin_main,
    real_message_event,
)

from recall_utils import is_query_length_error, query_max_chars, recall_timeout

RECENT_CREATED_AT = "2026-11-27T00:00:00+00:00"
OWNER_USER_ID = "qq_u1"


def _own_meta(uri: str, **extra) -> dict:
    return make_memory_meta(
        uri,
        owner=OWNER_USER_ID,
        umo="qq:private:u1",
        created_at=RECENT_CREATED_AT,
        **extra,
    )


def _private_event(**kwargs):
    return real_message_event(
        umo=kwargs.pop("umo", "qq:FriendMessage:u1"),
        sender_id=kwargs.pop("sender_id", "u1"),
        sender_name=kwargs.pop("sender_name", "小明"),
        **kwargs,
    )


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


class EmptyPoolShortCircuitTests(RealStackTestCase):
    async def asyncSetUp(self):
        await self.start_stack(with_decision=False, with_plugin=True)

    def _requests(self, kind):
        return [entry for entry in self.server.request_log if entry["kind"] == kind]

    async def test_empty_pool_skips_embedding_and_chat_but_keeps_contact_summary(self):
        event = _private_event()
        self.assertTrue(await self.mgr.observe_contact(event))

        self.assertEqual(await self.mgr.recall_memories(event, "咖啡"), [])

        request = ProviderRequest(prompt="还记得我吗", contexts=[])
        await self.plugin.inject_memories(event, request)
        await asyncio.gather(*list(self.plugin._background_tasks))

        self.assertEqual(self._requests("embeddings"), [])
        self.assertEqual(self._requests("chat"), [])
        injected = "".join(
            str(getattr(part, "text", "")) for part in request.extra_user_content_parts
        )
        self.assertIn("<contact_history_info>", injected)


class NonEmptyPoolTests(RealStackTestCase):
    async def asyncSetUp(self):
        await self.start_stack(
            with_decision=False,
            with_plugin=True,
            config={
                "optimize_recall_query": True,
                "extraction_provider_id": CHAT_PROVIDER_ID,
                "memory_consolidation_enabled": False,
                "recall_sparse_fusion": False,
                "max_memories_per_inject": 3,
            },
        )
        await self.mgr.vec_db.insert(
            "用户喜欢喝手冲咖啡", _own_meta("facts://coffee"), "doc-coffee"
        )
        self.server.queue_chat(json.dumps(["咖啡"], ensure_ascii=False))

    async def test_non_empty_pool_optimizes_query_and_injects_memories(self):
        event = _private_event()
        request = ProviderRequest(prompt="我平时喝什么", contexts=[])

        await self.plugin.inject_memories(event, request)
        await asyncio.gather(*list(self.plugin._background_tasks))

        chats = [entry for entry in self.server.request_log if entry["kind"] == "chat"]
        self.assertEqual(len(chats), 1)
        embeddings = [
            entry for entry in self.server.request_log if entry["kind"] == "embeddings"
        ]
        self.assertTrue(embeddings)
        # 检索词来自优化结果而非原始提问
        self.assertTrue(all("咖啡" in str(entry["input"]) for entry in embeddings))
        injected = "".join(
            str(getattr(part, "text", "")) for part in request.extra_user_content_parts
        )
        self.assertIn("手冲咖啡", injected)


class RerankShortCircuitTests(RealStackTestCase):
    async def _start(self, memories):
        await self.start_stack(
            with_decision=False,
            with_rerank=True,
            config={
                "use_reranker": True,
                "max_memories_per_inject": 2,
                "recall_sparse_fusion": False,
            },
        )
        for doc_id, content, meta in memories:
            await self.mgr.vec_db.insert(content, meta, doc_id)

    async def asyncTearDown(self):
        provider = self.context.provider_manager.inst_map.get(RERANK_PROVIDER_ID)
        if provider is not None:
            await provider.terminate()

    def _rerank_requests(self):
        return [entry for entry in self.server.request_log if entry["kind"] == "rerank"]

    async def test_candidates_within_top_k_skip_rerank(self):
        await self._start(
            [
                ("doc-1", "用户喜欢喝手冲咖啡", _own_meta("facts://m1")),
                ("doc-2", "用户常驻杭州", _own_meta("facts://m2")),
            ]
        )

        memories = await self.mgr.recall_memories(_private_event(), "咖啡", top_k=2)

        self.assertEqual(len(memories), 2)
        self.assertEqual(self._rerank_requests(), [])

    async def test_rerank_orders_candidates_and_excludes_foreign_private_memory(self):
        await self._start(
            [
                ("doc-1", "记忆一 咖啡", _own_meta("facts://m1")),
                ("doc-2", "记忆二 咖啡", _own_meta("facts://m2")),
                ("doc-3", "记忆三 咖啡", _own_meta("facts://m3")),
                ("doc-4", "记忆四 咖啡", _own_meta("facts://m4")),
                (
                    "doc-x",
                    "外人的私密咖啡记忆",
                    make_memory_meta(
                        "facts://foreign",
                        owner="qq_u2",
                        umo="qq:private:u2",
                        created_at=RECENT_CREATED_AT,
                    ),
                ),
            ]
        )
        scores = {"记忆三": 0.9, "记忆二": 0.6, "记忆一": 0.2, "记忆四": 0.1}

        def handler(request):
            results = []
            for index, document in enumerate(request["documents"]):
                score = next(value for key, value in scores.items() if key in document)
                results.append({"index": index, "relevance_score": score})
            return results

        self.server.set_rerank_handler(handler)

        memories = await self.mgr.recall_memories(_private_event(), "咖啡", top_k=2)

        requests = self._rerank_requests()
        self.assertEqual(len(requests), 1)
        documents = requests[0]["documents"]
        self.assertEqual(len(documents), 4)
        self.assertFalse(any("外人" in str(document) for document in documents))
        self.assertEqual(
            [memory["metadata"]["uri"] for memory in memories],
            ["facts://m3", "facts://m2"],
        )


class RecallResilienceTests(RealStackTestCase):
    async def asyncSetUp(self):
        await self.start_stack(
            with_decision=False,
            config={"recall_sparse_fusion": False, "use_reranker": False},
        )

    async def _seed(self, content="用户喜欢喝咖啡", **extra):
        await self.mgr.vec_db.insert(
            content, _own_meta("facts://coffee", **extra), "doc-coffee"
        )

    def _embedding_requests(self):
        return [
            entry for entry in self.server.request_log if entry["kind"] == "embeddings"
        ]

    async def test_scope_channels_share_one_length_retry(self):
        await self._seed()
        before = len(self._embedding_requests())
        self.server.queue_embedding_error(400, "maximum context length exceeded")
        query = "咖啡" * 200

        memories = await self.mgr.recall_memories(_private_event(), query, top_k=3)

        self.assertEqual(
            [memory["metadata"]["uri"] for memory in memories], ["facts://coffee"]
        )
        lengths = [
            len(str(entry["input"])) for entry in self._embedding_requests()[before:]
        ]
        self.assertEqual(lengths.count(200), 1)
        self.assertEqual(lengths.count(400), 3)

    async def test_server_error_keeps_disclosure_hit_without_network_retry(self):
        await self._seed(disclosure="咖啡 饮品")
        await self.mgr.vec_db.insert(
            "用户喜欢喝咖啡",
            make_memory_meta(
                "facts://global-hit",
                scope="global",
                created_at=RECENT_CREATED_AT,
                disclosure="咖啡 饮品",
            ),
            "doc-global-hit",
        )
        before = len(self._embedding_requests())
        self.server.queue_embedding_error(500, "server down")

        memories = await self.mgr.recall_memories(_private_event(), "咖啡", top_k=3)

        self.assertEqual(len(self._embedding_requests()) - before, 3)
        self.assertEqual(
            sorted(memory["metadata"]["uri"] for memory in memories),
            ["facts://coffee", "facts://global-hit"],
        )


if __name__ == "__main__":
    unittest.main()
