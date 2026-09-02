"""召回路径短路测试：零记忆短路（P3）、rerank 短路（P2）、注入路径短路。"""

import unittest
from types import SimpleNamespace

from test_memory_commands import _Event, memory_manager_module, plugin_main

MemoryManager = memory_manager_module.MemoryManager


class _FakeDocStorage:
    """最小 document_storage：只支持 get_documents；缺 get_session 时 disclosure 通道会优雅降级为空。"""

    def __init__(self, docs: list[dict]):
        self._docs = docs
        self.get_documents_calls: list[dict] = []

    async def get_documents(self, **kwargs):
        self.get_documents_calls.append(kwargs)
        return list(self._docs)


class _FakeRetrieveResult:
    def __init__(self, text: str, uri: str, similarity: float):
        self.data = {
            "text": text,
            "metadata": {"uri": uri, "created_at": "2026-01-01T00:00:00+00:00"},
        }
        self.similarity = similarity


class _FakeRerankProvider:
    def __init__(self):
        self.calls: list[list[str]] = []

    async def rerank(self, query, docs):
        self.calls.append(list(docs))
        return [
            SimpleNamespace(index=i, relevance_score=1.0 - i * 0.1)
            for i in range(len(docs))
        ]


class _FakeVecDB:
    def __init__(self, docs: list[dict], retrieve_results: list):
        self.document_storage = _FakeDocStorage(docs)
        self._retrieve_results = retrieve_results
        self.retrieve_calls = 0
        self.rerank_provider = None

    async def retrieve(self, **kwargs):
        self.retrieve_calls += 1
        return list(self._retrieve_results)


def _manager(docs, retrieve_results=(), config=None):
    manager = MemoryManager(
        kb_mgr=None,
        config={"recall_sparse_fusion": False, **(config or {})},
    )
    vec_db = _FakeVecDB(docs, retrieve_results)
    manager._kb_helper = SimpleNamespace(vec_db=vec_db)
    # 避免测试环境真实加载 AstrBot 分词器；分词缓存置 False 时回退空格分词
    manager._sparse_tokenize = False
    return manager, vec_db


_ACTIVE_DOC = {
    "id": "d1",
    "text": "memory: 用户喜欢苹果",
    "metadata": {"uri": "facts://fruit", "is_memory_record": True},
}


class EmptyPoolShortCircuitTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_pool_returns_early_without_retrieval(self):
        manager, vec_db = _manager(docs=[], retrieve_results=[])

        result = await manager.recall_memories(
            _Event("", admin=True), "苹果", all_users=True, source="unit_test"
        )

        self.assertEqual(result, [])
        self.assertEqual(vec_db.retrieve_calls, 0)
        # 短路判定本身只发一次 limit=1 的轻量探测
        probes = [
            call
            for call in vec_db.document_storage.get_documents_calls
            if call.get("limit") == 1
        ]
        self.assertEqual(len(probes), 1)
        self.assertEqual(
            probes[0]["metadata_filters"],
            {"is_memory_record": True, "deprecated": False},
        )

    async def test_non_empty_pool_recalls_normally(self):
        manager, vec_db = _manager(
            docs=[_ACTIVE_DOC],
            retrieve_results=[
                _FakeRetrieveResult("memory: 用户喜欢苹果", "facts://fruit", 0.9)
            ],
            config={"use_reranker": False},
        )

        result = await manager.recall_memories(
            _Event("", admin=True), "苹果", all_users=True, source="unit_test"
        )

        self.assertEqual(vec_db.retrieve_calls, 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["metadata"]["uri"], "facts://fruit")


class RerankShortCircuitTests(unittest.IsolatedAsyncioTestCase):
    async def test_candidates_within_final_top_k_skip_rerank(self):
        manager, vec_db = _manager(
            docs=[_ACTIVE_DOC],
            retrieve_results=[
                _FakeRetrieveResult(f"memory: 记忆 {i}", f"facts://m{i}", 0.9 - i * 0.1)
                for i in range(2)
            ],
            config={"use_reranker": True, "max_memories_per_inject": 2},
        )
        vec_db.rerank_provider = _FakeRerankProvider()

        result = await manager.recall_memories(
            _Event("", admin=True), "苹果", all_users=True, top_k=2, source="unit_test"
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(vec_db.rerank_provider.calls, [])

    async def test_candidates_above_final_top_k_still_rerank(self):
        manager, vec_db = _manager(
            docs=[_ACTIVE_DOC],
            retrieve_results=[
                _FakeRetrieveResult(f"memory: 记忆 {i}", f"facts://m{i}", 0.9 - i * 0.1)
                for i in range(3)
            ],
            config={"use_reranker": True, "max_memories_per_inject": 2},
        )
        vec_db.rerank_provider = _FakeRerankProvider()

        result = await manager.recall_memories(
            _Event("", admin=True), "苹果", all_users=True, top_k=2, source="unit_test"
        )

        self.assertEqual(len(vec_db.rerank_provider.calls), 1)
        self.assertEqual(len(vec_db.rerank_provider.calls[0]), 3)
        self.assertEqual(len(result), 2)


async def _noop(*args, **kwargs):
    return None


class InjectShortCircuitTests(unittest.IsolatedAsyncioTestCase):
    def _plugin(self, has_any: bool):
        optimize_calls = []
        recall_calls = []

        class Manager:
            async def has_any_active_memory(self):
                return has_any

            async def recall_memories(self, **kwargs):
                recall_calls.append(kwargs)
                return []

        async def optimize(event, query):
            optimize_calls.append(query)
            return f"优化后:{query}"

        plugin = SimpleNamespace(
            config={"auto_memorize": True, "optimize_recall_query": True},
            memory_mgr=Manager(),
            _background_tasks=set(),
            _maybe_expire_stale_memories=_noop,
            _maybe_consolidate_memories=_noop,
            _accumulate_request_snapshot=lambda event, request: None,
            _optimize_recall_query=optimize,
        )
        return plugin, optimize_calls, recall_calls

    async def test_empty_pool_skips_query_optimization_and_recall(self):
        plugin, optimize_calls, recall_calls = self._plugin(has_any=False)
        request = SimpleNamespace(prompt="你好", contexts=[])

        await plugin_main.MemoryPlugin.inject_memories(plugin, _Event("你好"), request)

        self.assertEqual(optimize_calls, [])
        self.assertEqual(recall_calls, [])

    async def test_non_empty_pool_runs_optimization_and_recall(self):
        plugin, optimize_calls, recall_calls = self._plugin(has_any=True)
        request = SimpleNamespace(prompt="你好", contexts=[])

        await plugin_main.MemoryPlugin.inject_memories(plugin, _Event("你好"), request)

        self.assertEqual(optimize_calls, ["你好"])
        self.assertEqual(len(recall_calls), 1)
        self.assertEqual(recall_calls[0]["query"], "优化后:你好")


if __name__ == "__main__":
    unittest.main()
