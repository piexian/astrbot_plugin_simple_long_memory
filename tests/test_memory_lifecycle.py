"""TTL 生命周期与自动提取同归属查重回归。

全部基于真实 AstrBot 组件：DocumentStorage/FaissVecDB（真实 SQLite/FAISS）、
OpenAIEmbeddingProvider 经本地 HTTP 测试服务驱动，不模拟任何 AstrBot 接口。
"""

import json
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from astrbot.core.db.vec_db.faiss_impl.document_storage import DocumentStorage
from astrbot.core.db.vec_db.faiss_impl.vec_db import FaissVecDB
from astrbot.core.provider.sources.openai_embedding_source import (
    OpenAIEmbeddingProvider,
)
from sqlalchemy import text as sa_text
from test_memory_commands import MemoryManager, memory_manager_module

TTL_EXPIRE_SET_CLAUSE = memory_manager_module.TTL_EXPIRE_SET_CLAUSE
TTL_EXPIRE_WHERE_CLAUSE = memory_manager_module.TTL_EXPIRE_WHERE_CLAUSE


def _meta(
    uri: str,
    *,
    days_old: float,
    domain: str | None = "facts",
    scope: str = "personal",
    recall_count: int = 0,
    last_recalled_at: str | None = None,
    memory_type: str = "normal",
) -> dict:
    created = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    meta = {
        "uri": uri,
        "is_memory_record": True,
        "deprecated": False,
        "kb_id": "kb1",
        "created_at": created,
        "memory_type": memory_type,
        "memory_scope": scope,
        "recall_count": recall_count,
    }
    if domain is not None:
        meta["domain"] = domain
    if last_recalled_at is not None:
        meta["last_recalled_at"] = last_recalled_at
    return meta


class TtlExpiryTests(unittest.IsolatedAsyncioTestCase):
    async def test_ttl_where_clause_boundaries_on_real_document_storage(self):
        cutoff_iso = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        now_iso = datetime.now(timezone.utc).isoformat()
        recent = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        metas = [
            # 高频 fact/preference 豁免
            _meta("facts://hot-fact", days_old=40, recall_count=10),
            _meta(
                "facts://hot-pref", days_old=40, domain="preferences", recall_count=5
            ),
            _meta("facts://legacy-alias", days_old=40, domain="fact", recall_count=6),
            # 高频但非核心域不豁免
            _meta("facts://hot-event", days_old=40, domain="events", recall_count=10),
            # 低频 fact 正常过期
            _meta("facts://cold-fact", days_old=40, recall_count=1),
            # 近期有召回活动的旧记忆不过期
            _meta(
                "facts://active-event",
                days_old=40,
                domain="events",
                recall_count=2,
                last_recalled_at=recent,
            ),
            # last_recalled_at 为空串时回退 created_at
            _meta("facts://empty-recalled", days_old=40, last_recalled_at=""),
            # 既有豁免不变
            _meta("facts://permanent", days_old=40, memory_type="permanent"),
            _meta("facts://global", days_old=40, scope="global"),
            # 缺失 domain 不享受豁免
            _meta("facts://no-domain", days_old=40, domain=None, recall_count=10),
        ]
        with tempfile.TemporaryDirectory() as directory:
            storage = DocumentStorage(str(Path(directory) / "documents.db"))
            await storage.initialize()
            for index, meta in enumerate(metas):
                await storage.insert_document(f"doc-{index}", "t", meta)
            async with storage.get_session() as session, session.begin():
                result = await session.execute(
                    sa_text(
                        f"UPDATE documents SET metadata = {TTL_EXPIRE_SET_CLAUSE} "
                        f"WHERE {TTL_EXPIRE_WHERE_CLAUSE}"
                    ),
                    {"kb_id": "kb1", "cutoff": cutoff_iso, "now_iso": now_iso},
                )
                marked = int(result.rowcount or 0)
            async with storage.get_session() as session:
                rows = (
                    await session.execute(sa_text("SELECT metadata FROM documents"))
                ).all()
        deprecated = set()
        for (raw,) in rows:
            meta = json.loads(raw)
            if meta.get("deprecated") == 1:
                deprecated.add(meta["uri"])
        self.assertEqual(
            deprecated,
            {
                "facts://hot-event",
                "facts://cold-fact",
                "facts://empty-recalled",
                "facts://no-domain",
            },
        )
        self.assertEqual(marked, len(deprecated))


class _EmbeddingsHandler(BaseHTTPRequestHandler):
    """本地 OpenAI embeddings 测试服务：按文本内容返回确定性向量。"""

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        texts = payload.get("input", [])
        if isinstance(texts, str):
            texts = [texts]
        data = [
            {
                "object": "embedding",
                "index": index,
                "embedding": [1.0, 0.0] if "咖啡" in str(text) else [0.0, 1.0],
            }
            for index, text in enumerate(texts)
        ]
        body = json.dumps(
            {
                "object": "list",
                "data": data,
                "model": payload.get("model", "test-model"),
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class ScopedDedupTests(unittest.IsolatedAsyncioTestCase):
    async def test_dedup_stays_within_scope_and_owner(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _EmbeddingsHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        port = server.server_address[1]

        provider = OpenAIEmbeddingProvider(
            {
                "id": "test-embed",
                "embedding_api_base": f"http://127.0.0.1:{port}/v1",
                "embedding_api_key": "test-key",
                "embedding_model": "test-model",
                "embedding_dimensions": "2",
                "timeout": 10,
            },
            {},
        )
        self.addAsyncCleanup(provider.terminate)
        with tempfile.TemporaryDirectory() as directory:
            vec_db = FaissVecDB(
                str(Path(directory) / "documents.db"),
                str(Path(directory) / "vectors.faiss"),
                provider,
            )
            await vec_db.initialize()
            self.addAsyncCleanup(vec_db.close)
            mgr = MemoryManager(None, {})
            base = {"is_memory_record": True, "deprecated": False, "kb_id": "kb1"}
            await vec_db.insert(
                "用户喜欢咖啡",
                {
                    **base,
                    "uri": "facts://a",
                    "memory_scope": "personal",
                    "owner_user_id": "qq_u1",
                    "umo": "qq:private:u1",
                },
                "a",
            )
            await vec_db.insert(
                "用户喜欢咖啡",
                {
                    **base,
                    "uri": "facts://b",
                    "memory_scope": "personal",
                    "owner_user_id": "qq_u2",
                    "umo": "qq:private:u2",
                },
                "b",
            )
            await vec_db.insert(
                "群友都喜欢咖啡",
                {
                    **base,
                    "uri": "facts://c",
                    "memory_scope": "group",
                    "owner_session_id": "qq_g1",
                    "umo": "qq:group:g1",
                },
                "c",
            )

            async def dedup(content, scope, session_id, umo, owner=""):
                return await mgr.find_scoped_duplicate(
                    content=content,
                    memory_scope=scope,
                    platform_id="qq",
                    session_id=session_id,
                    umo=umo,
                    owner_user_id=owner,
                    vec_db=vec_db,
                )

            # 同作用域同归属高相似命中
            self.assertEqual(
                await dedup("喜欢咖啡", "personal", "u1", "qq:private:u1", "qq_u1"),
                "facts://a",
            )
            # 同文不同主不命中（不跨归属查重）
            self.assertIsNone(
                await dedup("喜欢咖啡", "personal", "u3", "qq:private:u3", "qq_u3")
            )
            # 群分区只命中本群记忆
            self.assertEqual(
                await dedup("喜欢咖啡", "group", "g1", "qq:group:g1"),
                "facts://c",
            )
            self.assertIsNone(await dedup("喜欢咖啡", "group", "g2", "qq:group:g2"))
            # 低相似度不命中
            self.assertIsNone(
                await dedup("讨厌跑步", "personal", "u1", "qq:private:u1", "qq_u1")
            )
            # global_memory=False 时 personal 查重限定同一会话
            mgr.config["global_memory"] = False
            self.assertEqual(
                await dedup("喜欢咖啡", "personal", "u1", "qq:private:u1", "qq_u1"),
                "facts://a",
            )
            self.assertIsNone(
                await dedup("喜欢咖啡", "personal", "u1", "qq:group:g9", "qq_u1")
            )


if __name__ == "__main__":
    unittest.main()
