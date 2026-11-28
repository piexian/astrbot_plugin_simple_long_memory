"""TTL 生命周期、归属查重、启动恢复、待写缓冲与重建闭环回归。

全部基于真实 AstrBot 组件：DocumentStorage/FaissVecDB（真实 SQLite/FAISS）、
MemoryPlugin/MemoryManager 真实实例，Embedding 请求由 real_stack 的本地 HTTP
测试服务应答，不模拟任何 AstrBot 接口。
"""

from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from astrbot.core.db.vec_db.faiss_impl.document_storage import DocumentStorage
from real_stack import (
    RealStackTestCase,
    make_memory_meta,
    memory_manager_module,
    real_message_event,
)
from sqlalchemy import text as sa_text

MemoryManager = memory_manager_module.MemoryManager
TTL_EXPIRE_SET_CLAUSE = memory_manager_module.TTL_EXPIRE_SET_CLAUSE
TTL_EXPIRE_WHERE_CLAUSE = memory_manager_module.TTL_EXPIRE_WHERE_CLAUSE

COFFEE_VECTOR = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
RUNNING_VECTOR = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
RECENT_CREATED_AT = "2026-11-27T00:00:00+00:00"


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


def _doc_metadata(doc: dict) -> dict:
    raw = doc.get("metadata", {})
    return json.loads(raw) if isinstance(raw, str) else raw


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


class ScopedDedupTests(RealStackTestCase):
    async def asyncSetUp(self):
        await self.start_stack(with_decision=False)
        # 用真实 embedding 服务为不同文本返回精确向量：咖啡 与 跑步 正交
        self.server.set_embedding_vectors(
            {"咖啡": COFFEE_VECTOR, "跑步": RUNNING_VECTOR}
        )
        kb_id = self.mgr._kb_helper.kb.kb_id
        base = {
            "is_memory_record": True,
            "deprecated": False,
            "kb_id": kb_id,
            "created_at": RECENT_CREATED_AT,
        }
        await self.mgr.vec_db.insert(
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
        await self.mgr.vec_db.insert(
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
        await self.mgr.vec_db.insert(
            "用户喜欢咖啡",
            {
                **base,
                "uri": "facts://shared",
                "memory_scope": "personal",
                "owner_user_id": "qq_u1",
                "owner_user_ids": ["qq_u1", "qq_u2"],
                "visibility": "group",
                "umo": "qq:group:g1",
            },
            "shared",
        )
        await self.mgr.vec_db.insert(
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

    async def _dedup(self, content, scope, session_id, umo, owner="", owners=None):
        return await self.mgr.find_scoped_duplicate(
            content=content,
            memory_scope=scope,
            platform_id="qq",
            session_id=session_id,
            umo=umo,
            owner_user_id=owner,
            owner_user_ids=owners,
            vec_db=self.mgr.vec_db,
        )

    async def test_dedup_stays_within_scope_and_owner(self):
        # 同作用域同归属高相似命中
        self.assertEqual(
            await self._dedup("喜欢咖啡", "personal", "u1", "qq:private:u1", "qq_u1"),
            "facts://a",
        )
        # 旧记录归属集合不足时，不能阻止新共同归属者写入
        self.assertIsNone(
            await self._dedup(
                "喜欢咖啡",
                "personal",
                "u1",
                "qq:private:u1",
                owners=["qq_u1", "qq_u3"],
            )
        )
        self.assertEqual(
            await self._dedup(
                "喜欢咖啡",
                "personal",
                "u1",
                "qq:private:u1",
                owners=["qq_u1", "qq_u2"],
            ),
            "facts://shared",
        )
        # 同文不同主不命中（不跨归属查重）
        self.assertIsNone(
            await self._dedup("喜欢咖啡", "personal", "u3", "qq:private:u3", "qq_u3")
        )
        # 群分区只命中本群记忆
        self.assertEqual(
            await self._dedup("喜欢咖啡", "group", "g1", "qq:group:g1"),
            "facts://c",
        )
        self.assertIsNone(await self._dedup("喜欢咖啡", "group", "g2", "qq:group:g2"))
        # 低相似度不命中
        self.assertIsNone(
            await self._dedup("讨厌跑步", "personal", "u1", "qq:private:u1", "qq_u1")
        )
        # global_memory=False 时 personal 查重限定同一会话
        self.mgr.config["global_memory"] = False
        self.assertEqual(
            await self._dedup("喜欢咖啡", "personal", "u1", "qq:private:u1", "qq_u1"),
            "facts://a",
        )
        self.assertIsNone(
            await self._dedup("喜欢咖啡", "personal", "u1", "qq:group:g9", "qq_u1")
        )


class _KvPluginCase(RealStackTestCase):
    """真实 MemoryPlugin + 真实 KV 回调的公共装配方。"""

    async def start_kv_stack(self):
        await self.start_stack(with_decision=False, with_plugin=True)
        self.plugin.plugin_id = f"test/{uuid.uuid4().hex}"
        self.wire_kv(self.mgr)
        return self.mgr

    def wire_kv(self, manager):
        manager._kv_put = self.plugin.put_kv_data
        manager._kv_get = self.plugin.get_kv_data
        manager._kv_delete = self.plugin.delete_kv_data
        self.plugin.memory_mgr = manager


class StartupRecoveryTests(_KvPluginCase):
    async def _record_order(self, manager):
        order = []
        original_connect = manager.connect_kb
        original_cleanup = manager.cleanup_orphan_links
        original_recover = self.plugin._recover_interrupted_rebuild
        original_scheduler = self.plugin._start_scheduler

        async def connect():
            order.append("connect")
            return await original_connect()

        async def cleanup():
            order.append("cleanup")
            return await original_cleanup()

        async def recover():
            order.append("recover")
            return await original_recover()

        async def scheduler():
            order.append("scheduler")
            return await original_scheduler()

        manager.connect_kb = connect
        manager.cleanup_orphan_links = cleanup
        self.plugin._recover_interrupted_rebuild = recover
        self.plugin._start_scheduler = scheduler
        return order

    async def _seed_orphan_link(self, manager):
        await manager.link_manager.add_link(
            "facts://ghost", "facts://missing", "related", reason="孤儿关联"
        )
        self.assertEqual(await manager.link_manager.count_links(), 1)

    async def test_orphan_cleanup_follows_recovery_when_kb_already_connected(self):
        await self.start_kv_stack()
        await self._seed_orphan_link(self.mgr)
        order = await self._record_order(self.mgr)

        await self.plugin.on_loaded()

        self.assertEqual(order, ["recover", "cleanup"])
        self.assertEqual(await self.mgr.link_manager.count_links(), 0)

    async def test_reconnect_path_orders_connect_recover_cleanup_scheduler(self):
        await self.start_kv_stack()
        await self._seed_orphan_link(self.mgr)
        restarted = MemoryManager(self.context.kb_manager, dict(self.config))
        restarted.initialize()
        self.assertFalse(restarted.is_kb_connected)
        self.wire_kv(restarted)
        order = await self._record_order(restarted)

        await self.plugin.on_loaded()

        self.assertTrue(restarted.is_kb_connected)
        self.assertEqual(order, ["connect", "recover", "cleanup", "scheduler"])
        self.assertEqual(await restarted.link_manager.count_links(), 0)


class PendingWriteRecoveryTests(_KvPluginCase):
    async def test_offline_pending_writes_flush_into_real_kb_after_recovery(self):
        await self.start_kv_stack()
        event = real_message_event(
            umo="qq:FriendMessage:u1", sender_id="u1", sender_name="小明"
        )
        self.mgr._rebuilding = True
        uri = await self.mgr.store_memory(event, "用户喜欢喝手冲咖啡", "facts")
        self.assertEqual([item["uri"] for item in self.mgr._pending_writes], [uri])
        persisted = await self.plugin.get_kv_data("rebuild_pending_writes", None)
        self.assertEqual(
            [item["content"] for item in persisted], ["用户喜欢喝手冲咖啡"]
        )
        self.assertEqual(await self.active_docs(), [])
        self.mgr._rebuilding = False

        # 模拟进程重启：新管理器连上真实 KB 后从 KV 恢复缓冲写入
        restarted = MemoryManager(self.context.kb_manager, dict(self.config))
        restarted.initialize()
        await restarted.connect_kb()
        self.wire_kv(restarted)
        await self.plugin.on_loaded()

        self.assertEqual(restarted._pending_writes, [])
        self.assertIsNone(await self.plugin.get_kv_data("rebuild_pending_writes", None))
        docs = await restarted.vec_db.document_storage.get_documents(
            metadata_filters={"uri": uri}, limit=5
        )
        self.assertEqual(len(docs), 1)
        metadata = _doc_metadata(docs[0])
        self.assertTrue(metadata.get("is_memory_record"))
        self.assertIn("手冲咖啡", docs[0]["text"])
        hits = await restarted.vec_db.retrieve("手冲咖啡", k=3)
        self.assertEqual({_doc_metadata(hit.data)["uri"] for hit in hits}, {uri})


class RebuildLifecycleTests(_KvPluginCase):
    async def test_rebuild_keeps_active_memories_and_link_edges(self):
        await self.start_kv_stack()
        kb_id = self.mgr._kb_helper.kb.kb_id
        for doc_id, uri, content in (
            ("doc-coffee", "facts://coffee", "用户喜欢喝手冲咖啡"),
            ("doc-city", "facts://city", "用户常驻杭州"),
        ):
            await self.mgr.vec_db.insert(
                content,
                make_memory_meta(uri, kb_id=kb_id, created_at=RECENT_CREATED_AT),
                doc_id,
            )
        await self.mgr.link_manager.add_link(
            "facts://coffee", "facts://city", "related", reason="同一用户"
        )
        self.assertEqual(await self.mgr.link_manager.count_links(), 1)

        result = await self.mgr.rebuild_memories()

        self.assertFalse(self.mgr.is_rebuilding)
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["success"], 2)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["verification"]["passed"])
        documents = await self.active_docs()
        self.assertEqual(
            {_doc_metadata(doc)["uri"] for doc in documents},
            {"facts://coffee", "facts://city"},
        )
        hits = await self.mgr.vec_db.retrieve("手冲咖啡", k=5)
        self.assertTrue(hits)
        self.assertEqual(await self.mgr.link_manager.count_links(), 1)
        links = await self.mgr.link_manager.get_links_for_uri(
            "facts://coffee", injectable_only=False, min_confidence=0.0, limit=10
        )
        self.assertEqual(
            [(link["target_uri"], link["relation_type"]) for link in links],
            [("facts://city", "related")],
        )


if __name__ == "__main__":
    unittest.main()
