"""记忆关联表管理。

独立 SQLite 表存储记忆之间的显式关联关系，
支持关联发现、矛盾检测、级联清理。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from astrbot.api import logger

if TYPE_CHECKING:
    pass

# 关联类型
RELATION_RELATED = "related"  # 主题/场景相关
RELATION_SUPPORTS = "supports"  # 互相支撑
RELATION_CONTEXT = "context"  # 上下文关联
RELATION_CONTRADICTS = "contradicts"  # 矛盾
RELATION_SUPERSEDES = "supersedes"  # 新记忆取代旧记忆

# 可注入召回的关联类型（contradicts/supersedes 仅用于后台分析，不注入）
INJECTABLE_RELATIONS = frozenset(
    (RELATION_RELATED, RELATION_SUPPORTS, RELATION_CONTEXT)
)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS memory_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_uri TEXT NOT NULL,
    target_uri TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    reason TEXT DEFAULT '',
    confidence REAL DEFAULT 1.0,
    created_by TEXT DEFAULT 'analyst',
    created_at TEXT NOT NULL,
    UNIQUE(source_uri, target_uri, relation_type)
)
"""

_CREATE_INDEX_SOURCE_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_links_source ON memory_links(source_uri)"
)
_CREATE_INDEX_TARGET_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_links_target ON memory_links(target_uri)"
)


class MemoryLinkManager:
    """记忆关联表 CRUD，使用 KB 的 SQLite 数据库。"""

    def __init__(self, vec_db: Any) -> None:
        self._vec_db = vec_db
        self._initialized = False

    async def ensure_table(self) -> None:
        """确保关联表和索引存在（幂等）。"""
        if self._initialized:
            return
        try:
            doc_storage = self._vec_db.document_storage
            async with doc_storage.get_session() as session, session.begin():
                from sqlalchemy import text as sa_text

                await session.execute(sa_text(_CREATE_TABLE_SQL))
                await session.execute(sa_text(_CREATE_INDEX_SOURCE_SQL))
                await session.execute(sa_text(_CREATE_INDEX_TARGET_SQL))
            self._initialized = True
            logger.debug("[简单长期记忆] 关联表就绪")
        except Exception as e:
            logger.warning(f"[简单长期记忆] 关联表创建失败: {e}")

    async def add_link(
        self,
        source_uri: str,
        target_uri: str,
        relation_type: str,
        reason: str = "",
        confidence: float = 1.0,
        created_by: str = "analyst",
    ) -> bool:
        """添加一条关联（幂等，重复插入忽略）。"""
        if not source_uri or not target_uri:
            return False
        try:
            doc_storage = self._vec_db.document_storage
            async with doc_storage.get_session() as session, session.begin():
                from sqlalchemy import text as sa_text

                await session.execute(
                    sa_text(
                        "INSERT OR IGNORE INTO memory_links "
                        "(source_uri, target_uri, relation_type, reason, "
                        " confidence, created_by, created_at) "
                        "VALUES (:src, :tgt, :rel, :reason, :conf, :by, :at)"
                    ),
                    {
                        "src": source_uri,
                        "tgt": target_uri,
                        "rel": relation_type,
                        "reason": reason,
                        "conf": confidence,
                        "by": created_by,
                        "at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            return True
        except Exception as e:
            logger.warning(f"[简单长期记忆] 添加关联失败: {e}")
            return False

    async def get_links_for_uri(
        self,
        uri: str,
        injectable_only: bool = True,
        min_confidence: float = 0.0,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """查询某条记忆的关联（单跳，不递归）。

        Args:
            uri: 记忆 URI
            injectable_only: 只返回可注入召回的关联类型
            min_confidence: 最低置信度
            limit: 最大返回条数
        """
        try:
            doc_storage = self._vec_db.document_storage
            async with doc_storage.get_session() as session:
                from sqlalchemy import text as sa_text

                rel_filter = ""
                if injectable_only:
                    placeholders = ",".join(
                        f":rel{i}" for i in range(len(INJECTABLE_RELATIONS))
                    )
                    rel_filter = f" AND relation_type IN ({placeholders})"

                rows = (
                    await session.execute(
                        sa_text(
                            "SELECT target_uri, relation_type, reason, confidence "
                            "FROM memory_links "
                            "WHERE source_uri = :uri"
                            f"{rel_filter}"
                            " AND confidence >= :min_conf"
                            " ORDER BY confidence DESC"
                            + (" LIMIT :lim" if limit > 0 else "")
                        ),
                        {
                            "uri": uri,
                            "min_conf": min_confidence,
                            **({"lim": limit} if limit > 0 else {}),
                            **{
                                f"rel{i}": r for i, r in enumerate(INJECTABLE_RELATIONS)
                            },
                        },
                    )
                ).all()
                return [
                    {
                        "target_uri": row.target_uri,
                        "relation_type": row.relation_type,
                        "reason": row.reason,
                        "confidence": row.confidence,
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.debug(f"[简单长期记忆] 查询关联失败: {e}")
            return []

    async def get_links_to_uri(
        self,
        uri: str,
        injectable_only: bool = True,
        min_confidence: float = 0.0,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """查询指向某条记忆的入边关联。"""
        try:
            doc_storage = self._vec_db.document_storage
            async with doc_storage.get_session() as session:
                from sqlalchemy import text as sa_text

                rel_filter = ""
                if injectable_only:
                    placeholders = ",".join(
                        f":rel{i}" for i in range(len(INJECTABLE_RELATIONS))
                    )
                    rel_filter = f" AND relation_type IN ({placeholders})"

                rows = (
                    await session.execute(
                        sa_text(
                            "SELECT source_uri, relation_type, reason, confidence "
                            "FROM memory_links "
                            "WHERE target_uri = :uri"
                            f"{rel_filter}"
                            " AND confidence >= :min_conf"
                            " ORDER BY confidence DESC"
                            + (" LIMIT :lim" if limit > 0 else "")
                        ),
                        {
                            "uri": uri,
                            "min_conf": min_confidence,
                            **({"lim": limit} if limit > 0 else {}),
                            **{
                                f"rel{i}": r for i, r in enumerate(INJECTABLE_RELATIONS)
                            },
                        },
                    )
                ).all()
                return [
                    {
                        "source_uri": row.source_uri,
                        "relation_type": row.relation_type,
                        "reason": row.reason,
                        "confidence": row.confidence,
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.debug(f"[简单长期记忆] 查询入边关联失败: {e}")
            return []

    async def delete_links_for_uris(self, uris: list[str]) -> int:
        """级联删除涉及指定 URI 的所有关联边。"""
        uris = [u for u in uris if u]
        if not uris:
            return 0
        try:
            doc_storage = self._vec_db.document_storage
            async with doc_storage.get_session() as session, session.begin():
                from sqlalchemy import text as sa_text

                placeholders = ",".join(f":u{i}" for i in range(len(uris)))
                params = {f"u{i}": u for i, u in enumerate(uris)}
                result = await session.execute(
                    sa_text(
                        f"DELETE FROM memory_links "
                        f"WHERE source_uri IN ({placeholders}) "
                        f"   OR target_uri IN ({placeholders})"
                    ),
                    params,
                )
                return result.rowcount or 0
        except Exception as e:
            logger.warning(f"[简单长期记忆] 级联删除关联失败: {e}")
            return 0

    async def count_links(self) -> int:
        """统计关联总数。"""
        try:
            doc_storage = self._vec_db.document_storage
            async with doc_storage.get_session() as session:
                from sqlalchemy import text as sa_text

                row = (
                    await session.execute(sa_text("SELECT COUNT(*) FROM memory_links"))
                ).scalar()
                return int(row or 0)
        except Exception:
            return 0

    async def export_all(self) -> list[dict[str, Any]]:
        """导出所有关联记录（用于 KB 迁移）。"""
        try:
            doc_storage = self._vec_db.document_storage
            async with doc_storage.get_session() as session:
                from sqlalchemy import text as sa_text

                rows = (
                    await session.execute(
                        sa_text(
                            "SELECT source_uri, target_uri, relation_type, "
                            "reason, confidence, created_by, created_at "
                            "FROM memory_links"
                        )
                    )
                ).all()
                return [
                    {
                        "source_uri": r.source_uri,
                        "target_uri": r.target_uri,
                        "relation_type": r.relation_type,
                        "reason": r.reason,
                        "confidence": r.confidence,
                        "created_by": r.created_by,
                        "created_at": r.created_at,
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.warning(f"[简单长期记忆] 导出关联失败: {e}")
            return []

    async def import_all(self, records: list[dict[str, Any]]) -> int:
        """导入关联记录（用于 KB 迁移）。"""
        if not records:
            return 0
        imported = 0
        try:
            doc_storage = self._vec_db.document_storage
            async with doc_storage.get_session() as session, session.begin():
                from sqlalchemy import text as sa_text

                for rec in records:
                    await session.execute(
                        sa_text(
                            "INSERT OR IGNORE INTO memory_links "
                            "(source_uri, target_uri, relation_type, reason, "
                            " confidence, created_by, created_at) "
                            "VALUES (:src, :tgt, :rel, :reason, :conf, :by, :at)"
                        ),
                        {
                            "src": rec["source_uri"],
                            "tgt": rec["target_uri"],
                            "rel": rec["relation_type"],
                            "reason": rec.get("reason", ""),
                            "conf": rec.get("confidence", 1.0),
                            "by": rec.get("created_by", "migration"),
                            "at": rec.get("created_at", ""),
                        },
                    )
                    imported += 1
        except Exception as e:
            logger.warning(f"[简单长期记忆] 导入关联失败: {e}")
        return imported
