"""已知联系人接触记录：仅保存结构化会话观察。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from astrbot.api import logger

MAX_CONTACT_SESSIONS_PER_USER = 20

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS contact_observations (
    platform_id TEXT NOT NULL,
    bot_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    umo TEXT NOT NULL,
    session_type TEXT NOT NULL DEFAULT '',
    group_id TEXT NOT NULL DEFAULT '',
    group_name TEXT NOT NULL DEFAULT '',
    sender_name TEXT NOT NULL DEFAULT '',
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    seen_count INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (platform_id, bot_id, user_id, umo)
)
"""

_CREATE_INDEX_USER_SQL = """
CREATE INDEX IF NOT EXISTS idx_contact_user
ON contact_observations(platform_id, user_id, last_seen_at DESC)
"""

_CREATE_INDEX_UMO_SQL = """
CREATE INDEX IF NOT EXISTS idx_contact_umo
ON contact_observations(umo, last_seen_at DESC)
"""


@dataclass(frozen=True, slots=True)
class ContactObservation:
    """一次由消息事件确认的接触事实。"""

    platform_id: str
    bot_id: str
    user_id: str
    umo: str
    session_type: str
    group_id: str = ""
    group_name: str = ""
    sender_name: str = ""
    observed_at: str = ""


class ContactIndex:
    """接触记录 CRUD，使用 KB 的 SQLite 数据库。"""

    def __init__(self, vec_db: Any) -> None:
        self._vec_db = vec_db
        self._initialized = False

    async def ensure_table(self) -> bool:
        """幂等创建接触表与索引。"""
        if self._initialized:
            return True
        try:
            doc_storage = self._vec_db.document_storage
            async with doc_storage.get_session() as session, session.begin():
                from sqlalchemy import text as sa_text

                await session.execute(sa_text(_CREATE_TABLE_SQL))
                await session.execute(sa_text(_CREATE_INDEX_USER_SQL))
                await session.execute(sa_text(_CREATE_INDEX_UMO_SQL))
            self._initialized = True
            logger.debug("[简单长期记忆] 接触记录表就绪")
            return True
        except Exception as e:
            logger.warning(f"[简单长期记忆] 接触记录表创建失败: {e}")
            return False

    async def observe(self, item: ContactObservation) -> bool:
        """幂等更新用户在一个会话中的接触记录。"""
        if not self._initialized:
            return False
        if not item.platform_id or not item.user_id or not item.umo:
            return False
        observed_at = item.observed_at or datetime.now(timezone.utc).isoformat()
        params = {
            "platform_id": item.platform_id,
            "bot_id": item.bot_id,
            "user_id": item.user_id,
            "umo": item.umo,
            "session_type": item.session_type,
            "group_id": item.group_id,
            "group_name": item.group_name,
            "sender_name": item.sender_name,
            "observed_at": observed_at,
            "max_sessions": MAX_CONTACT_SESSIONS_PER_USER,
        }
        try:
            doc_storage = self._vec_db.document_storage
            async with doc_storage.get_session() as session, session.begin():
                from sqlalchemy import text as sa_text

                await session.execute(
                    sa_text(
                        """
                        INSERT INTO contact_observations (
                            platform_id, bot_id, user_id, umo, session_type,
                            group_id, group_name, sender_name,
                            first_seen_at, last_seen_at, seen_count
                        ) VALUES (
                            :platform_id, :bot_id, :user_id, :umo, :session_type,
                            :group_id, :group_name, :sender_name,
                            :observed_at, :observed_at, 1
                        )
                        ON CONFLICT(platform_id, bot_id, user_id, umo) DO UPDATE SET
                            session_type = excluded.session_type,
                            group_id = excluded.group_id,
                            group_name = CASE WHEN excluded.group_name <> ''
                                THEN excluded.group_name
                                ELSE contact_observations.group_name END,
                            sender_name = CASE WHEN excluded.sender_name <> ''
                                THEN excluded.sender_name
                                ELSE contact_observations.sender_name END,
                            last_seen_at = excluded.last_seen_at,
                            seen_count = contact_observations.seen_count + 1
                        """
                    ),
                    params,
                )
                await session.execute(
                    sa_text(
                        """
                        DELETE FROM contact_observations
                        WHERE platform_id = :platform_id
                          AND bot_id = :bot_id
                          AND user_id = :user_id
                          AND umo NOT IN (
                              SELECT umo FROM contact_observations
                              WHERE platform_id = :platform_id
                                AND bot_id = :bot_id
                                AND user_id = :user_id
                              ORDER BY last_seen_at DESC
                              LIMIT :max_sessions
                          )
                        """
                    ),
                    params,
                )
            logger.debug(
                "[简单长期记忆] 接触记录已更新: platform=%s, bot=%s, user=%s, session=%s",
                item.platform_id,
                item.bot_id,
                item.user_id,
                item.session_type,
            )
            return True
        except Exception as e:
            logger.debug(f"[简单长期记忆] 接触记录写入失败: {e}")
            return False

    async def list_user_records(
        self,
        *,
        platform_id: str,
        bot_id: str,
        user_id: str,
        include_other_bots: bool = False,
        umo: str = "",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """查询一个用户的已知会话。"""
        conditions = ["platform_id = :platform_id", "user_id = :user_id"]
        params: dict[str, Any] = {"platform_id": platform_id, "user_id": user_id}
        if not include_other_bots:
            conditions.append("bot_id = :bot_id")
            params["bot_id"] = bot_id
        if umo:
            conditions.append("umo = :umo")
            params["umo"] = umo
        return await self._list_records(conditions, params, limit)

    async def list_group_records(
        self,
        *,
        platform_id: str,
        bot_id: str,
        umo: str,
        include_other_bots: bool = False,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """查询当前群中 Bot 见过的用户。"""
        conditions = ["platform_id = :platform_id", "umo = :umo"]
        params: dict[str, Any] = {"platform_id": platform_id, "umo": umo}
        if not include_other_bots:
            conditions.append("bot_id = :bot_id")
            params["bot_id"] = bot_id
        return await self._list_records(conditions, params, limit)

    async def _list_records(
        self, conditions: list[str], params: dict[str, Any], limit: int
    ) -> list[dict[str, Any]]:
        if not self._initialized or limit <= 0:
            return []
        params = {**params, "limit": limit}
        try:
            doc_storage = self._vec_db.document_storage
            async with doc_storage.get_session() as session:
                from sqlalchemy import text as sa_text

                rows = (
                    await session.execute(
                        sa_text(
                            "SELECT * FROM contact_observations "
                            f"WHERE {' AND '.join(conditions)} "
                            "ORDER BY last_seen_at DESC LIMIT :limit"
                        ),
                        params,
                    )
                ).mappings()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.debug(f"[简单长期记忆] 接触记录查询失败: {e}")
            return []

    async def clear_user(self, user_id: str) -> int:
        """删除指定用户在全部会话和 Bot 下的接触记录。"""
        if not self._initialized or not user_id:
            return 0
        return await self._delete(
            "DELETE FROM contact_observations WHERE user_id = :user_id",
            {"user_id": user_id},
        )

    async def clear_all(self) -> int:
        """删除全部接触记录。"""
        if not self._initialized:
            return 0
        return await self._delete("DELETE FROM contact_observations", {})

    async def _delete(self, statement: str, params: dict[str, Any]) -> int:
        try:
            doc_storage = self._vec_db.document_storage
            async with doc_storage.get_session() as session, session.begin():
                from sqlalchemy import text as sa_text

                result = await session.execute(sa_text(statement), params)
                return max(0, result.rowcount or 0)
        except Exception as e:
            logger.warning(f"[简单长期记忆] 接触记录清理失败: {e}")
            return 0
