"""后台提取的轻量轮次缓冲表与纯规则对话块聚合。

maintenance_inbox 建在记忆 KB 的 doc.db 中（与接触记录、关联表同库），
按 (umo, conversation_id) 分组、相邻轮次时间窗口与字符上限切块，
不做任何 LLM 分段判断；游标即行状态，终态行由 purge 清理。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from astrbot.api import logger

try:  # 正常插件加载（包内相对导入）
    from ..memory_protocol import UMOInfo
except ImportError:  # 测试环境：仓库根作为顶层目录直接在 sys.path
    from memory_protocol import UMOInfo

INBOX_QUIET_MINUTES = 10
"""末块距当前不足该分钟数且未达字符上限时留待下周期合并。"""

INBOX_MAX_RETRIES = 3
"""单行提取失败重试上限，超过后不再进入提取，保留到清理。"""

INBOX_RETENTION_DAYS = 14
"""终态（processed/skipped/放弃的 failed）行超过该天数后由 purge 清理。"""

INBOX_USER_TEXT_MAX_CHARS = 2000
INBOX_ASSISTANT_BRIEF_MAX_CHARS = 300

STATUS_PENDING = "pending"
STATUS_SKIPPED = "skipped"
STATUS_PROCESSED = "processed"
STATUS_FAILED = "failed"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS maintenance_inbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    umo TEXT NOT NULL,
    platform_id TEXT NOT NULL DEFAULT '',
    session_type TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    conversation_id TEXT NOT NULL DEFAULT '',
    sender_id TEXT NOT NULL DEFAULT '',
    sender_name TEXT NOT NULL DEFAULT '',
    user_text TEXT NOT NULL DEFAULT '',
    assistant_brief TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    realtime_extracted INTEGER NOT NULL DEFAULT 0,
    retry_count INTEGER NOT NULL DEFAULT 0
)
"""

_CREATE_PENDING_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_maintenance_inbox_fetch
ON maintenance_inbox(status, umo, conversation_id, id)
"""

_CREATE_CREATED_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_maintenance_inbox_created
ON maintenance_inbox(status, created_at)
"""


@dataclass(frozen=True)
class InboxRecord:
    """一次正常收口的主 Agent 轮次。"""

    umo: str
    platform_id: str
    session_type: str
    session_id: str
    conversation_id: str
    sender_id: str
    sender_name: str
    user_text: str
    assistant_brief: str
    created_at: str = ""


@dataclass
class ConversationBlock:
    """纯规则聚合产出的对话块，供整理师提取。"""

    conv_key: str  # f"{umo}:{conversation_id}"
    platform_id: str
    umo: str
    conversation_id: str
    session_type: str
    session_id: str
    text: str
    message_count: int
    char_count: int
    first_id: int
    last_id: int
    row_ids: list[int] = field(default_factory=list)
    sender_ids: list[str] = field(default_factory=list)
    user_chars: int = 0
    truncated: bool = False
    realtime_extracted: bool = False

    @property
    def key(self) -> str:
        position = [self.conv_key, self.first_id, self.last_id]
        return hashlib.sha256(json.dumps(position).encode()).hexdigest()


@dataclass(frozen=True)
class _InboxRow:
    """一行缓冲记录的内存表示。"""

    id: int
    umo: str
    platform_id: str
    session_type: str
    session_id: str
    conversation_id: str
    sender_id: str
    sender_name: str
    user_text: str
    assistant_brief: str
    created_at: datetime
    realtime_extracted: bool
    retry_count: int


def _parse_created_at(raw: str) -> datetime:
    """解析存储时间戳；无时区时按 UTC 处理。"""
    try:
        parsed = datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def render_turn_line(row: _InboxRow) -> str:
    """把一轮对话渲染为带时间戳与说话者标签的单行文本。"""
    stamp = row.created_at.strftime("%m-%d %H:%M")
    if row.sender_id:
        sender = f"用户(sender_id={row.sender_id}"
        if row.sender_name:
            sender += f", {row.sender_name}"
        sender += ")"
    else:
        sender = "用户"
    lines = [f"[{stamp}] {sender}: {row.user_text}"]
    if row.assistant_brief:
        lines.append(f"[{stamp}] 助手: {row.assistant_brief}")
    return "\n".join(lines)


class MaintenanceInbox:
    """轮次缓冲表 CRUD 与纯规则聚合，使用 KB 的 SQLite 文档库。"""

    def __init__(self, vec_db: Any) -> None:
        self._vec_db = vec_db
        self._initialized = False
        self._write_lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        """表是否已就绪（KB 未连接时采集方应跳过写入）。"""
        return self._initialized

    async def ensure_table(self) -> bool:
        """幂等创建缓冲表与索引。"""
        if self._initialized:
            return True
        try:
            doc_storage = self._vec_db.document_storage
            async with doc_storage.get_session() as session, session.begin():
                from sqlalchemy import text as sa_text

                await session.execute(sa_text(_CREATE_TABLE_SQL))
                await session.execute(sa_text(_CREATE_PENDING_INDEX_SQL))
                await session.execute(sa_text(_CREATE_CREATED_INDEX_SQL))
            self._initialized = True
            logger.debug("[简单长期记忆] 后台提取缓冲表就绪")
            return True
        except Exception as e:
            logger.warning(f"[简单长期记忆] 后台提取缓冲表创建失败: {e}")
            return False

    async def append(self, record: InboxRecord) -> int | None:
        """写入一轮待提取对话，返回行 id；表未就绪或写入失败返回 None。"""
        if not self._initialized:
            return None
        created_at = record.created_at or datetime.now(timezone.utc).isoformat()
        params = {
            "umo": record.umo,
            "platform_id": record.platform_id,
            "session_type": record.session_type,
            "session_id": record.session_id,
            "conversation_id": record.conversation_id,
            "sender_id": record.sender_id,
            "sender_name": record.sender_name,
            "user_text": record.user_text,
            "assistant_brief": record.assistant_brief,
            "created_at": created_at,
        }
        try:
            doc_storage = self._vec_db.document_storage
            async with (
                self._write_lock,
                doc_storage.get_session() as session,
                session.begin(),
            ):
                from sqlalchemy import text as sa_text

                result = await session.execute(
                    sa_text(
                        "INSERT INTO maintenance_inbox ("
                        "umo, platform_id, session_type, session_id, "
                        "conversation_id, sender_id, sender_name, "
                        "user_text, assistant_brief, created_at"
                        ") VALUES ("
                        ":umo, :platform_id, :session_type, :session_id, "
                        ":conversation_id, :sender_id, :sender_name, "
                        ":user_text, :assistant_brief, :created_at"
                        ")"
                    ),
                    params,
                )
                row_id = int(result.lastrowid or 0)
            logger.debug(
                "[简单长期记忆] 轮次已入缓冲: umo=%s, cid=%s, row=%s",
                record.umo,
                record.conversation_id,
                row_id,
            )
            return row_id
        except Exception as e:
            logger.warning(f"[简单长期记忆] 轮次入缓冲失败: {type(e).__name__}")
            return None

    async def mark_realtime_extracted(self, row_id: int) -> bool:
        """实时提取已实际完成后标记该行，后台整理只做增量补漏。"""
        return await self._execute_write(
            "UPDATE maintenance_inbox SET realtime_extracted = 1 WHERE id = :id",
            {"id": int(row_id)},
        )

    async def _mark_rows_status(self, row_ids: list[int], status: str) -> bool:
        """批量提交行状态（processed/skipped/failed）。"""
        ids = [int(i) for i in row_ids if int(i) > 0]
        if not ids or not self._initialized:
            return False
        placeholders = ", ".join(f":id_{n}" for n in range(len(ids)))
        params = {f"id_{n}": value for n, value in enumerate(ids)}
        return await self._execute_write(
            f"UPDATE maintenance_inbox SET status = '{status}' "
            f"WHERE id IN ({placeholders})",
            params,
        )

    async def mark_rows(self, row_ids: list[int], status: str) -> bool:
        """提交行终态（processed/skipped）。"""
        if status not in (STATUS_PROCESSED, STATUS_SKIPPED):
            return False
        return await self._mark_rows_status(row_ids, status)

    async def mark_failed(self, row_ids: list[int]) -> bool:
        """真实处理失败：状态置 failed 并递增重试计数。"""
        ids = [int(i) for i in row_ids if int(i) > 0]
        if not ids or not self._initialized:
            return False
        placeholders = ", ".join(f":id_{n}" for n in range(len(ids)))
        params = {f"id_{n}": value for n, value in enumerate(ids)}
        return await self._execute_write(
            "UPDATE maintenance_inbox SET status = 'failed', "
            "retry_count = retry_count + 1 "
            f"WHERE id IN ({placeholders})",
            params,
        )

    async def collect_blocks(
        self,
        *,
        gap_minutes: int,
        max_chars: int,
        min_user_chars: int,
        max_blocks: int,
        quiet_minutes: int = INBOX_QUIET_MINUTES,
        max_retries: int = INBOX_MAX_RETRIES,
        commit: bool = True,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """按 (umo, conversation_id) 分组做纯规则切块，返回对话块与统计。

        commit 为 False（dry-run）时不写任何行状态；短块与无效 UMO 行在
        commit 模式下直接标 skipped，不消耗任何 LLM 预算。
        """
        stats: dict[str, Any] = {
            "rows_fetched": 0,
            "invalid_umo_skipped": 0,
            "blocks_emitted": 0,
            "blocks_skipped_short": 0,
            "blocks_held_quiet": 0,
            "block_cap_reached": False,
        }
        result: dict[str, Any] = {"blocks": [], "stats": stats}
        if not self._initialized:
            return result

        rows = await self._fetch_rows(max_retries)
        stats["rows_fetched"] = len(rows)
        if not rows:
            return result

        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        gap = timedelta(minutes=max(1, int(gap_minutes)))
        quiet = timedelta(minutes=max(0, int(quiet_minutes)))
        max_chars = max(64, int(max_chars))
        skipped_ids: list[int] = []

        # SQL 排序保证同 (umo, conversation_id) 的行连续且按 id 升序
        groups: list[tuple[str, list[_InboxRow]]] = []
        for row in rows:
            if not UMOInfo.parse(row.umo).is_valid:
                stats["invalid_umo_skipped"] += 1
                skipped_ids.append(row.id)
                continue
            group_key = f"{row.umo}:{row.conversation_id}"
            if groups and groups[-1][0] == group_key:
                groups[-1][1].append(row)
            else:
                groups.append((group_key, [row]))

        for _, group_rows in groups:
            if stats["block_cap_reached"]:
                break
            blocks = self._split_group(group_rows, gap, max_chars)
            if not blocks:
                continue
            last_row = group_rows[-1]
            last_block_chars = sum(len(render_turn_line(row)) for row in blocks[-1])
            hold_last = (
                current - last_row.created_at < quiet and last_block_chars < max_chars
            )
            if hold_last:
                blocks = blocks[:-1]
                stats["blocks_held_quiet"] += 1
            for block_rows in blocks:
                if len(result["blocks"]) >= max(1, int(max_blocks)):
                    stats["block_cap_reached"] = True
                    logger.info(
                        "[简单长期记忆] 缓冲切块达上限: max=%s，剩余块下周期处理",
                        max_blocks,
                    )
                    break
                block = self._build_block(group_rows[0], block_rows)
                if block.user_chars < max(0, int(min_user_chars)):
                    stats["blocks_skipped_short"] += 1
                    skipped_ids.extend(row.id for row in block_rows)
                    continue
                result["blocks"].append(block)
                stats["blocks_emitted"] += 1

        if commit and skipped_ids:
            await self.mark_rows(skipped_ids, STATUS_SKIPPED)
        logger.debug(
            "[简单长期记忆] 缓冲切块完成: fetched=%s, emitted=%s, short=%s, "
            "quiet=%s, invalid_umo=%s, cap=%s",
            stats["rows_fetched"],
            stats["blocks_emitted"],
            stats["blocks_skipped_short"],
            stats["blocks_held_quiet"],
            stats["invalid_umo_skipped"],
            stats["block_cap_reached"],
        )
        return result

    async def purge_terminal(
        self,
        retention_days: int = INBOX_RETENTION_DAYS,
        *,
        dry_run: bool = False,
    ) -> dict[str, int]:
        """清理超期终态行；pending 行不因超期删除。"""
        result = {"candidates": 0, "purged": 0}
        if not self._initialized:
            return result
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=max(0, int(retention_days)))
        ).isoformat()
        terminal_clause = (
            f"status IN ('{STATUS_PROCESSED}', '{STATUS_SKIPPED}') "
            f"OR (status = '{STATUS_FAILED}' AND retry_count >= {INBOX_MAX_RETRIES})"
        )
        try:
            doc_storage = self._vec_db.document_storage
            async with doc_storage.get_session() as session:
                from sqlalchemy import text as sa_text

                count = (
                    await session.execute(
                        sa_text(
                            "SELECT COUNT(*) FROM maintenance_inbox "
                            f"WHERE ({terminal_clause}) AND created_at < :cutoff"
                        ),
                        {"cutoff": cutoff},
                    )
                ).scalar()
                result["candidates"] = int(count or 0)
                if dry_run or not result["candidates"]:
                    return result
            async with (
                self._write_lock,
                doc_storage.get_session() as session,
                session.begin(),
            ):
                from sqlalchemy import text as sa_text

                deleted = await session.execute(
                    sa_text(
                        "DELETE FROM maintenance_inbox "
                        f"WHERE ({terminal_clause}) AND created_at < :cutoff"
                    ),
                    {"cutoff": cutoff},
                )
                result["purged"] = max(0, deleted.rowcount or 0)
        except Exception as e:
            logger.warning(f"[简单长期记忆] 缓冲表清理失败: {type(e).__name__}")
        if result["purged"]:
            logger.info(
                "[简单长期记忆] 缓冲表清理完成: %s 行（候选 %s）",
                result["purged"],
                result["candidates"],
            )
        return result

    async def status_counts(self) -> dict[str, int]:
        """按状态统计行数，仅供诊断与测试。"""
        if not self._initialized:
            return {}
        try:
            doc_storage = self._vec_db.document_storage
            async with doc_storage.get_session() as session:
                from sqlalchemy import text as sa_text

                rows = (
                    await session.execute(
                        sa_text(
                            "SELECT status, COUNT(*) FROM maintenance_inbox "
                            "GROUP BY status"
                        )
                    )
                ).all()
                return {str(status): int(count or 0) for status, count in rows}
        except Exception:
            return {}

    async def _fetch_rows(self, max_retries: int) -> list[_InboxRow]:
        """拉取可提取行：pending 或重试额度内的 failed，按 (umo, cid, id) 排序。"""
        try:
            doc_storage = self._vec_db.document_storage
            async with doc_storage.get_session() as session:
                from sqlalchemy import text as sa_text

                rows = (
                    await session.execute(
                        sa_text(
                            "SELECT id, umo, platform_id, session_type, session_id, "
                            "conversation_id, sender_id, sender_name, user_text, "
                            "assistant_brief, created_at, realtime_extracted, "
                            "retry_count FROM maintenance_inbox "
                            "WHERE status IN ('pending', 'failed') "
                            "AND retry_count < :max_retries "
                            "ORDER BY umo, conversation_id, id"
                        ),
                        {"max_retries": max(1, int(max_retries))},
                    )
                ).all()
                return [
                    _InboxRow(
                        id=int(row.id),
                        umo=str(row.umo),
                        platform_id=str(row.platform_id),
                        session_type=str(row.session_type),
                        session_id=str(row.session_id),
                        conversation_id=str(row.conversation_id),
                        sender_id=str(row.sender_id),
                        sender_name=str(row.sender_name),
                        user_text=str(row.user_text),
                        assistant_brief=str(row.assistant_brief),
                        created_at=_parse_created_at(row.created_at),
                        realtime_extracted=bool(row.realtime_extracted),
                        retry_count=int(row.retry_count),
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.warning(f"[简单长期记忆] 缓冲表读取失败: {type(e).__name__}")
            return []

    @staticmethod
    def _split_group(
        rows: list[_InboxRow], gap: timedelta, max_chars: int
    ) -> list[list[_InboxRow]]:
        """单组内按时间间隙与字符上限切块。"""
        blocks: list[list[_InboxRow]] = []
        current: list[_InboxRow] = []
        current_chars = 0
        prev: _InboxRow | None = None
        for row in rows:
            line_len = len(render_turn_line(row))
            beyond_gap = prev is not None and row.created_at - prev.created_at > gap
            if current and (beyond_gap or current_chars + line_len > max_chars):
                blocks.append(current)
                current = []
                current_chars = 0
            current.append(row)
            current_chars += line_len
            prev = row
        if current:
            blocks.append(current)
        return blocks

    @staticmethod
    def _build_block(head: _InboxRow, rows: list[_InboxRow]) -> ConversationBlock:
        sender_ids = list(dict.fromkeys(row.sender_id for row in rows if row.sender_id))
        return ConversationBlock(
            conv_key=f"{head.umo}:{head.conversation_id}",
            platform_id=head.platform_id,
            umo=head.umo,
            conversation_id=head.conversation_id,
            session_type=head.session_type,
            session_id=head.session_id,
            text="\n".join(render_turn_line(row) for row in rows),
            message_count=len(rows),
            char_count=sum(len(render_turn_line(row)) for row in rows),
            first_id=rows[0].id,
            last_id=rows[-1].id,
            row_ids=[row.id for row in rows],
            sender_ids=sender_ids,
            user_chars=sum(len(row.user_text) for row in rows),
            truncated=False,
            realtime_extracted=any(row.realtime_extracted for row in rows),
        )

    async def _execute_write(self, statement: str, params: dict[str, Any]) -> bool:
        if not self._initialized:
            return False
        try:
            doc_storage = self._vec_db.document_storage
            async with (
                self._write_lock,
                doc_storage.get_session() as session,
                session.begin(),
            ):
                from sqlalchemy import text as sa_text

                await session.execute(sa_text(statement), params)
            return True
        except Exception as e:
            logger.warning(f"[简单长期记忆] 缓冲表状态更新失败: {type(e).__name__}")
            return False
