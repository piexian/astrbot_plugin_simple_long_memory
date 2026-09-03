"""分段员：从平台消息历史切出完整对话块。

核心职责：
1. 按会话游标从 platform_message_history 分段滚动拉取消息
2. 零成本预切（时间间隙）+ LLM 完整性判定，不完整则滚动并入下一段重判
3. 产出 ConversationBlock 与游标/carry 更新
4. ConversationV2 兜底：pmh 无数据的会话（私聊等）按消息数切段，
   游标用内容锚点续跑

本批次只返回结果不写 KV，游标/carry 落盘由 runner 负责。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from astrbot.api import logger

from ..prompts import SEGMENT_BOUNDARY_PROMPT, build_prompt

SEGMENT_PAGE_SIZE = 100
CURSOR_KV_KEY = "maintenance_extract_cursors"
CARRY_KV_KEY = "maintenance_extract_carry"


@dataclass
class ConversationBlock:
    """一段完整闭环的对话块。"""

    conv_key: str  # f"{platform_id}:{user_id}"
    platform_id: str
    umo: str  # user_id 原值（UMO 字符串）
    text: str  # 扁平化文本
    message_count: int
    char_count: int  # len(text)，供下游估算体积
    truncated: bool  # 连续多周期未闭环被强制截断产出
    first_id: int  # 首行 id（conv2 路径为消息下标）
    last_id: int  # 末行 id（游标推进目标；conv2 路径为消息下标）
    source: str = "pmh"  # pmh=平台消息历史，conv2=ConversationV2 兜底
    end_anchor: str = ""  # conv2 路径填块末消息内容锚点，pmh 为空


@dataclass
class _Candidate:
    """预切出的候选块，可滚动并入下一段扩展。"""

    first_id: int
    last_id: int
    last_ts: datetime
    lines: list[str] = field(default_factory=list)
    body_chars: int = 0  # 纯文本字符数（不含时间戳/发送者装饰），预过滤用
    first_anchor: str = ""  # 首条消息内容锚点（conv2 carry 起点比对用）
    last_anchor: str = ""  # 末条消息内容锚点（conv2 游标推进目标）

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def absorb(self, other: _Candidate) -> None:
        self.lines.extend(other.lines)
        self.body_chars += other.body_chars
        self.last_id = other.last_id
        self.last_ts = other.last_ts
        self.last_anchor = other.last_anchor


class SegmenterAgent:
    """分段员 Agent。"""

    def __init__(
        self,
        context: Any,
        llm: Any,
        config: dict[str, Any],
        kv_get: Any,
        kv_put: Any,
    ) -> None:
        self._context = context
        self._llm = llm
        self._config = config
        self._kv_get = kv_get
        self._kv_put = kv_put  # 本批次保留参数不用，游标写入在 runner

        self._gap = timedelta(
            minutes=config.get("maintenance_segment_time_gap_minutes", 30)
        )
        self._min_chars = config.get("extraction_min_content_length", 150)
        self._max_chars = config.get("maintenance_segment_max_chars", 8000)
        self._max_extensions = config.get("maintenance_segment_max_extensions", 3)
        self._max_blocks = config.get("maintenance_extract_max_blocks_per_cycle", 20)
        self._model_id = config.get("maintenance_segmenter_model_id", "")
        self._conv2_enabled = config.get("maintenance_extract_conv2_enabled", True)
        self._conv2_chunk = config.get("maintenance_extract_conv2_chunk_messages", 40)

    # ─── 数据访问（薄方法，测试可覆写） ──────────────────────

    def _history_db(self) -> Any:
        mgr = getattr(self._context, "message_history_manager", None)
        return getattr(mgr, "db", None) if mgr is not None else None

    async def _list_active_conversations(
        self, since: datetime
    ) -> list[tuple[str, str]]:
        """枚举 since 之后有消息的会话，返回 [(platform_id, user_id)]。"""
        db = self._history_db()
        if db is None:
            return []
        try:
            from astrbot.core.db.po import PlatformMessageHistory
            from sqlmodel import select

            async with db.get_db() as session:
                stmt = (
                    select(
                        PlatformMessageHistory.platform_id,
                        PlatformMessageHistory.user_id,
                    )
                    .where(PlatformMessageHistory.created_at >= self._naive(since))
                    .distinct()
                )
                rows = (await session.execute(stmt)).all()
                return [(str(r[0]), str(r[1])) for r in rows]
        except Exception as e:
            logger.warning(f"[简单长期记忆] 分段员枚举活跃会话失败: {e}")
            return []

    async def _fetch_rows(
        self, platform_id: str, user_id: str, after_id: int, limit: int
    ) -> list[Any]:
        """按 id > after_id 升序拉取一页消息行。"""
        db = self._history_db()
        if db is None:
            return []
        try:
            from astrbot.core.db.po import PlatformMessageHistory
            from sqlmodel import select

            async with db.get_db() as session:
                stmt = (
                    select(PlatformMessageHistory)
                    .where(
                        PlatformMessageHistory.platform_id == platform_id,
                        PlatformMessageHistory.user_id == user_id,
                        PlatformMessageHistory.id > after_id,
                    )
                    .order_by(PlatformMessageHistory.id)
                    .limit(limit)
                )
                return list((await session.execute(stmt)).scalars().all())
        except Exception as e:
            logger.warning(
                f"[简单长期记忆] 分段员拉取消息失败: "
                f"conv={platform_id}:{user_id}, after={after_id}, err={e}"
            )
            return []

    async def _list_conv2_conversations(self, since_epoch: float) -> list[Any]:
        """枚举 updated_at >= since_epoch 的 ConversationV2 会话（跳过空 user_id）。"""
        mgr = getattr(self._context, "conversation_manager", None)
        if mgr is None:
            return []
        try:
            convs = await mgr.get_conversations()
        except Exception as e:
            logger.warning(f"[简单长期记忆] 分段员枚举 ConversationV2 会话失败: {e}")
            return []
        out: list[Any] = []
        for conv in convs:
            if not str(getattr(conv, "user_id", "") or ""):
                continue
            try:
                updated = int(getattr(conv, "updated_at", 0) or 0)
            except (TypeError, ValueError):
                continue
            # updated_at 为 0 无法判断新鲜度，保守跳过
            if updated < since_epoch:
                continue
            out.append(conv)
        return out

    # ─── 主入口 ─────────────────────────────────────────────

    async def collect_blocks(self, max_llm_calls: int) -> dict[str, Any]:
        """滚动拉取并切分对话块。

        Returns:
            {
                "blocks": [ConversationBlock, ...],
                "cursor_updates": {conv_key: last_id 或 conv2 的 {"source","anchor","idx"}},
                "carry_updates": {conv_key: {...} 或 None(清除)},
                "stats": {...},
            }
        """
        stats: dict[str, Any] = {
            "conversations_scanned": 0,
            "blocks_emitted": 0,
            "blocks_skipped_short": 0,
            "blocks_carried": 0,
            "blocks_truncated": 0,
            "poison_skipped": 0,
            "conv2_conversations": 0,
            "conv2_blocks_emitted": 0,
            "anchor_miss": 0,
            "llm_calls": 0,
            "budget_exhausted": False,
            "block_cap_reached": False,
        }
        result: dict[str, Any] = {
            "blocks": [],
            "cursor_updates": {},
            "carry_updates": {},
            "stats": stats,
        }

        history_db_missing = self._history_db() is None
        if history_db_missing:
            logger.debug(
                "[简单长期记忆] 分段员跳过平台消息历史扫描: context 无 message_history_manager"
            )

        cursors = self._normalize_cursors(
            await self._kv_get(CURSOR_KV_KEY, None) if self._kv_get else None
        )
        carries = self._normalize_carries(
            await self._kv_get(CARRY_KV_KEY, None) if self._kv_get else None
        )

        now = datetime.now(timezone.utc)
        # 游标早于 24h 的会话可能仍有未处理旧消息，窗口要覆盖最早游标
        since = now - timedelta(hours=24)
        for cur in cursors.values():
            ts = self._parse_ts(cur.get("ts", ""))
            if ts is not None and ts < since:
                since = ts

        conversations = (
            [] if history_db_missing else await self._list_active_conversations(since)
        )
        logger.debug(
            "[简单长期记忆] 分段员扫描: since=%s, conversations=%s, cursors=%s",
            since.isoformat(),
            len(conversations),
            len(cursors),
        )

        for platform_id, user_id in conversations:
            if stats["budget_exhausted"] or stats["block_cap_reached"]:
                break
            stats["conversations_scanned"] += 1
            conv_key = f"{platform_id}:{user_id}"
            cursor_id = cursors.get(conv_key, {}).get("id", 0)
            try:
                rows = await self._fetch_all(platform_id, user_id, cursor_id)
                if rows:
                    await self._collect_conversation(
                        platform_id,
                        user_id,
                        segments=self._pre_split(rows),
                        prev_carry=carries.get(conv_key),
                        source="pmh",
                        cursor_id=cursor_id,
                        max_llm_calls=max_llm_calls,
                        now=now,
                        result=result,
                        stats=stats,
                    )
            except Exception as e:
                logger.warning(
                    f"[简单长期记忆] 分段员处理会话失败: conv={conv_key}, err={e}"
                )

        # ConversationV2 兜底：只处理 pmh 窗口内无数据的会话，避免双源重复提取
        if self._conv2_enabled:
            pmh_keys = {f"{p}:{u}" for p, u in conversations}
            conv2_convs = await self._list_conv2_conversations(since.timestamp())
            for conv in conv2_convs:
                if stats["budget_exhausted"] or stats["block_cap_reached"]:
                    break
                platform_id = str(getattr(conv, "platform_id", "") or "")
                user_id = str(getattr(conv, "user_id", "") or "")
                conv_key = f"{platform_id}:{user_id}"
                if conv_key in pmh_keys:
                    continue
                stats["conv2_conversations"] += 1
                try:
                    await self._collect_conv2_conversation(
                        conv,
                        platform_id,
                        user_id,
                        cursors.get(conv_key),
                        carries.get(conv_key),
                        max_llm_calls,
                        now,
                        result,
                        stats,
                    )
                except Exception as e:
                    logger.warning(
                        f"[简单长期记忆] 分段员处理 ConversationV2 会话失败: "
                        f"conv={conv_key}, err={e}"
                    )

        logger.debug(
            "[简单长期记忆] 分段员完成: scanned=%s, emitted=%s, short=%s, "
            "carried=%s, truncated=%s, poison=%s, conv2_scanned=%s, "
            "conv2_emitted=%s, anchor_miss=%s, llm=%s, budget_exhausted=%s",
            stats["conversations_scanned"],
            stats["blocks_emitted"],
            stats["blocks_skipped_short"],
            stats["blocks_carried"],
            stats["blocks_truncated"],
            stats["poison_skipped"],
            stats["conv2_conversations"],
            stats["conv2_blocks_emitted"],
            stats["anchor_miss"],
            stats["llm_calls"],
            stats["budget_exhausted"],
        )
        return result

    # ─── 单会话处理 ─────────────────────────────────────────

    async def _collect_conv2_conversation(
        self,
        conv: Any,
        platform_id: str,
        user_id: str,
        cursor: dict[str, Any] | None,
        prev_carry: dict[str, Any] | None,
        max_llm_calls: int,
        now: datetime,
        result: dict[str, Any],
        stats: dict[str, Any],
    ) -> None:
        """ConversationV2 兜底：扁平化 history → 按消息数切段 → 复用闭环循环。"""
        conv_key = f"{platform_id}:{user_id}"
        items = self._flatten_conv2_history(getattr(conv, "history", None), conv_key)
        if not items:
            return

        # 定位续跑起点：conv2 无行 id，用内容锚点在消息列表中找回进度
        conv2_cursor = (
            cursor
            if isinstance(cursor, dict) and cursor.get("source") == "conv2"
            else None
        )
        if conv2_cursor is None:
            # 首见会话只处理最新一段，防首日积压全量倾倒
            start = max(0, len(items) - self._conv2_chunk)
        else:
            anchor = str(conv2_cursor.get("anchor") or "")
            # 从后往前找：内容重复时取最后出现位置，宁可少提不重复
            hit = -1
            for i in range(len(items) - 1, -1, -1):
                if items[i][3] == anchor:
                    hit = i
                    break
            if hit >= 0:
                start = hit + 1
            else:
                # 锚点丢失（窗口滑动/历史重置）：保守只处理最新一段并重锚
                stats["anchor_miss"] += 1
                logger.warning(
                    "[简单长期记忆] 分段员 ConversationV2 锚点丢失，回退最新一段: "
                    "conv=%s, idx=%s, anchor=%s",
                    conv_key,
                    conv2_cursor.get("idx"),
                    anchor[:8],
                )
                start = max(0, len(items) - self._conv2_chunk)
        if start >= len(items):
            return

        # 单条消息无时间戳，闭环等待判定复用会话 updated_at
        try:
            updated = int(getattr(conv, "updated_at", 0) or 0)
        except (TypeError, ValueError):
            updated = 0
        last_ts = datetime.fromtimestamp(updated, tz=timezone.utc) if updated else now
        segments = self._pre_split_conv2(items, start, last_ts)
        await self._collect_conversation(
            platform_id,
            user_id,
            segments=segments,
            prev_carry=prev_carry,
            source="conv2",
            cursor_id=0,
            max_llm_calls=max_llm_calls,
            now=now,
            result=result,
            stats=stats,
        )

    async def _collect_conversation(
        self,
        platform_id: str,
        user_id: str,
        *,
        segments: list[_Candidate],
        prev_carry: dict[str, Any] | None,
        source: str,
        cursor_id: int,
        max_llm_calls: int,
        now: datetime,
        result: dict[str, Any],
        stats: dict[str, Any],
    ) -> None:
        """候选块闭环循环：LLM 判定 + 滚动扩展 + carry 挂起 + 毒块跳过（双源共享）。"""
        conv_key = f"{platform_id}:{user_id}"
        adv_id: int | None = None  # 游标推进到的末条（pmh=行 id，conv2=消息下标）
        adv_anchor = ""  # conv2 末条消息内容锚点
        new_carry = prev_carry
        idx = 0
        first_block = True  # 只有首个候选块继承 prev_carry（跨周期挂起的块必在块首）
        while idx < len(segments):
            if stats["budget_exhausted"]:
                break
            emitted = (
                stats["blocks_emitted"]
                + stats["conv2_blocks_emitted"]
                + stats["blocks_truncated"]
            )
            if emitted >= self._max_blocks:
                stats["block_cap_reached"] = True
                logger.info(
                    "[简单长期记忆] 分段员产出达上限: max=%s，剩余块下周期处理",
                    self._max_blocks,
                )
                break
            cand = segments[idx]
            idx += 1

            # 预过滤：短闲聊块不消耗 LLM，游标照常推进
            if cand.body_chars < self._min_chars:
                adv_id, adv_anchor = cand.last_id, cand.last_anchor
                stats["blocks_skipped_short"] += 1
                logger.debug(
                    "[简单长期记忆] 分段员预过滤短块: conv=%s, ids=%s-%s, chars=%s",
                    conv_key,
                    cand.first_id,
                    cand.last_id,
                    cand.body_chars,
                )
                continue

            pending = prev_carry if first_block else None
            first_block = False
            if pending:
                # runner 按块提交游标后崩溃重跑/内容变化时，旧 carry 起点已与候选块
                # 错配，视为过期 carry 丢弃并清除（不继承其 cycles/fail_count）
                if source == "conv2":
                    stale = cand.first_anchor != pending.get("start_anchor", "")
                else:
                    stale = cand.first_id != pending.get("start_id")
                if stale:
                    pending = None
                    new_carry = None
            prev_cycles = pending.get("cycles", 0) if pending else 0
            prev_fails = pending.get("fail_count", 0) if pending else 0
            advance = False  # True=继续下一块；False=游标停块首，本会话结束
            extensions = 0

            while True:
                if stats["llm_calls"] >= max_llm_calls or (
                    getattr(self._llm, "remaining_calls", 1) <= 0
                ):
                    stats["budget_exhausted"] = True
                    logger.info(
                        "[简单长期记忆] 分段员 LLM 预算耗尽: conv=%s, calls=%s/%s",
                        conv_key,
                        stats["llm_calls"],
                        max_llm_calls,
                    )
                    break

                system = build_prompt(
                    SEGMENT_BOUNDARY_PROMPT, {"block_text": cand.text}
                )
                verdict = await self._llm.chat_json(
                    system, "只回 JSON。", model_id=self._model_id
                )
                stats["llm_calls"] += 1

                if verdict is None:
                    # LLM 失败/解析失败：连续 2 次视为毒块跳过，否则挂起下周期重试
                    fail_count = prev_fails + 1
                    if fail_count >= 2:
                        logger.warning(
                            "[简单长期记忆] 分段员 LLM 连续失败跳过对话块: "
                            "conv=%s, ids=%s-%s",
                            conv_key,
                            cand.first_id,
                            cand.last_id,
                        )
                        adv_id, adv_anchor = cand.last_id, cand.last_anchor
                        new_carry = None
                        stats["poison_skipped"] += 1
                        advance = True
                    else:
                        new_carry = self._new_carry(
                            cand, prev_cycles, fail_count, source
                        )
                        stats["blocks_carried"] += 1
                        logger.debug(
                            "[简单长期记忆] 分段员 LLM 失败挂起: conv=%s, "
                            "start_id=%s, fail_count=%s",
                            conv_key,
                            cand.first_id,
                            fail_count,
                        )
                    break

                if verdict.get("complete") is True:
                    self._emit(
                        result, conv_key, platform_id, user_id, cand, False, source
                    )
                    adv_id, adv_anchor = cand.last_id, cand.last_anchor
                    new_carry = None
                    stats[
                        "conv2_blocks_emitted"
                        if source == "conv2"
                        else "blocks_emitted"
                    ] += 1
                    advance = True
                    break

                # 判定不完整：后面还有消息且未撞上限则滚动并入下一段重判
                has_more = idx < len(segments)
                if (
                    has_more
                    and extensions < self._max_extensions
                    and len(cand.text) + 1 + len(segments[idx].text) <= self._max_chars
                ):
                    cand.absorb(segments[idx])
                    idx += 1
                    extensions += 1
                    continue

                if has_more:
                    # 撞字符/滚动上限：连续 3 个周期未闭环则强制截断产出
                    if prev_cycles >= 2:
                        self._emit(
                            result, conv_key, platform_id, user_id, cand, True, source
                        )
                        adv_id, adv_anchor = cand.last_id, cand.last_anchor
                        new_carry = None
                        stats["blocks_truncated"] += 1
                        advance = True
                    else:
                        new_carry = self._new_carry(cand, prev_cycles + 1, 0, source)
                        stats["blocks_carried"] += 1
                        logger.debug(
                            "[简单长期记忆] 分段员撞上限挂起: conv=%s, start_id=%s, "
                            "cycles=%s, chars=%s",
                            conv_key,
                            cand.first_id,
                            prev_cycles + 1,
                            len(cand.text),
                        )
                    break

                # 无更多消息：块尾已超间隙视为自然结束，否则挂起等待后续
                if now - cand.last_ts > self._gap:
                    self._emit(
                        result, conv_key, platform_id, user_id, cand, False, source
                    )
                    adv_id, adv_anchor = cand.last_id, cand.last_anchor
                    new_carry = None
                    stats[
                        "conv2_blocks_emitted"
                        if source == "conv2"
                        else "blocks_emitted"
                    ] += 1
                    advance = True
                else:
                    new_carry = self._new_carry(cand, prev_cycles + 1, 0, source)
                    stats["blocks_carried"] += 1
                    logger.debug(
                        "[简单长期记忆] 分段员等待闭环挂起: conv=%s, start_id=%s, "
                        "cycles=%s",
                        conv_key,
                        cand.first_id,
                        prev_cycles + 1,
                    )
                break

            if not advance:
                break

        if adv_id is not None:
            if source == "conv2":
                result["cursor_updates"][conv_key] = {
                    "source": "conv2",
                    "anchor": adv_anchor,
                    "idx": adv_id,
                }
            elif adv_id != cursor_id:
                result["cursor_updates"][conv_key] = adv_id
        if new_carry != prev_carry:
            result["carry_updates"][conv_key] = new_carry

    # ─── 内部工具 ───────────────────────────────────────────

    async def _fetch_all(
        self, platform_id: str, user_id: str, after_id: int
    ) -> list[Any]:
        rows: list[Any] = []
        cursor = after_id
        while True:
            page = await self._fetch_rows(
                platform_id, user_id, cursor, SEGMENT_PAGE_SIZE
            )
            if not page:
                break
            rows.extend(page)
            cursor = page[-1].id
            if len(page) < SEGMENT_PAGE_SIZE:
                break
        return rows

    def _pre_split(self, rows: list[Any]) -> list[_Candidate]:
        """零成本预切：相邻消息 created_at 间隙超阈值切成候选块。"""
        segments: list[_Candidate] = []
        current: _Candidate | None = None
        prev_ts: datetime | None = None
        for row in rows:
            ts = self._aware(row.created_at)
            if current is not None and prev_ts is not None and ts - prev_ts > self._gap:
                segments.append(current)
                current = None
            if current is None:
                current = _Candidate(first_id=row.id, last_id=row.id, last_ts=ts)
            # 空文本行不入文本但仍在块内，游标按行 id 推进照常越过
            current.last_id = row.id
            current.last_ts = ts
            line, body_len = self._flatten_row(row)
            if line:
                current.lines.append(line)
                current.body_chars += body_len
            prev_ts = ts
        if current is not None:
            segments.append(current)
        return segments

    def _pre_split_conv2(
        self, items: list[tuple[str, str, str, str]], start: int, last_ts: datetime
    ) -> list[_Candidate]:
        """ConversationV2 无时间戳，从 start 起按消息数等分切段（id 为全列表下标）。"""
        segments: list[_Candidate] = []
        idx = start
        while idx < len(items):
            chunk = items[idx : idx + self._conv2_chunk]
            cand = _Candidate(
                first_id=idx,
                last_id=idx + len(chunk) - 1,
                last_ts=last_ts,
                first_anchor=chunk[0][3],
                last_anchor=chunk[-1][3],
            )
            for _, sender, text, _ in chunk:
                cand.lines.append(f"{sender}: {text}")
                cand.body_chars += len(text)
            segments.append(cand)
            idx += len(chunk)
        return segments

    def _flatten_conv2_history(
        self, history_raw: Any, conv_key: str
    ) -> list[tuple[str, str, str, str]]:
        """ConversationV2 history 扁平化为 [(role, 显示名, 文本, 内容锚点)]。"""
        if not history_raw:
            return []
        try:
            history = (
                json.loads(history_raw) if isinstance(history_raw, str) else history_raw
            )
        except Exception:
            logger.debug(
                "[简单长期记忆] 分段员 ConversationV2 history 解析失败: conv=%s",
                conv_key,
            )
            return []
        if not isinstance(history, list):
            return []
        items: list[tuple[str, str, str, str]] = []
        for entry in history:
            if not isinstance(entry, dict):
                continue
            role = entry.get("role")
            if role not in ("user", "assistant"):
                continue
            content = entry.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "".join(
                    str(part.get("text") or "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            else:
                continue
            text = text.strip()
            if not text:
                continue
            sender = "bot" if role == "assistant" else "user"
            items.append((role, sender, text, self._anchor(role, text)))
        return items

    @staticmethod
    def _anchor(role: str, text: str) -> str:
        """消息内容锚点：sha256(f\"{role}:{文本}\")[:16]，作 conv2 游标/carry 比对。"""
        return hashlib.sha256(f"{role}:{text}".encode()).hexdigest()[:16]

    @staticmethod
    def _new_carry(
        cand: _Candidate, cycles: int, fail_count: int, source: str
    ) -> dict[str, Any]:
        carry: dict[str, Any] = {
            "start_id": cand.first_id,
            "cycles": cycles,
            "fail_count": fail_count,
        }
        if source == "conv2":
            # conv2 跨周期 carry 用首条消息锚点做起点比对（下标会随窗口滑动）
            carry["start_anchor"] = cand.first_anchor
        return carry

    def _flatten_row(self, row: Any) -> tuple[str, int]:
        """消息行扁平化为 `[MM-DD HH:MM] 发送者: 文本`，返回 (行文本, 纯文本字符数)。"""
        content = getattr(row, "content", None)
        if not isinstance(content, dict):
            return "", 0
        # 实际键为 message（见 insert_message_chain），防御性兼容 chain/parts
        parts = (
            content.get("message") or content.get("chain") or content.get("parts") or []
        )
        if not isinstance(parts, list):
            return "", 0
        texts: list[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "plain":
                texts.append(str(part.get("text") or ""))
            elif ptype == "at":
                texts.append("@" + str(part.get("name") or ""))
            # 其余类型忽略
        body = "".join(texts).strip()
        if not body:
            return "", 0
        sender_id = getattr(row, "sender_id", None)
        if sender_id == "bot":
            sender = "bot"
        else:
            sender = str(getattr(row, "sender_name", None) or sender_id or "未知")
        ts = self._aware(row.created_at).strftime("%m-%d %H:%M")
        return f"[{ts}] {sender}: {body}", len(body)

    def _emit(
        self,
        result: dict[str, Any],
        conv_key: str,
        platform_id: str,
        user_id: str,
        cand: _Candidate,
        truncated: bool,
        source: str,
    ) -> None:
        block = ConversationBlock(
            conv_key=conv_key,
            platform_id=platform_id,
            umo=user_id,
            text=cand.text,
            message_count=len(cand.lines),
            char_count=len(cand.text),
            truncated=truncated,
            first_id=cand.first_id,
            last_id=cand.last_id,
            source=source,
            end_anchor=cand.last_anchor if source == "conv2" else "",
        )
        result["blocks"].append(block)
        logger.debug(
            "[简单长期记忆] 分段员产出对话块: conv=%s, ids=%s-%s, msgs=%s, "
            "chars=%s, truncated=%s, source=%s",
            conv_key,
            cand.first_id,
            cand.last_id,
            block.message_count,
            block.char_count,
            truncated,
            source,
        )

    # ─── KV 数据规范化 ──────────────────────────────────────

    @staticmethod
    def _normalize_cursors(raw: Any) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        if not isinstance(raw, dict):
            return out
        for key, value in raw.items():
            if isinstance(value, dict):
                if value.get("source") == "conv2":
                    try:
                        idx = int(value.get("idx") or 0)
                    except (TypeError, ValueError):
                        continue
                    out[key] = {
                        "source": "conv2",
                        "anchor": str(value.get("anchor") or ""),
                        "idx": idx,
                        "ts": str(value.get("ts") or ""),
                    }
                    continue
                try:
                    cid = int(value.get("id") or 0)
                except (TypeError, ValueError):
                    continue
                out[key] = {"id": cid, "ts": str(value.get("ts") or "")}
            elif isinstance(value, int) and not isinstance(value, bool):
                out[key] = {"id": value, "ts": ""}  # 旧格式纯 int
        return out

    @staticmethod
    def _normalize_carries(raw: Any) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        if not isinstance(raw, dict):
            return out
        for key, value in raw.items():
            if not isinstance(value, dict):
                continue
            try:
                carry: dict[str, Any] = {
                    "start_id": int(value.get("start_id") or 0),
                    "cycles": int(value.get("cycles") or 0),
                    "fail_count": int(value.get("fail_count") or 0),
                }
            except (TypeError, ValueError):
                continue
            anchor = value.get("start_anchor")
            if isinstance(anchor, str) and anchor:
                carry["start_anchor"] = anchor
            out[key] = carry
        return out

    @staticmethod
    def _parse_ts(raw: Any) -> datetime | None:
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(str(raw))
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    @staticmethod
    def _aware(dt: datetime) -> datetime:
        """SQLite 读出的 datetime 可能无时区，统一按 UTC 处理。"""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    @staticmethod
    def _naive(dt: datetime) -> datetime:
        """SQL 过滤用 naive UTC（SQLite DateTime 列不存时区）。"""
        if dt.tzinfo is not None:
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
