"""Inbox 纯规则聚合与终态清理测试：真实 AstrBot DocumentStorage。

不模拟任何 AstrBot 接口：缓冲表建在真实 SQLite 文档库上，
按时间窗口、字符上限与块数上限等纯规则切块，不调用任何 LLM。
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from test_maintenance_extract import _real_inbox
from test_memory_commands import MemoryManager

from maintenance.inbox import (
    INBOX_MAX_RETRIES,
    InboxRecord,
    MaintenanceInbox,
)


def _now() -> datetime:
    return datetime(2026, 11, 27, 12, 0, tzinfo=timezone.utc)


class InboxAggregationTests(unittest.IsolatedAsyncioTestCase):
    async def _inbox(self) -> MaintenanceInbox:
        inbox, _storage = await _real_inbox(self)
        self.assertTrue(await inbox.ensure_table())
        return inbox

    async def _append(
        self,
        inbox,
        *,
        minutes_ago,
        text="聊" * 200,
        cid="c1",
        umo="qq:private:u1",
        sender_id="u1",
        sender_name="小明",
        brief="助手回复",
        realtime=False,
    ):
        record = InboxRecord(
            umo=umo,
            platform_id=umo.split(":", 1)[0],
            session_type="private",
            session_id=umo.rsplit(":", 1)[-1],
            conversation_id=cid,
            sender_id=sender_id,
            sender_name=sender_name,
            user_text=text,
            assistant_brief=brief,
            created_at=(_now() - timedelta(minutes=minutes_ago)).isoformat(),
        )
        row_id = await inbox.append(record)
        self.assertIsNotNone(row_id)
        if realtime:
            self.assertTrue(await inbox.mark_realtime_extracted(row_id))
        return row_id

    async def _collect(self, inbox, **overrides):
        params = {
            "gap_minutes": 30,
            "max_chars": 8000,
            "min_user_chars": 50,
            "max_blocks": 20,
            "commit": True,
            "now": _now(),
        }
        params.update(overrides)
        return await inbox.collect_blocks(**params)

    async def test_time_gap_and_char_cap_split_blocks(self):
        inbox = await self._inbox()
        await self._append(inbox, minutes_ago=90)
        await self._append(inbox, minutes_ago=40)  # 与上行间隔 50 分钟，切新块
        await self._append(inbox, minutes_ago=35)  # 与上行同窗，并入
        result = await self._collect(inbox)
        blocks = result["blocks"]
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].row_ids, [1])
        self.assertEqual(blocks[1].row_ids, [2, 3])
        self.assertEqual(blocks[1].conv_key, "qq:private:u1:c1")
        self.assertEqual(result["stats"]["blocks_emitted"], 2)

        # 字符上限独立于时间窗：连续轮次超上限同样切块
        inbox2 = await self._inbox()
        for _ in range(3):
            await self._append(inbox2, minutes_ago=60, text="聊" * 300)
        result2 = await self._collect(inbox2, max_chars=500)
        self.assertEqual(len(result2["blocks"]), 3)
        self.assertTrue(all(b.char_count <= 500 for b in result2["blocks"]))

    async def test_quiet_window_holds_recent_tail(self):
        inbox = await self._inbox()
        await self._append(inbox, minutes_ago=90)
        await self._append(inbox, minutes_ago=5)  # 距当前不足 10 分钟静默阈值
        result = await self._collect(inbox)
        self.assertEqual(len(result["blocks"]), 1)
        self.assertEqual(result["stats"]["blocks_held_quiet"], 1)
        # 被保留的尾块与已产出块都保持 pending，前者等待下周期合并
        counts = await inbox.status_counts()
        self.assertEqual(counts.get("pending"), 2)

        # 尾块已达字符上限时不适用静默保留
        inbox2 = await self._inbox()
        await self._append(inbox2, minutes_ago=5, text="聊" * 600)
        result2 = await self._collect(inbox2, max_chars=500)
        self.assertEqual(len(result2["blocks"]), 1)
        self.assertEqual(result2["stats"]["blocks_held_quiet"], 0)

    async def test_block_cap_leaves_remaining_rows_pending(self):
        inbox = await self._inbox()
        for n in range(3):
            await self._append(inbox, minutes_ago=60, umo=f"qq:private:u{n}")
        result = await self._collect(inbox, max_blocks=2)
        self.assertEqual(len(result["blocks"]), 2)
        self.assertTrue(result["stats"]["block_cap_reached"])
        counts = await inbox.status_counts()
        # collect 只标 skipped 终态；已产出与未轮到的行都保持 pending，
        # 由 Runner 在操作完成后标记 processed
        self.assertEqual(counts.get("pending"), 3)

    async def test_invalid_umo_rows_are_skipped(self):
        inbox = await self._inbox()
        await self._append(inbox, minutes_ago=60, umo="not-a-valid-umo")
        await self._append(inbox, minutes_ago=60, umo="qq:private:u1")
        result = await self._collect(inbox)
        self.assertEqual(result["stats"]["invalid_umo_skipped"], 1)
        self.assertEqual(len(result["blocks"]), 1)
        counts = await inbox.status_counts()
        self.assertEqual(counts.get("skipped"), 1)
        self.assertEqual(counts.get("pending"), 1)

    async def test_short_block_is_skipped_only_in_commit_mode(self):
        inbox = await self._inbox()
        await self._append(inbox, minutes_ago=60, text="嗯嗯")
        result = await self._collect(inbox)
        self.assertEqual(result["stats"]["blocks_skipped_short"], 1)
        self.assertEqual(result["blocks"], [])
        counts = await inbox.status_counts()
        self.assertEqual(counts.get("skipped"), 1)

        inbox2 = await self._inbox()
        await self._append(inbox2, minutes_ago=60, text="嗯嗯")
        dry = await self._collect(inbox2, commit=False)
        self.assertEqual(dry["stats"]["blocks_skipped_short"], 1)
        counts2 = await inbox2.status_counts()
        self.assertEqual(counts2.get("pending"), 1)  # dry-run 不写行状态

    async def test_block_text_keeps_sender_labels_and_realtime_flag(self):
        inbox = await self._inbox()
        text = "用户喜欢美式咖啡，每天早上都要喝两杯，周末还会去店里手冲单品豆，最近换了深烘的肯尼亚豆，还说以后想学着自己在家里烘豆子。"
        await self._append(
            inbox,
            minutes_ago=60,
            text=text,
            sender_id="10086",
            sender_name="小明",
            brief="好的，记住了",
            realtime=True,
        )
        result = await self._collect(inbox)
        block = result["blocks"][0]
        stamp = (_now() - timedelta(minutes=60)).strftime("%m-%d %H:%M")
        self.assertIn(f"[{stamp}] 用户(sender_id=10086, 小明): {text}", block.text)
        self.assertIn(f"[{stamp}] 助手: 好的，记住了", block.text)
        self.assertEqual(block.sender_ids, ["10086"])
        self.assertTrue(block.realtime_extracted)

    async def test_groups_are_separated_by_umo_and_conversation(self):
        inbox = await self._inbox()
        # 交错插入两个会话，SQL 排序须保证同组行连续
        await self._append(inbox, minutes_ago=60, umo="qq:private:u1", cid="c1")
        await self._append(inbox, minutes_ago=60, umo="qq:private:u2", cid="c9")
        await self._append(inbox, minutes_ago=60, umo="qq:private:u1", cid="c1")
        result = await self._collect(inbox)
        keys = [b.conv_key for b in result["blocks"]]
        self.assertEqual(keys, ["qq:private:u1:c1", "qq:private:u2:c9"])
        self.assertEqual(result["blocks"][0].message_count, 2)

    async def test_failed_rows_retry_until_cap_then_stay_out(self):
        inbox = await self._inbox()
        row_id = await self._append(inbox, minutes_ago=60)
        await inbox.mark_failed([row_id])
        await inbox.mark_failed([row_id])
        # 重试额度内：failed 行继续参与提取
        fetched = await self._collect(inbox)
        self.assertEqual(fetched["stats"]["rows_fetched"], 1)
        counts = await inbox.status_counts()
        self.assertEqual(counts.get("failed"), 1)

        await inbox.mark_failed([row_id])  # 第 3 次失败，重试额度用尽
        exhausted = await self._collect(inbox)
        self.assertEqual(exhausted["stats"]["rows_fetched"], 0)
        self.assertEqual(exhausted["stats"]["blocks_emitted"], 0)


class InboxPurgeTests(unittest.IsolatedAsyncioTestCase):
    async def _inbox(self) -> MaintenanceInbox:
        inbox, _storage = await _real_inbox(self)
        self.assertTrue(await inbox.ensure_table())
        return inbox

    async def _seed(self, inbox, *, days_ago, status=None, retries=0, text="聊" * 200):
        created_at = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
        record = InboxRecord(
            umo="qq:private:u1",
            platform_id="qq",
            session_type="private",
            session_id="u1",
            conversation_id="c1",
            sender_id="u1",
            sender_name="小明",
            user_text=text,
            assistant_brief="",
            created_at=created_at,
        )
        row_id = await inbox.append(record)
        self.assertIsNotNone(row_id)
        for _ in range(retries):
            self.assertTrue(await inbox.mark_failed([row_id]))
        if status == "processed":
            self.assertTrue(await inbox.mark_rows([row_id], "processed"))
        elif status == "skipped":
            self.assertTrue(await inbox.mark_rows([row_id], "skipped"))
        return row_id

    async def test_purge_removes_only_terminal_rows_past_retention(self):
        inbox = await self._inbox()
        old_processed = await self._seed(inbox, days_ago=15, status="processed")
        old_skipped = await self._seed(inbox, days_ago=15, status="skipped")
        old_pending = await self._seed(inbox, days_ago=15)
        old_failed = await self._seed(inbox, days_ago=15, retries=INBOX_MAX_RETRIES)
        retrying_failed = await self._seed(inbox, days_ago=15, retries=1)
        fresh_processed = await self._seed(inbox, days_ago=1, status="processed")

        result = await inbox.purge_terminal()
        self.assertEqual(result["candidates"], 3)
        self.assertEqual(result["purged"], 3)
        counts = await inbox.status_counts()
        # pending 与重试中的 failed 超期不删；新近终态保留
        self.assertEqual(counts.get("pending"), 1)
        self.assertEqual(counts.get("failed"), 1)
        self.assertEqual(counts.get("processed"), 1)
        kept = {old_pending, retrying_failed, fresh_processed}
        self.assertNotIn(old_processed, kept)
        self.assertNotIn(old_skipped, kept)
        self.assertNotIn(old_failed, kept)

    async def test_purge_dry_run_counts_without_deleting(self):
        inbox = await self._inbox()
        await self._seed(inbox, days_ago=15, status="processed")
        result = await inbox.purge_terminal(dry_run=True)
        self.assertEqual(result["candidates"], 1)
        self.assertEqual(result["purged"], 0)
        counts = await inbox.status_counts()
        self.assertEqual(counts.get("processed"), 1)


class MemoryManagerPurgeInboxTests(unittest.IsolatedAsyncioTestCase):
    async def test_purge_deprecated_cleans_terminal_inbox_rows(self):
        inbox, storage = await _real_inbox(self)
        await inbox.ensure_table()
        created_at = (datetime.now(timezone.utc) - timedelta(days=15)).isoformat()

        async def _seed(status):
            record = InboxRecord(
                umo="qq:private:u1",
                platform_id="qq",
                session_type="private",
                session_id="u1",
                conversation_id="c1",
                sender_id="u1",
                sender_name="小明",
                user_text="聊" * 200,
                assistant_brief="",
                created_at=created_at,
            )
            row_id = await inbox.append(record)
            if status:
                self.assertTrue(await inbox.mark_rows([row_id], status))
            return row_id

        await _seed("processed")
        await _seed(None)  # pending 行超期不删

        manager = MemoryManager(None, {})
        manager._kb_helper = SimpleNamespace(
            vec_db=SimpleNamespace(document_storage=storage),
            kb=SimpleNamespace(kb_id="kb1"),
        )
        manager.inbox = inbox
        result = await manager.purge_deprecated(after_days=7)

        self.assertEqual(result["inbox"]["purged"], 1)
        counts = await inbox.status_counts()
        self.assertEqual(counts.get("pending"), 1)
        self.assertNotIn("processed", counts)


if __name__ == "__main__":
    unittest.main()
