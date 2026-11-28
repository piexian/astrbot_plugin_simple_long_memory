"""接触记录真实栈测试：真实 KB SQLite、真实消息事件与真实工具入口。"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from astrbot_plugin_simple_long_memory.contact_index import (
    ContactIndex,
    ContactObservation,
)
from real_stack import MemoryPlugin, RealStackTestCase, real_message_event
from sqlalchemy import text as sa_text

PLATFORM = "webchat"
USER_ID = "webchat_42"
GROUP_UMO = "webchat:GroupMessage:g1"
PRIVATE_UMO = "webchat:FriendMessage:42"


def _event(
    *,
    umo: str = GROUP_UMO,
    sender_id: str = "42",
    sender_name: str = "小明",
    self_id: str = "bot-a",
    group_name: str = "一号群",
    admin: bool = False,
):
    """构造真实消息事件；群名写入真实 Group 对象。"""
    event = real_message_event(
        umo=umo,
        sender_id=sender_id,
        sender_name=sender_name,
        self_id=self_id,
        admin=admin,
    )
    group = event.message_obj.group
    if group is not None and group_name:
        group.group_name = group_name
    return event


class _PausingStorage:
    """转发到真实文档存储，并在首个事务语句后暂停以制造观测与清空交错。"""

    def __init__(self, inner):
        self.inner = inner
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.paused = False

    def get_session(self):
        return _PausingSession(self)


class _PausingSession:
    """真实会话代理：首个 execute 之后等待放行。"""

    def __init__(self, pauser):
        self.pauser = pauser
        self.session_cm = pauser.inner.get_session()
        self.session = None

    async def __aenter__(self):
        self.session = await self.session_cm.__aenter__()
        return self

    async def __aexit__(self, *args):
        return await self.session_cm.__aexit__(*args)

    def begin(self):
        return self.session.begin()

    async def execute(self, *args):
        result = await self.session.execute(*args)
        if not self.pauser.paused:
            self.pauser.paused = True
            self.pauser.entered.set()
            await self.pauser.release.wait()
        return result


class ContactIndexTests(RealStackTestCase):
    async def asyncSetUp(self):
        await self.start_stack(with_decision=False, with_plugin=True)
        self.assertIsInstance(self.plugin, MemoryPlugin)
        self.index = self.mgr._contact_index
        self.assertIsNotNone(self.index)

    async def _user_rows(self, user_id: str = USER_ID, **kwargs):
        return await self.index.list_user_records(
            platform_id=PLATFORM, bot_id="bot-a", user_id=user_id, **kwargs
        )

    async def _group_rows(self, umo: str = GROUP_UMO, **kwargs):
        return await self.index.list_group_records(
            platform_id=PLATFORM, bot_id="bot-a", umo=umo, **kwargs
        )

    async def test_repeated_observation_updates_existing_row(self):
        self.assertTrue(await self.mgr.observe_contact(_event()))
        first = (await self._user_rows())[0]
        self.assertTrue(await self.mgr.observe_contact(_event(sender_name="新昵称")))

        rows = await self._user_rows()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["seen_count"], 2)
        self.assertEqual(rows[0]["sender_name"], "新昵称")
        self.assertEqual(rows[0]["first_seen_at"], first["first_seen_at"])
        self.assertGreaterEqual(rows[0]["last_seen_at"], first["last_seen_at"])

    async def test_only_observed_sessions_are_returned(self):
        await self.mgr.observe_contact(_event())
        await self.mgr.observe_contact(_event(umo=PRIVATE_UMO, group_name=""))

        rows = await self._user_rows()

        self.assertEqual({row["umo"] for row in rows}, {GROUP_UMO, PRIVATE_UMO})
        self.assertEqual(await self._group_rows("webchat:GroupMessage:g2"), [])

    async def test_bot_scope_follows_sharing_switch(self):
        await self.mgr.observe_contact(_event())
        await self.mgr.observe_contact(
            _event(self_id="bot-b", umo="webchat:GroupMessage:g2", group_name="二号群")
        )
        event = _event()

        isolated = await self._user_rows()
        shared = await self._user_rows(include_other_bots=True)
        manager_isolated = await self.mgr.list_contact_records(event, scope="self")
        self.mgr.config["share_contacts_across_bots"] = True
        manager_shared = await self.mgr.list_contact_records(event, scope="self")

        self.assertEqual([row["bot_id"] for row in isolated], ["bot-a"])
        self.assertEqual({row["bot_id"] for row in shared}, {"bot-a", "bot-b"})
        self.assertEqual([row["bot_id"] for row in manager_isolated], ["bot-a"])
        self.assertEqual({row["bot_id"] for row in manager_shared}, {"bot-a", "bot-b"})

    async def test_same_nickname_does_not_merge_users(self):
        await self.mgr.observe_contact(_event(sender_id="42"))
        await self.mgr.observe_contact(_event(sender_id="99"))

        rows = await self._group_rows()

        self.assertEqual({row["user_id"] for row in rows}, {"webchat_42", "webchat_99"})

    async def test_session_retention_is_bounded(self):
        for number in range(21):
            await self.mgr.observe_contact(
                _event(
                    umo=f"webchat:GroupMessage:g{number}",
                    group_name=f"{number} 群",
                )
            )
            # 保留按最近接触时间排序，逐条写入需保证时间戳可区分
            await asyncio.sleep(0.002)

        rows = await self._user_rows(limit=50)

        self.assertEqual(len(rows), 20)
        self.assertNotIn("webchat:GroupMessage:g0", {row["umo"] for row in rows})
        self.assertIn("webchat:GroupMessage:g20", {row["umo"] for row in rows})

    async def test_clear_user_and_clear_all(self):
        await self.mgr.observe_contact(_event())
        await self.mgr.observe_contact(_event(self_id="bot-b"))
        await self.mgr.observe_contact(_event(sender_id="99"))

        self.assertEqual(await self.index.clear_user(USER_ID), 2)
        remaining = await self._group_rows(include_other_bots=True)
        self.assertEqual([row["user_id"] for row in remaining], ["webchat_99"])
        self.assertEqual(await self.index.clear_all(), 1)

    async def test_contact_summary_respects_global_memory_switch(self):
        await self.mgr.observe_contact(_event())
        event = _event()

        summary = await self.mgr.get_contact_summary(event)
        self.mgr.config["global_memory"] = False

        self.assertIn("一号群", summary)
        self.assertIn("不代表完整群成员关系", summary)
        self.assertEqual(await self.mgr.get_contact_summary(event), "")

    async def test_only_full_memory_clear_cascades_to_contacts(self):
        await self.mgr.observe_contact(_event())
        event = _event()

        partial = await self.mgr.clear_memories(event, domain="preferences")
        retained = await self._user_rows()
        cleared = await self.mgr.clear_memories(event)
        remaining = await self._user_rows()

        self.assertFalse(partial.contact_cleanup_failed)
        self.assertEqual(len(retained), 1)
        self.assertFalse(cleared.contact_cleanup_failed)
        self.assertEqual(remaining, [])

    async def test_admin_target_clear_uses_canonical_user_id(self):
        await self.mgr.observe_contact(_event())

        result = await self.mgr.clear_memories_by_user(_event(), USER_ID)

        self.assertFalse(result.contact_cleanup_failed)
        self.assertEqual(await self._user_rows(), [])

    async def test_contact_tool_self_and_current_group(self):
        await self.mgr.observe_contact(_event())
        await self.mgr.observe_contact(_event(sender_id="99", sender_name="小红"))
        event = _event()

        own = await self.plugin.tool_contacts(event)
        group = await self.plugin.tool_contacts(event, scope="current_group")

        self.assertIn("一号群", own)
        self.assertIn("不是完整平台通讯录", own)
        self.assertIn(USER_ID, group)
        self.assertIn("webchat_99", group)

    async def test_contact_tool_self_keeps_current_session_when_global_off(self):
        self.mgr.config["global_memory"] = False
        await self.mgr.observe_contact(_event())
        await self.mgr.observe_contact(_event(umo=PRIVATE_UMO, group_name=""))

        result = await self.plugin.tool_contacts(_event())

        self.assertIn("一号群", result)
        self.assertNotIn("私聊", result)

    async def test_contact_tool_validates_arguments_and_admin_permission(self):
        await self.mgr.observe_contact(_event())
        event = _event()

        invalid_scope = await self.plugin.tool_contacts(event, scope="everyone")
        invalid_limit = await self.plugin.tool_contacts(event, limit=0)
        disabled_admin = await self.plugin.tool_contacts(
            event, scope="user", user_id=USER_ID
        )
        self.plugin.config["enable_admin_global_memory_tool"] = True
        non_admin = await self.plugin.tool_contacts(
            event, scope="user", user_id=USER_ID
        )
        admin = await self.plugin.tool_contacts(
            _event(admin=True), scope="user", user_id=USER_ID
        )

        self.assertIn("Valid scopes: current_group, self, user", invalid_scope)
        self.assertIn("between 1 and 20", invalid_limit)
        self.assertIn("disabled", disabled_admin)
        self.assertIn("Only administrators", non_admin)
        self.assertIn("一号群", admin)

    async def test_contact_tool_rejects_current_group_in_private_chat(self):
        result = await self.plugin.tool_contacts(
            _event(umo=PRIVATE_UMO, group_name=""), scope="current_group"
        )

        self.assertIn("requires a group conversation", result)

    async def test_contact_capture_is_throttled_per_bot_and_session(self):
        await self.plugin._maybe_observe_contact(_event())
        await self.plugin._maybe_observe_contact(_event())
        await self.plugin._maybe_observe_contact(_event(self_id="bot-b"))

        rows = await self._user_rows(include_other_bots=True)

        self.assertEqual(len(rows), 2)
        self.assertEqual([row["seen_count"] for row in rows], [1, 1])

    async def test_clear_reports_contact_cleanup_failure(self):
        await self.mgr.observe_contact(_event())
        storage = self.mgr.vec_db.document_storage
        async with storage.get_session() as session, session.begin():
            await session.execute(sa_text("DROP TABLE contact_observations"))

        result = await self.mgr.clear_memories(_event())

        self.assertTrue(result.contact_cleanup_failed)

    async def test_clear_waits_for_in_flight_observation(self):
        storage = self.mgr.vec_db.document_storage
        pausing = _PausingStorage(storage)
        self.mgr.vec_db.document_storage = pausing

        observe_task = asyncio.create_task(self.mgr.observe_contact(_event()))
        await asyncio.wait_for(pausing.entered.wait(), timeout=5)
        clear_task = asyncio.create_task(self.index.clear_user(USER_ID))
        await asyncio.sleep(0)
        self.assertFalse(clear_task.done())

        pausing.release.set()
        self.assertTrue(await asyncio.wait_for(observe_task, timeout=5))
        self.assertEqual(await asyncio.wait_for(clear_task, timeout=5), 1)
        self.mgr.vec_db.document_storage = storage

        self.assertEqual(await self._user_rows(), [])

    async def test_storage_failures_degrade_to_empty_results(self):
        broken = ContactIndex(self.mgr.vec_db)
        storage = self.mgr.vec_db.document_storage
        observation = ContactObservation(
            platform_id=PLATFORM,
            bot_id="bot-a",
            user_id=USER_ID,
            umo=GROUP_UMO,
            session_type="group",
        )

        with patch.object(
            storage, "get_session", side_effect=RuntimeError("database unavailable")
        ):
            self.assertFalse(await broken.ensure_table())
            self.assertFalse(await broken.observe(observation))
            self.assertEqual(
                await broken.list_user_records(
                    platform_id=PLATFORM, bot_id="bot-a", user_id=USER_ID
                ),
                [],
            )
            self.assertIsNone(await broken.clear_all())


if __name__ == "__main__":
    unittest.main()
