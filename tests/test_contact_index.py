"""接触记录存储与隔离测试。"""

import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import StaticPool
from test_memory_commands import memory_manager_module, plugin_main  # noqa: F401

contact_index_module = importlib.import_module(
    "astrbot_plugin_simple_long_memory.contact_index"
)
ContactIndex = contact_index_module.ContactIndex
MemoryManager = memory_manager_module.MemoryManager
ContactObservation = contact_index_module.ContactObservation


class _DocumentStorage:
    def __init__(self, engine):
        self.engine = engine

    def get_session(self):
        return AsyncSession(self.engine, expire_on_commit=False)


def _observation(
    *,
    user_id="test_42",
    bot_id="bot-a",
    umo="test:GroupMessage:1",
    group_id="1",
    group_name="1 群",
    sender_name="小明",
    observed_at="2026-01-01T00:00:00+00:00",
):
    return ContactObservation(
        platform_id="test",
        bot_id=bot_id,
        user_id=user_id,
        umo=umo,
        session_type="group" if group_id else "private",
        group_id=group_id,
        group_name=group_name,
        sender_name=sender_name,
        observed_at=observed_at,
    )


class _ContactEvent:
    def __init__(
        self,
        umo="test:GroupMessage:1",
        sender_id="42",
        bot_id="bot-a",
        admin=False,
    ):
        self.unified_msg_origin = umo
        self.sender_id = sender_id
        self.bot_id = bot_id
        self._admin = admin
        self.group_id = "1" if ":GroupMessage:" in umo else ""
        group_name = "1 群" if self.group_id else ""
        self.message_obj = SimpleNamespace(
            group=SimpleNamespace(group_name=group_name) if self.group_id else None
        )

    def is_admin(self):
        return self._admin

    def get_platform_id(self):
        return "test"

    def get_self_id(self):
        return self.bot_id

    def get_sender_id(self):
        return self.sender_id

    def get_sender_name(self):
        return "小明"

    def get_group_id(self):
        return self.group_id


class ContactIndexTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:", poolclass=StaticPool
        )
        storage = _DocumentStorage(self.engine)
        self.index = ContactIndex(SimpleNamespace(document_storage=storage))
        self.assertTrue(await self.index.ensure_table())

    async def asyncTearDown(self):
        await self.engine.dispose()

    async def test_repeated_observation_updates_existing_row(self):
        await self.index.observe(_observation())
        await self.index.observe(
            _observation(
                sender_name="新昵称",
                observed_at="2026-01-02T00:00:00+00:00",
            )
        )

        rows = await self.index.list_user_records(
            platform_id="test", bot_id="bot-a", user_id="test_42"
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["seen_count"], 2)
        self.assertEqual(rows[0]["sender_name"], "新昵称")
        self.assertEqual(rows[0]["first_seen_at"], "2026-01-01T00:00:00+00:00")
        self.assertEqual(rows[0]["last_seen_at"], "2026-01-02T00:00:00+00:00")

    async def test_only_observed_sessions_are_returned(self):
        await self.index.observe(_observation())
        await self.index.observe(
            _observation(
                umo="test:FriendMessage:42",
                group_id="",
                group_name="",
                observed_at="2026-01-02T00:00:00+00:00",
            )
        )

        rows = await self.index.list_user_records(
            platform_id="test", bot_id="bot-a", user_id="test_42"
        )
        group_two = await self.index.list_group_records(
            platform_id="test", bot_id="bot-a", umo="test:GroupMessage:2"
        )

        self.assertEqual(
            {row["umo"] for row in rows},
            {"test:GroupMessage:1", "test:FriendMessage:42"},
        )
        self.assertEqual(group_two, [])

    async def test_bot_filter_is_applied_only_when_sharing_is_disabled(self):
        await self.index.observe(_observation())
        await self.index.observe(
            _observation(
                bot_id="bot-b",
                umo="test:GroupMessage:2",
                group_id="2",
                group_name="2 群",
            )
        )

        isolated = await self.index.list_user_records(
            platform_id="test", bot_id="bot-a", user_id="test_42"
        )
        shared = await self.index.list_user_records(
            platform_id="test",
            bot_id="bot-a",
            user_id="test_42",
            include_other_bots=True,
        )

        self.assertEqual([row["bot_id"] for row in isolated], ["bot-a"])
        self.assertEqual({row["bot_id"] for row in shared}, {"bot-a", "bot-b"})

    async def test_same_nickname_does_not_merge_users(self):
        await self.index.observe(_observation(user_id="test_42"))
        await self.index.observe(_observation(user_id="test_99"))

        rows = await self.index.list_group_records(
            platform_id="test", bot_id="bot-a", umo="test:GroupMessage:1"
        )

        self.assertEqual({row["user_id"] for row in rows}, {"test_42", "test_99"})

    async def test_session_retention_is_bounded(self):
        for number in range(21):
            await self.index.observe(
                _observation(
                    umo=f"test:GroupMessage:{number}",
                    group_id=str(number),
                    group_name=f"{number} 群",
                    observed_at=f"2026-01-{number + 1:02d}T00:00:00+00:00",
                )
            )

        rows = await self.index.list_user_records(
            platform_id="test", bot_id="bot-a", user_id="test_42", limit=50
        )

        self.assertEqual(len(rows), 20)
        self.assertNotIn("test:GroupMessage:0", {row["umo"] for row in rows})
        self.assertIn("test:GroupMessage:20", {row["umo"] for row in rows})

    async def test_clear_user_and_clear_all(self):
        await self.index.observe(_observation())
        await self.index.observe(_observation(bot_id="bot-b"))
        await self.index.observe(_observation(user_id="test_99"))

        removed = await self.index.clear_user("test_42")
        remaining = await self.index.list_group_records(
            platform_id="test",
            bot_id="bot-a",
            umo="test:GroupMessage:1",
            include_other_bots=True,
        )

        self.assertEqual(removed, 2)
        self.assertEqual([row["user_id"] for row in remaining], ["test_99"])
        self.assertEqual(await self.index.clear_all(), 1)

    async def test_manager_summary_respects_global_memory_switch(self):
        manager = MemoryManager(kb_mgr=None, config={"global_memory": True})
        manager._contact_index = self.index
        event = _ContactEvent()

        self.assertTrue(await manager.observe_contact(event))
        summary = await manager.get_contact_summary(event)
        manager.config["global_memory"] = False

        self.assertIn("1 群", summary)
        self.assertIn("不代表完整群成员关系", summary)
        self.assertEqual(await manager.get_contact_summary(event), "")

    async def test_only_full_memory_clear_cascades_to_contacts(self):
        manager = MemoryManager(kb_mgr=None, config={})
        manager._contact_index = self.index
        manager._clear_by_filters = AsyncMock(return_value=1)
        event = _ContactEvent()
        await manager.observe_contact(event)

        await manager.clear_memories(event, domain="preferences")
        retained = await self.index.list_user_records(
            platform_id="test", bot_id="bot-a", user_id="test_42"
        )
        await manager.clear_memories(event)
        cleared = await self.index.list_user_records(
            platform_id="test", bot_id="bot-a", user_id="test_42"
        )

        self.assertEqual(len(retained), 1)
        self.assertEqual(cleared, [])

    async def test_admin_target_clear_uses_canonical_user_id(self):
        manager = MemoryManager(kb_mgr=None, config={})
        manager._contact_index = self.index
        manager._clear_by_filters = AsyncMock(return_value=1)
        await manager.observe_contact(_ContactEvent())

        await manager.clear_memories_by_user(_ContactEvent(), "test_42")

        self.assertEqual(
            await self.index.list_user_records(
                platform_id="test", bot_id="bot-a", user_id="test_42"
            ),
            [],
        )

    async def test_contact_tool_self_and_current_group(self):
        manager = MemoryManager(kb_mgr=None, config={"global_memory": True})
        manager._contact_index = self.index
        plugin = SimpleNamespace(
            config={},
            memory_mgr=manager,
            _format_contact_records=lambda records, scope: (
                plugin_main.MemoryPlugin._format_contact_records(plugin, records, scope)
            ),
        )
        event = _ContactEvent()
        await manager.observe_contact(event)
        await manager.observe_contact(_ContactEvent(sender_id="99"))

        own = await plugin_main.MemoryPlugin.tool_contacts(plugin, event)
        group = await plugin_main.MemoryPlugin.tool_contacts(
            plugin, event, scope="current_group"
        )

        self.assertIn("1 群", own)
        self.assertIn("不是完整平台通讯录", own)
        self.assertIn("test_42", group)
        self.assertIn("test_99", group)

    async def test_contact_tool_self_respects_current_session_when_global_off(self):
        manager = MemoryManager(kb_mgr=None, config={"global_memory": False})
        manager._contact_index = self.index
        plugin = SimpleNamespace(
            config={},
            memory_mgr=manager,
            _format_contact_records=lambda records, scope: (
                plugin_main.MemoryPlugin._format_contact_records(plugin, records, scope)
            ),
        )
        event = _ContactEvent()
        await manager.observe_contact(event)
        await manager.observe_contact(_ContactEvent(umo="test:FriendMessage:42"))

        result = await plugin_main.MemoryPlugin.tool_contacts(plugin, event)

        self.assertIn("1 群", result)
        self.assertNotIn("私聊", result)

    async def test_contact_tool_validates_arguments_and_admin_permission(self):
        manager = MemoryManager(kb_mgr=None, config={})
        manager._contact_index = self.index
        plugin = SimpleNamespace(
            config={},
            memory_mgr=manager,
            _format_contact_records=lambda records, scope: (
                plugin_main.MemoryPlugin._format_contact_records(plugin, records, scope)
            ),
        )
        event = _ContactEvent()
        await manager.observe_contact(event)

        invalid_scope = await plugin_main.MemoryPlugin.tool_contacts(
            plugin, event, scope="everyone"
        )
        invalid_limit = await plugin_main.MemoryPlugin.tool_contacts(
            plugin, event, limit=0
        )
        disabled_admin = await plugin_main.MemoryPlugin.tool_contacts(
            plugin, event, scope="user", user_id="test_42"
        )
        plugin.config["enable_admin_global_memory_tool"] = True
        non_admin = await plugin_main.MemoryPlugin.tool_contacts(
            plugin, event, scope="user", user_id="test_42"
        )
        admin = await plugin_main.MemoryPlugin.tool_contacts(
            plugin,
            _ContactEvent(admin=True),
            scope="user",
            user_id="test_42",
        )

        self.assertIn("Valid scopes: current_group, self, user", invalid_scope)
        self.assertIn("between 1 and 20", invalid_limit)
        self.assertIn("disabled", disabled_admin)
        self.assertIn("Only administrators", non_admin)
        self.assertIn("1 群", admin)

    async def test_contact_tool_rejects_current_group_in_private_chat(self):
        manager = MemoryManager(kb_mgr=None, config={})
        manager._contact_index = self.index
        plugin = SimpleNamespace(
            config={},
            memory_mgr=manager,
            _format_contact_records=lambda records, scope: (
                plugin_main.MemoryPlugin._format_contact_records(plugin, records, scope)
            ),
        )

        result = await plugin_main.MemoryPlugin.tool_contacts(
            plugin,
            _ContactEvent(umo="test:FriendMessage:42"),
            scope="current_group",
        )

        self.assertIn("requires a group conversation", result)

    async def test_contact_capture_is_throttled_per_bot_and_session(self):
        manager = SimpleNamespace(observe_contact=AsyncMock(return_value=True))
        plugin = SimpleNamespace(
            memory_mgr=manager,
            _contact_write_times={},
            CONTACT_WRITE_INTERVAL=60,
            CONTACT_WRITE_CACHE_MAX=10000,
        )
        event = _ContactEvent()

        await plugin_main.MemoryPlugin._maybe_observe_contact(plugin, event)
        await plugin_main.MemoryPlugin._maybe_observe_contact(plugin, event)
        await plugin_main.MemoryPlugin._maybe_observe_contact(
            plugin, _ContactEvent(bot_id="bot-b")
        )

        self.assertEqual(manager.observe_contact.await_count, 2)

    async def test_storage_failures_degrade_to_empty_results(self):
        class BrokenStorage:
            def get_session(self):
                raise RuntimeError("database unavailable")

        broken = ContactIndex(SimpleNamespace(document_storage=BrokenStorage()))
        self.assertFalse(await broken.ensure_table())
        self.assertFalse(await broken.observe(_observation()))
        self.assertEqual(
            await broken.list_user_records(
                platform_id="test", bot_id="bot-a", user_id="test_42"
            ),
            [],
        )
        self.assertEqual(await broken.clear_all(), 0)


if __name__ == "__main__":
    unittest.main()
