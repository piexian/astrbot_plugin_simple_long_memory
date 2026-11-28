"""AstrBot 4.29.4 compatibility smoke tests."""

from __future__ import annotations

import importlib
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, PropertyMock, patch

from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata

from maintenance.version_gate import parse_release_tuple


def _real_message_event(umo: str = "qq:FriendMessage:u1") -> AstrMessageEvent:
    """构造真实 AstrMessageEvent；4.29 的 PlatformMetadata 没有 UI 交互字段。"""
    platform_id, kind, session_id = umo.split(":", 2)
    message = AstrBotMessage()
    message.type = (
        MessageType.GROUP_MESSAGE
        if kind == "GroupMessage"
        else MessageType.FRIEND_MESSAGE
    )
    message.session_id = session_id
    message.sender = MessageMember(user_id=session_id, nickname="测试用户")
    message.self_id = "bot"
    return AstrMessageEvent(
        message_str="今天聊点什么",
        message_obj=message,
        platform_meta=PlatformMetadata(
            name=platform_id,
            description="test platform",
            id=platform_id,
        ),
        session_id=session_id,
    )


class AstrBot429ImportTests(TestCase):
    def test_plugin_imports_without_decision_choice_exports(self):
        provider = importlib.import_module("astrbot.api.provider")
        if hasattr(provider, "DecisionChoiceAnswer"):
            self.skipTest("Decision Choice types are available in this AstrBot version")

        importlib.import_module("astrbot_plugin_simple_long_memory.main")
        analyst = importlib.import_module(
            "astrbot_plugin_simple_long_memory.maintenance.agents.analyst"
        )

        self.assertIsNone(analyst.DecisionChoiceAnswer)
        self.assertIsNone(analyst.DecisionChoiceQuestion)


class LegacyMaintenanceFallbackTests(IsolatedAsyncioTestCase):
    async def test_429_keeps_legacy_consolidation_available(self):
        main_module = importlib.import_module("astrbot_plugin_simple_long_memory.main")
        memory_manager_module = importlib.import_module(
            "astrbot_plugin_simple_long_memory.memory_manager"
        )
        astrbot = importlib.import_module("astrbot")
        version = str(getattr(astrbot, "__version__", ""))
        if parse_release_tuple(version) != (4, 29, 4):
            self.skipTest(f"requires AstrBot 4.29.4, got {version}")

        plugin = main_module.MemoryPlugin(
            None,
            {"maintenance_enabled": True, "memory_consolidation_enabled": True},
        )
        manager = memory_manager_module.MemoryManager(None, {})
        plugin.memory_mgr = manager
        fetch_candidates = AsyncMock(return_value=[])
        event = _real_message_event()

        self.assertFalse(main_module.maintenance_supported())
        with (
            patch.object(manager, "fetch_consolidation_candidates", fetch_candidates),
            patch.object(
                type(manager),
                "is_kb_connected",
                new_callable=PropertyMock,
                return_value=True,
            ),
        ):
            await plugin._maybe_consolidate_memories(event)

        fetch_candidates.assert_awaited_once_with(event, 14, 1, 30)
