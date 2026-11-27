"""AstrBot 4.29.4 compatibility smoke tests."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock

from maintenance.version_gate import parse_release_tuple


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
        plugin = main_module.MemoryPlugin.__new__(main_module.MemoryPlugin)
        fetch_candidates = AsyncMock(return_value=[])
        plugin.config = {
            "maintenance_enabled": True,
            "memory_consolidation_enabled": True,
        }
        plugin.memory_mgr = SimpleNamespace(
            is_kb_connected=True,
            fetch_consolidation_candidates=fetch_candidates,
        )
        plugin._last_consolidation = 0.0
        plugin._get_llm_provider_id = AsyncMock(return_value=None)
        event = SimpleNamespace(unified_msg_origin="qq:FriendMessage:u1")
        astrbot = importlib.import_module("astrbot")
        version = str(getattr(astrbot, "__version__", ""))
        if parse_release_tuple(version) != (4, 29, 4):
            self.skipTest(f"requires AstrBot 4.29.4, got {version}")
        self.assertFalse(main_module.maintenance_supported())
        await plugin._maybe_consolidate_memories(event)

        fetch_candidates.assert_awaited_once_with(event, 14, 1, 30)
