import importlib
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class _Logger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _identity_decorator(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


class _Filter:
    on_astrbot_loaded = staticmethod(_identity_decorator)
    on_llm_request = staticmethod(_identity_decorator)
    on_llm_response = staticmethod(_identity_decorator)
    llm_tool = staticmethod(_identity_decorator)

    @staticmethod
    def command_group(*args, **kwargs):
        def decorator(func):
            func.command = _identity_decorator
            return func

        return decorator


class _Star:
    pass


def _load_plugin_modules():
    project_parent = Path(__file__).resolve().parents[2]
    if str(project_parent) not in sys.path:
        sys.path.insert(0, str(project_parent))

    astrbot_module = types.ModuleType("astrbot")
    api_module = types.ModuleType("astrbot.api")
    event_module = types.ModuleType("astrbot.api.event")
    provider_module = types.ModuleType("astrbot.api.provider")
    star_module = types.ModuleType("astrbot.api.star")

    api_module.logger = _Logger()
    event_module.AstrMessageEvent = type("AstrMessageEvent", (), {})
    event_module.filter = _Filter()
    provider_module.LLMResponse = type("LLMResponse", (), {})
    provider_module.ProviderRequest = type("ProviderRequest", (), {})
    star_module.Context = type("Context", (), {})
    star_module.Star = _Star

    modules = {
        "astrbot": astrbot_module,
        "astrbot.api": api_module,
        "astrbot.api.event": event_module,
        "astrbot.api.provider": provider_module,
        "astrbot.api.star": star_module,
    }
    with patch.dict(sys.modules, modules):
        main_module = importlib.import_module("astrbot_plugin_simple_long_memory.main")
        manager_module = importlib.import_module(
            "astrbot_plugin_simple_long_memory.memory_manager"
        )
    return main_module, manager_module


plugin_main, memory_manager_module = _load_plugin_modules()
MemoryManager = memory_manager_module.MemoryManager


class _Event:
    def __init__(self, message_str: str, admin: bool = False):
        self.message_str = message_str
        self._admin = admin

    def plain_result(self, text: str) -> str:
        return text

    def is_admin(self) -> bool:
        return self._admin

    def get_sender_id(self) -> str:
        return "42"

    def get_platform_id(self) -> str:
        return "test"


async def _collect(handler, plugin, event):
    return [item async for item in handler(plugin, event)]


def _memory(index: int) -> dict:
    return {
        "text": f"memory: 记忆 {index}",
        "metadata": {
            "uri": f"facts://memory-{index}",
            "created_at": "2026-07-10T00:00:00+00:00",
        },
    }


class MemoryCommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_list_rejects_invalid_page(self):
        class Manager:
            called = False

            async def list_memories(self, *args, **kwargs):
                self.called = True
                return [], 0, False

        for value in ("invalid", "0", "-1"):
            with self.subTest(value=value):
                manager = Manager()
                plugin = SimpleNamespace(
                    memory_mgr=manager,
                    _get_cmd_prefix=lambda: "/",
                )
                result = await _collect(
                    plugin_main.MemoryPlugin.cmd_list,
                    plugin,
                    _Event(f"memory list {value}"),
                )

                self.assertEqual(result, ["页码必须是正整数"])
                self.assertFalse(manager.called)

    async def test_search_renders_all_results_as_one_page(self):
        class Manager:
            async def recall_memories(self, *args, **kwargs):
                return [_memory(index) for index in range(1, 13)]

        plugin = SimpleNamespace(
            memory_mgr=Manager(),
            _get_cmd_prefix=lambda: "!",
        )
        result = await _collect(
            plugin_main.MemoryPlugin.cmd_search,
            plugin,
            _Event("memory search --all 关键词", admin=True),
        )

        self.assertIn("第 1/1 页", result[0])
        self.assertIn("\n12. [facts://memory-12]", result[0])
        self.assertNotIn("查看下一页", result[0])

    async def test_memory_test_reports_recall_failure(self):
        class Manager:
            async def store_memory(self, **kwargs):
                return kwargs["uri"]

            async def recall_memories(self, **kwargs):
                raise RuntimeError("embedding unavailable")

            async def forget_memory(self, **kwargs):
                return 1

        plugin = SimpleNamespace(memory_mgr=Manager())
        result = await plugin_main.MemoryPlugin._run_memory_test(
            plugin, _Event("memory test", admin=True)
        )

        self.assertIn("召回: 失败 (embedding unavailable)", result)
        self.assertIn("清理: 已删除测试记忆", result)
        self.assertIn("结论: 召回异常，请检查 embedding 配置", result)

    async def test_stats_label_marks_compressed_count_as_history(self):
        class Manager:
            async def get_memory_stats(self, *args, **kwargs):
                return {"total": 3, "permanent": 1, "normal": 2, "compressed": 4}

        plugin = SimpleNamespace(memory_mgr=Manager())
        result = await _collect(
            plugin_main.MemoryPlugin.cmd_stats,
            plugin,
            _Event("memory stats"),
        )

        self.assertIn("历史已压缩: 4", result[0])

    async def test_rebuild_completion_uses_configured_prefix(self):
        class Manager:
            current_kb_name = "source"

            def __init__(self, verification):
                self.verification = verification

            async def get_kb_id_by_name(self, name):
                return "source-id"

            async def rebuild_memories(self, target_kb_name=None):
                return {
                    "status": "completed",
                    "is_migration": False,
                    "target_kb": "source",
                    "total": 2,
                    "success": 2,
                    "failed": 0,
                    "pending_flushed": 0,
                    "verification": self.verification,
                    "migration_committed": False,
                }

        confirm_code = plugin_main._confirmation_code(
            "rebuild", "source:source-id:source-id"
        )
        for verification in (
            {"passed": True, "expected": 2, "actual": 2},
            {"passed": False, "expected": 2, "actual": 1, "diff": -1},
        ):
            with self.subTest(passed=verification["passed"]):
                plugin = SimpleNamespace(
                    memory_mgr=Manager(verification),
                    _get_cmd_prefix=lambda: "!",
                )
                result = await _collect(
                    plugin_main.MemoryPlugin.cmd_rebuild,
                    plugin,
                    _Event(f"memory rebuild --confirm {confirm_code}", admin=True),
                )

                self.assertIn("!memory rebuild --clear-cache", result[-1])
                self.assertNotIn("/memory rebuild --clear-cache", result[-1])


class _CountingVectorDB:
    def __init__(self):
        self.filters = []

    async def count_documents(self, metadata_filter):
        self.filters.append(dict(metadata_filter))
        if metadata_filter.get("compressed"):
            return 4
        if metadata_filter.get("memory_type") == "permanent":
            return 1
        if metadata_filter.get("memory_type") == "normal":
            return 2
        return 3


class MemoryManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_stats_use_active_records_and_include_compressed_history(self):
        for all_users, base_filter in (
            (False, {"user_id": "test_42"}),
            (True, {"is_memory_record": True}),
        ):
            with self.subTest(all_users=all_users):
                vec_db = _CountingVectorDB()
                manager = MemoryManager(kb_mgr=None, config={})
                manager._kb_helper = SimpleNamespace(vec_db=vec_db)

                result = await manager.get_memory_stats(
                    _Event("memory stats", admin=all_users),
                    all_users=all_users,
                )

                self.assertEqual(
                    result,
                    {"total": 3, "permanent": 1, "normal": 2, "compressed": 4},
                )
                self.assertEqual(
                    vec_db.filters[0],
                    {**base_filter, "deprecated": False},
                )
                self.assertEqual(
                    vec_db.filters[3],
                    {**base_filter, "compressed": True},
                )

    async def test_empty_migration_commits_target_switch(self):
        class DocumentStorage:
            async def get_documents(self, **kwargs):
                return []

        class VectorDB:
            def __init__(self):
                self.document_storage = DocumentStorage()
                self.deleted_filters = []

            async def count_documents(self, metadata_filter):
                return 0

            async def delete_documents(self, metadata_filters):
                self.deleted_filters.append(dict(metadata_filters))

        source = SimpleNamespace(
            kb=SimpleNamespace(kb_id="source-id"),
            vec_db=VectorDB(),
        )
        target = SimpleNamespace(
            kb=SimpleNamespace(kb_id="target-id"),
            vec_db=VectorDB(),
        )

        class KBManager:
            async def get_kb_by_name(self, name):
                return target if name == "target" else None

        manager = MemoryManager(kb_mgr=KBManager(), config={})
        manager._kb_helper = source
        manager._kb_name = "source"

        async def sync_stats(kb_helper=None):
            return None

        manager._sync_kb_stats = sync_stats

        result = await manager.rebuild_memories(target_kb_name="target")

        self.assertTrue(result["migration_committed"])
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["verification"]["passed"])
        self.assertIs(manager._kb_helper, target)
        self.assertEqual(manager.current_kb_name, "target")


if __name__ == "__main__":
    unittest.main()
