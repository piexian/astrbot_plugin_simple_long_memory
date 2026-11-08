"""启动时中断恢复与孤儿关联清理顺序测试。"""

import unittest
from types import SimpleNamespace

from test_memory_commands import plugin_main


class StartupRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_orphan_cleanup_follows_recovery_in_both_loaded_paths(self):
        for already_connected in (False, True):
            with self.subTest(already_connected=already_connected):
                calls = []

                async def connect_kb():
                    calls.append("connect")

                async def cleanup_orphan_links():
                    calls.append("cleanup")
                    return 0

                async def recover():
                    calls.append("recover")

                async def start_scheduler():
                    calls.append("scheduler")

                manager = SimpleNamespace(
                    is_kb_connected=already_connected,
                    connect_kb=connect_kb,
                    cleanup_orphan_links=cleanup_orphan_links,
                )
                plugin = SimpleNamespace(
                    memory_mgr=manager,
                    _recover_interrupted_rebuild=recover,
                    _start_scheduler=start_scheduler,
                )

                await plugin_main.MemoryPlugin.on_loaded(plugin)

                expected = (
                    ["recover", "cleanup"]
                    if already_connected
                    else ["connect", "recover", "cleanup", "scheduler"]
                )
                self.assertEqual(calls, expected)


if __name__ == "__main__":
    unittest.main()
