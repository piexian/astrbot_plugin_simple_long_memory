"""记忆关联孤儿清理回归测试。"""

import importlib
import json
import unittest
from types import SimpleNamespace

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import StaticPool
from test_memory_commands import plugin_main  # noqa: F401

links_module = importlib.import_module(
    "astrbot_plugin_simple_long_memory.maintenance.links"
)
MemoryLinkManager = links_module.MemoryLinkManager


class _DocumentStorage:
    def __init__(self, engine):
        self.engine = engine

    def get_session(self):
        return AsyncSession(self.engine, expire_on_commit=False)


class MemoryLinkCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:", poolclass=StaticPool
        )
        storage = _DocumentStorage(self.engine)
        self.manager = MemoryLinkManager(SimpleNamespace(document_storage=storage))
        await self.manager.ensure_table()
        async with self.engine.begin() as conn:
            await conn.execute(text("CREATE TABLE documents (metadata TEXT)"))
            for uri, deprecated in (
                ("facts://active-a", 0),
                ("facts://active-b", 0),
                ("facts://deprecated", 1),
            ):
                await conn.execute(
                    text("INSERT INTO documents VALUES (:metadata)"),
                    {"metadata": json.dumps({"uri": uri, "deprecated": deprecated})},
                )

    async def asyncTearDown(self):
        await self.engine.dispose()

    async def test_cleanup_removes_only_missing_document_endpoints(self):
        for source, target in (
            ("facts://active-a", "facts://active-b"),
            ("facts://active-a", "facts://deprecated"),
            ("facts://missing-source", "facts://active-a"),
            ("facts://active-a", "facts://missing-target"),
        ):
            self.assertTrue(await self.manager.add_link(source, target, "related"))

        removed = await self.manager.cleanup_orphan_links()
        remaining = await self.manager.export_all()
        repeated = await self.manager.cleanup_orphan_links()

        self.assertEqual(removed, 2)
        self.assertEqual(
            {(row["source_uri"], row["target_uri"]) for row in remaining},
            {
                ("facts://active-a", "facts://active-b"),
                ("facts://active-a", "facts://deprecated"),
            },
        )
        self.assertEqual(repeated, 0)


if __name__ == "__main__":
    unittest.main()
