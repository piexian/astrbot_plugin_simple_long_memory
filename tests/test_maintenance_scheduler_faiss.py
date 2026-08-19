"""后台整理调度与 FAISS 向量读取回归测试。"""

import json
import sys
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from maintenance.scheduler import (  # noqa: E402
    _is_in_maintenance_window,
    _next_run,
    _parse_maintenance_window,
)
from astrbot_plugin_simple_long_memory.memory_manager import MemoryManager  # noqa: E402


class SchedulerRegressionTests(unittest.TestCase):
    def test_exact_cron_boundary_advances_strictly(self):
        base = datetime.fromisoformat("2026-08-19T03:00:00+08:00")
        next_run = _next_run("0 3 * * *", after=base, tz_str="Asia/Shanghai")

        self.assertGreater(next_run, base)
        self.assertEqual(next_run.isoformat(), "2026-08-20T03:00:00+08:00")

    def test_non_boundary_cron_advances_to_next_day(self):
        base = datetime.fromisoformat("2026-08-19T03:00:01+08:00")
        next_run = _next_run("0 3 * * *", after=base, tz_str="Asia/Shanghai")

        self.assertEqual(next_run.isoformat(), "2026-08-20T03:00:00+08:00")

    def test_maintenance_window_supports_regular_and_overnight_ranges(self):
        self.assertEqual(_parse_maintenance_window("02:00-06:00"), (120, 360))
        self.assertEqual(_parse_maintenance_window("03:00"), (180, 180))

        self.assertTrue(
            _is_in_maintenance_window(
                datetime.fromisoformat("2026-08-19T03:00:00+08:00"),
                "02:00-06:00",
            )
        )
        self.assertFalse(
            _is_in_maintenance_window(
                datetime.fromisoformat("2026-08-19T07:00:00+08:00"),
                "02:00-06:00",
            )
        )
        self.assertTrue(
            _is_in_maintenance_window(
                datetime.fromisoformat("2026-08-19T00:30:00+08:00"),
                "23:00-02:00",
            )
        )


class FaissIDMapRegressionTests(unittest.TestCase):
    def setUp(self):
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(3))
        self.index.add_with_ids(
            np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            np.asarray([101, 303], dtype=np.int64),
        )
        self.prepared = MemoryManager._prepare_faiss_index(self.index)

    def test_id_map_resolves_external_document_id_to_internal_position(self):
        vector, status, detail = MemoryManager._reconstruct_faiss_vector(
            self.index,
            303,
            self.prepared,
        )

        self.assertEqual(status, "ok", detail)
        self.assertEqual(vector, [4.0, 5.0, 6.0])

    def test_missing_id_is_reported_without_silent_failure(self):
        vector, status, detail = MemoryManager._reconstruct_faiss_vector(
            self.index,
            999,
            self.prepared,
        )

        self.assertIsNone(vector)
        self.assertEqual(status, "id_not_found")
        self.assertIn("999", detail)


class ActiveMemoryVectorLoadTests(unittest.IsolatedAsyncioTestCase):
    async def test_active_memory_fetch_loads_id_map_vectors_and_counts_missing(self):
        index = faiss.IndexIDMap(faiss.IndexFlatL2(3))
        index.add_with_ids(
            np.asarray([[4.0, 5.0, 6.0]], dtype=np.float32),
            np.asarray([303], dtype=np.int64),
        )

        class DocumentStorage:
            async def get_documents(self, **kwargs):
                self.filters = kwargs
                return [
                    {
                        "id": 303,
                        "text": "with vector",
                        "metadata": json.dumps({"uri": "memory://303"}),
                    },
                    {
                        "id": 999,
                        "text": "without vector",
                        "metadata": json.dumps({"uri": "memory://999"}),
                    },
                ]

        vec_db = SimpleNamespace(
            document_storage=DocumentStorage(),
            embedding_storage=SimpleNamespace(index=index),
        )
        manager = MemoryManager(kb_mgr=None, config={})
        manager._kb_helper = SimpleNamespace(vec_db=vec_db)
        memories = await manager.get_all_active_memories(limit=10)

        self.assertEqual(memories[0]["vector"], [4.0, 5.0, 6.0])
        self.assertIsNone(memories[1]["vector"])
        self.assertEqual(manager.last_active_memory_stats["vectors_loaded"], 1)
        self.assertEqual(manager.last_active_memory_stats["vectors_missing"], 1)
        self.assertEqual(
            manager.last_active_memory_stats["missing_by_reason"],
            {"id_not_found": 1},
        )


if __name__ == "__main__":
    unittest.main()
