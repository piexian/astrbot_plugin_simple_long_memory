"""maintenance 折叠配置迁移与平面视图的真实 AstrBotConfig 回归测试。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from astrbot.core.config.astrbot_config import AstrBotConfig
from real_stack import plugin_main

from config_compat import (
    MAINTENANCE_OBJECT_KEY,
    MIGRATION_MARKER_KEY,
    build_flat_config_view,
    migrate_maintenance_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((PROJECT_ROOT / "_conf_schema.json").read_text(encoding="utf-8"))

# 旧版顶层平面配置：覆盖布尔真假、字符串、整数与各组键
OLD_FLAT_CONFIG = {
    "kb_name": ["memory-kb"],
    "auto_memorize": False,
    "maintenance_enabled": True,
    "maintenance_window": "03:00-07:00",
    "maintenance_max_llm_calls": 30,
    "maintenance_analyst_detect_contradiction": False,
    "maintenance_curator_model_id": "provider-x",
    "auto_purge_after_days": 14,
    "review_notify_enabled": False,
    "review_notify_umo": "webchat:FriendMessage:admin",
    "context_max_rounds": 20,
    "persona_mode": "manual",
    "persona_summary": "简洁中文回复",
}


def _write_config(directory: str, payload: dict) -> Path:
    path = Path(directory) / "plugin_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MigrationRealConfigTests(unittest.TestCase):
    """用真实 AstrBotConfig + 插件 schema 验证迁移语义。"""

    def test_old_flat_config_migrates_into_object(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_config(directory, OLD_FLAT_CONFIG)
            cfg = AstrBotConfig(config_path=str(path), schema=SCHEMA)

            # 构造阶段：invisible 旧键保留原值，对象先补入默认值
            self.assertTrue(cfg["maintenance_enabled"])
            self.assertFalse(cfg[MAINTENANCE_OBJECT_KEY]["enabled"])

            self.assertTrue(migrate_maintenance_config(cfg))
            obj = cfg[MAINTENANCE_OBJECT_KEY]
            self.assertTrue(obj["enabled"])
            self.assertEqual(obj["window"], "03:00-07:00")
            self.assertEqual(obj["max_llm_calls"], 30)
            self.assertFalse(obj["analyst_detect_contradiction"])
            self.assertEqual(obj["curator_model_id"], "provider-x")
            self.assertEqual(obj["auto_purge_after_days"], 14)
            self.assertFalse(obj["review_notify_enabled"])
            self.assertEqual(obj["review_notify_umo"], "webchat:FriendMessage:admin")
            self.assertEqual(obj["context_max_rounds"], 20)
            self.assertEqual(obj["persona_mode"], "manual")
            self.assertEqual(obj["persona_summary"], "简洁中文回复")
            self.assertTrue(cfg[MIGRATION_MARKER_KEY])

            # 迁移已写盘：重新加载后标记与新值仍在
            reloaded = AstrBotConfig(config_path=str(path), schema=SCHEMA)
            self.assertTrue(reloaded[MIGRATION_MARKER_KEY])
            self.assertTrue(reloaded[MAINTENANCE_OBJECT_KEY]["enabled"])
            self.assertEqual(reloaded[MAINTENANCE_OBJECT_KEY]["window"], "03:00-07:00")

            # 现有代码经平面视图读到迁移后的值
            flat = build_flat_config_view(reloaded)
            self.assertTrue(flat["maintenance_enabled"])
            self.assertEqual(flat["maintenance_window"], "03:00-07:00")
            self.assertEqual(flat["maintenance_max_llm_calls"], 30)
            self.assertEqual(flat["review_notify_umo"], "webchat:FriendMessage:admin")

    def test_explicit_object_value_wins_over_old_key(self):
        raw = dict(OLD_FLAT_CONFIG)
        raw["maintenance"] = {"window": "23:00-01:00"}
        with tempfile.TemporaryDirectory() as directory:
            path = _write_config(directory, raw)
            cfg = AstrBotConfig(config_path=str(path), schema=SCHEMA)
            self.assertTrue(migrate_maintenance_config(cfg))
            obj = cfg[MAINTENANCE_OBJECT_KEY]
            # 对象中已显式改为非默认值的键保留新值
            self.assertEqual(obj["window"], "23:00-01:00")
            # 其余旧键仍正常迁移
            self.assertTrue(obj["enabled"])
            self.assertEqual(obj["persona_mode"], "manual")

    def test_second_load_default_revert_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_config(directory, OLD_FLAT_CONFIG)
            cfg = AstrBotConfig(config_path=str(path), schema=SCHEMA)
            self.assertTrue(migrate_maintenance_config(cfg))

            # 用户把对象值改回默认并保存
            reloaded = AstrBotConfig(config_path=str(path), schema=SCHEMA)
            reloaded[MAINTENANCE_OBJECT_KEY]["window"] = "02:00-06:00"
            reloaded.save_config()

            # 二次加载：标记已置位，遗留旧键不得再次覆盖默认值
            again = AstrBotConfig(config_path=str(path), schema=SCHEMA)
            self.assertFalse(migrate_maintenance_config(again))
            self.assertEqual(again[MAINTENANCE_OBJECT_KEY]["window"], "02:00-06:00")
            # 遗留旧键仍在文件里，但读取路径以对象为准
            self.assertEqual(again["maintenance_window"], "03:00-07:00")
            flat = build_flat_config_view(again)
            self.assertEqual(flat["maintenance_window"], "02:00-06:00")

    def test_steady_state_reload_does_not_rewrite_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_config(directory, OLD_FLAT_CONFIG)
            cfg = AstrBotConfig(config_path=str(path), schema=SCHEMA)
            self.assertTrue(migrate_maintenance_config(cfg))

            digest = _file_digest(path)
            for _ in range(2):
                AstrBotConfig(config_path=str(path), schema=SCHEMA)
                self.assertEqual(_file_digest(path), digest)

    def test_fresh_install_object_defaults_reach_flat_view(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_config(directory, {"kb_name": []})
            cfg = AstrBotConfig(config_path=str(path), schema=SCHEMA)
            obj = cfg[MAINTENANCE_OBJECT_KEY]
            self.assertFalse(obj["enabled"])
            # 迁移补齐此前仅存在于代码里的默认值
            self.assertEqual(obj["llm_timeout"], 120)
            self.assertEqual(obj["organizer_model_id"], "")
            self.assertEqual(obj["organizer_batch_size"], 30)
            self.assertEqual(obj["organizer_merge_batch_size"], 200)
            self.assertEqual(obj["analyst_max_merges"], 5)
            self.assertEqual(obj["analyst_batch_size"], 200)
            self.assertEqual(obj["reviewer_prompt_extra"], "")
            self.assertEqual(obj["verification_max_per_cycle"], 2)
            self.assertEqual(obj["verification_confidence_threshold"], 0.6)

            self.assertTrue(migrate_maintenance_config(cfg))
            flat = build_flat_config_view(cfg)
            self.assertEqual(flat["maintenance_llm_timeout"], 120)
            self.assertEqual(flat["maintenance_verification_max_per_cycle"], 2)
            self.assertEqual(flat["maintenance_verification_confidence_threshold"], 0.6)
            self.assertEqual(flat["maintenance_analyst_max_merges"], 5)
            self.assertEqual(flat["maintenance_organizer_model_id"], "")

    def test_organizer_merge_batch_key_drives_sampling(self):
        # merge 采样读独立的 organizer_merge_batch_size，巡检批量不受影响
        from maintenance.agents.organizer import OrganizerAgent

        def memories(count):
            return [
                {
                    "uri": f"facts://m{i}",
                    "vector": [1.0, 0.0],
                    "metadata": {"memory_scope": "group"},
                }
                for i in range(count)
            ]

        organizer = OrganizerAgent(
            context=None,
            memory_mgr=None,
            llm=None,
            config={"maintenance_organizer_merge_batch_size": 2},
        )
        sampled = asyncio.run(organizer._screen_merge_candidates(memories(5)))
        # 采样 2 条后只剩 1 个同作用域候选对
        self.assertEqual(len(sampled), 1)

        organizer = OrganizerAgent(context=None, memory_mgr=None, llm=None, config={})
        sampled = asyncio.run(organizer._screen_merge_candidates(memories(5)))
        # 缺省 200 不采样：5 条全保留，候选对 C(5,2)=10
        self.assertEqual(len(sampled), 10)


class PluginInitMigrationTests(unittest.TestCase):
    """真实 AstrBotConfig 直接进入插件构造函数的生产迁移路径。"""

    def test_plugin_init_migrates_and_builds_flat_view(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_config(directory, OLD_FLAT_CONFIG)
            cfg = AstrBotConfig(config_path=str(path), schema=SCHEMA)
            plugin = plugin_main.MemoryPlugin(None, cfg)
            self.assertTrue(plugin._flat_config.get("maintenance_enabled"))
            self.assertEqual(
                plugin._flat_config.get("maintenance_window"), "03:00-07:00"
            )
            self.assertEqual(plugin._flat_config.get("maintenance_max_llm_calls"), 30)
            # 通用键原样保留，折叠对象不进入平面视图
            self.assertEqual(plugin._flat_config.get("kb_name"), ["memory-kb"])
            self.assertNotIn(MAINTENANCE_OBJECT_KEY, plugin._flat_config)
            self.assertNotIn(MIGRATION_MARKER_KEY, plugin._flat_config)
            # 迁移与标记已写盘
            saved = json.loads(path.read_text(encoding="utf-8-sig"))
            self.assertTrue(saved[MIGRATION_MARKER_KEY])
            self.assertEqual(saved[MAINTENANCE_OBJECT_KEY]["window"], "03:00-07:00")

    def test_config_and_flat_view_are_one_mutable_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_config(directory, OLD_FLAT_CONFIG)
            cfg = AstrBotConfig(config_path=str(path), schema=SCHEMA)
            plugin = plugin_main.MemoryPlugin(None, cfg)
            # self.config 与 _flat_config 是同一对象，不存在会漂移的双源
            self.assertIs(plugin._flat_config, plugin.config)
            # 修改 self.config 后别名同步，MemoryManager 等消费者看到同一来源
            plugin.config["maintenance_enabled"] = False
            plugin.config["maintenance_window"] = "05:00-06:00"
            self.assertFalse(plugin._flat_config.get("maintenance_enabled"))
            self.assertEqual(
                plugin._flat_config.get("maintenance_window"), "05:00-06:00"
            )


class FlatViewTests(unittest.TestCase):
    """平面 dict（旧配置/单测直传）与对象优先级的纯函数行为。"""

    def test_object_value_overrides_stale_flat_key(self):
        cfg = {
            "maintenance_enabled": False,
            MAINTENANCE_OBJECT_KEY: {"enabled": True},
            MIGRATION_MARKER_KEY: True,
        }
        self.assertFalse(migrate_maintenance_config(cfg))
        flat = build_flat_config_view(cfg)
        self.assertTrue(flat["maintenance_enabled"])

    def test_plain_dict_object_absorbs_old_values_without_schema(self):
        cfg = {
            "maintenance_enabled": True,
            MAINTENANCE_OBJECT_KEY: {"enabled": False},
        }
        self.assertTrue(migrate_maintenance_config(cfg))
        self.assertTrue(cfg[MAINTENANCE_OBJECT_KEY]["enabled"])
        # 标记置位后迁移幂等，平面视图以对象为准
        self.assertFalse(migrate_maintenance_config(cfg))
        flat = build_flat_config_view(cfg)
        self.assertTrue(flat["maintenance_enabled"])


if __name__ == "__main__":
    unittest.main()
