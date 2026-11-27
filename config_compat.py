"""maintenance 折叠配置的迁移与平面视图辅助。

现有代码（Runner、Scheduler、Reviewer 等）继续读取 maintenance_* 等平面键，
本模块把 _conf_schema.json 中 maintenance 嵌套对象的值映射回平面键，
并在首次加载时把旧顶层配置键一次性迁入嵌套对象。
"""

from __future__ import annotations

from typing import Any

MAINTENANCE_OBJECT_KEY = "maintenance"
"""唯一可见的后台整理配置折叠组。"""

MIGRATION_MARKER_KEY = "maintenance_config_migrated"
"""隐藏的迁移完成标记：置 True 后遗留旧键不再参与读取或覆盖。"""

MAINTENANCE_KEY_MAP: dict[str, str] = {
    # 现有代码使用的平面键 -> maintenance 对象内短键
    "maintenance_enabled": "enabled",
    "maintenance_model_id": "model_id",
    "maintenance_window": "window",
    "maintenance_cron": "cron",
    "maintenance_max_llm_calls": "max_llm_calls",
    "maintenance_max_ops_per_cycle": "max_ops_per_cycle",
    "maintenance_llm_timeout": "llm_timeout",
    "maintenance_pending_queue_max": "pending_queue_max",
    "maintenance_extract_enabled": "extract_enabled",
    "maintenance_curator_model_id": "curator_model_id",
    "maintenance_segment_time_gap_minutes": "segment_time_gap_minutes",
    "maintenance_segment_max_chars": "segment_max_chars",
    "maintenance_extract_max_blocks_per_cycle": "extract_max_blocks_per_cycle",
    "maintenance_extract_llm_budget_ratio": "extract_llm_budget_ratio",
    "maintenance_organizer_enabled": "organizer_enabled",
    "maintenance_organizer_model_id": "organizer_model_id",
    "maintenance_organizer_batch_size": "organizer_batch_size",
    "maintenance_organizer_merge_batch_size": "organizer_merge_batch_size",
    "maintenance_analyst_enabled": "analyst_enabled",
    "maintenance_analyst_max_new_links": "analyst_max_new_links",
    "maintenance_analyst_detect_contradiction": "analyst_detect_contradiction",
    "maintenance_analyst_max_contradictions": "analyst_max_contradictions",
    "maintenance_analyst_max_merges": "analyst_max_merges",
    "maintenance_analyst_batch_size": "analyst_batch_size",
    "maintenance_reviewer_enabled": "reviewer_enabled",
    "maintenance_reviewer_model_id": "reviewer_model_id",
    "maintenance_reviewer_prompt_extra": "reviewer_prompt_extra",
    "maintenance_verification_max_per_cycle": "verification_max_per_cycle",
    "maintenance_verification_confidence_threshold": (
        "verification_confidence_threshold"
    ),
    "auto_purge_enabled": "auto_purge_enabled",
    "auto_purge_after_days": "auto_purge_after_days",
    "auto_purge_cron": "auto_purge_cron",
    "review_notify_enabled": "review_notify_enabled",
    "review_notify_umo": "review_notify_umo",
    "review_notify_schedule": "review_notify_schedule",
    "review_notify_time": "review_notify_time",
    "context_max_rounds": "context_max_rounds",
    "context_max_chars": "context_max_chars",
    "context_max_age_days": "context_max_age_days",
    "persona_mode": "persona_mode",
    "persona_summary": "persona_summary",
}


def build_flat_config_view(config: Any) -> dict[str, Any]:
    """合并 maintenance 对象与顶层平面键，返回现有代码使用的平面配置字典。

    对象中的值优先；对象缺失（旧配置或测试传入的平面 dict）时直接沿用平面键。
    """
    if not isinstance(config, dict):
        return {}
    flat = {
        key: value
        for key, value in config.items()
        if key not in (MAINTENANCE_OBJECT_KEY, MIGRATION_MARKER_KEY)
    }
    obj = config.get(MAINTENANCE_OBJECT_KEY)
    if isinstance(obj, dict):
        for code_key, short_key in MAINTENANCE_KEY_MAP.items():
            value = obj.get(short_key)
            if value is not None:
                flat[code_key] = value
    return flat


def _schema_defaults(config: Any) -> dict[str, Any]:
    """从 AstrBotConfig 携带的 schema 读取 maintenance 对象内各键默认值。"""
    schema = getattr(config, "schema", None)
    if not isinstance(schema, dict):
        return {}
    items = (schema.get(MAINTENANCE_OBJECT_KEY) or {}).get("items")
    if not isinstance(items, dict):
        return {}
    return {
        short_key: item.get("default")
        for short_key, item in items.items()
        if isinstance(item, dict)
    }


def migrate_maintenance_config(
    config: Any,
    *,
    defaults: dict[str, Any] | None = None,
) -> bool:
    """一次性把旧顶层维护键迁入 maintenance 对象，返回是否执行了本次迁移。

    对象中已显式改为非默认值的键保留新值；迁移完成写入隐藏标记并保存配置，
    之后用户把对象值改回默认也不会再被遗留旧值覆盖。无嵌套对象（旧版配置、
    测试平面 dict）时无事可做，不写标记。
    """
    if not isinstance(config, dict) or config.get(MIGRATION_MARKER_KEY) is True:
        return False
    obj = config.get(MAINTENANCE_OBJECT_KEY)
    if not isinstance(obj, dict):
        return False
    if defaults is None:
        defaults = _schema_defaults(config)
    for code_key, short_key in MAINTENANCE_KEY_MAP.items():
        old_value = config.get(code_key)
        if old_value is None:
            continue
        current = obj.get(short_key)
        known_default = defaults.get(short_key)
        if (
            known_default is not None
            and current is not None
            and current != known_default
        ):
            continue
        obj[short_key] = old_value
    config[MIGRATION_MARKER_KEY] = True
    save = getattr(config, "save_config", None)
    if callable(save):
        save()
    return True
