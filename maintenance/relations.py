"""关系判定共享工具：矛盾新旧定向、纯文本提取与数值校验。"""

from __future__ import annotations

import math
from typing import Any

RELATION_TYPES = ("supports", "supersedes", "context", "related")
PAIR_VERDICTS = ("merge", "link", "conflict", "none")


def valid_unit_number(value: Any) -> float | None:
    """校验 0 到 1 的有限数值（拒收 NaN/无穷/布尔），无效返回 None。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return None
    return number


def _memory_time(mem: dict[str, Any]) -> str:
    meta = mem.get("metadata", {})
    return str(meta.get("updated_at", "") or meta.get("created_at", "") or "")


def orient_conflict(mem_a: dict[str, Any], mem_b: dict[str, Any]) -> tuple[str, str]:
    """按 updated_at/created_at 确定矛盾对的 (old_uri, new_uri)。"""
    time_a = _memory_time(mem_a)
    time_b = _memory_time(mem_b)
    if time_a >= time_b:
        return mem_b["uri"], mem_a["uri"]
    return mem_a["uri"], mem_b["uri"]


def extract_plain_text(content: Any) -> str:
    """从消息 content 提取纯文本，跳过思考块、临时块与非文本块。"""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                if part:
                    parts.append(part)
            elif isinstance(part, dict):
                if part.get("_no_save") or part.get("type") != "text":
                    continue
                text = str(part.get("text") or "")
                if text:
                    parts.append(text)
        return "".join(parts).strip()
    return ""
