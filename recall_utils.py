"""召回输入预算与可重试错误分类。"""

from __future__ import annotations

import re
from typing import Any

DEFAULT_QUERY_MAX_CHARS = 1200
DEFAULT_RECALL_TIMEOUT = 12


def query_max_chars(config: dict[str, Any]) -> int:
    try:
        value = int(config.get("recall_query_max_chars", DEFAULT_QUERY_MAX_CHARS))
    except (TypeError, ValueError):
        value = DEFAULT_QUERY_MAX_CHARS
    return max(128, min(value, 2048))


def recall_timeout(config: dict[str, Any]) -> int:
    try:
        value = int(config.get("recall_timeout_seconds", DEFAULT_RECALL_TIMEOUT))
    except (TypeError, ValueError):
        value = DEFAULT_RECALL_TIMEOUT
    return max(2, min(value, 30))


def bound_query(query: str, limit: int = DEFAULT_QUERY_MAX_CHARS) -> str:
    return query[:limit].strip()


def is_query_length_error(error: Exception) -> bool:
    """只识别明确的输入长度超限，认证/限流/连接失败均不重试。"""
    status = getattr(error, "status_code", None)
    if status not in (None, 400, 413, 422):
        return False
    text = str(error).lower()
    return bool(
        "context_length_exceeded" in text
        or re.search(r"input.{0,80}tokens.{0,40}exceed", text)
        or re.search(r"maximum context length.{0,100}(tokens|exceed)", text)
        or re.search(r"(query|input).{0,30}(too long|max_length)", text)
    )
