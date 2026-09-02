"""记忆提取解析共享工具。

main.py 的自动提取路径与 maintenance.agents.curator 共用本模块；
置于包根而非 main.py，避免 maintenance 包反向 import main 形成循环。
"""

from __future__ import annotations

import json
import re
from typing import Any

try:  # 正常插件加载（包内相对导入）
    from .memory_protocol import MemoryScope, normalize_memory_scope
    from .prompts import (
        ALLOWED_MEMORY_TYPES,
        MAX_EXTRACTED_MEMORIES,
        sanitize_memory_content,
    )
except ImportError:  # 测试环境：仓库根作为顶层目录直接在 sys.path
    from memory_protocol import MemoryScope, normalize_memory_scope
    from prompts import (
        ALLOWED_MEMORY_TYPES,
        MAX_EXTRACTED_MEMORIES,
        sanitize_memory_content,
    )


def sanitize_string_list(value: Any, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []

    result = []
    for item in value[:limit]:
        text = sanitize_memory_content(str(item))[:80]
        if text:
            result.append(text)
    return result


def normalize_extracted_scope(scope: str, session_type: str) -> str:
    """规范化自动提取的记忆作用域。

    自动提取不能写入全局记忆；全局记忆只能由管理员工具显式创建。
    """
    scope = normalize_memory_scope(scope)
    if scope == MemoryScope.GLOBAL:
        return MemoryScope.PERSONAL
    if session_type != "group" and scope == MemoryScope.GROUP:
        return MemoryScope.PERSONAL
    return scope


def normalize_subject_id(subject: str) -> str:
    subject = subject.strip()
    for prefix in ("用户:", "user:", "sender:"):
        if subject.lower().startswith(prefix):
            return subject[len(prefix) :].strip()
    return subject


def normalize_subject_ids(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    raw_values = value if isinstance(value, list) else str(value).split(",")
    subjects = []
    for item in raw_values:
        subject = normalize_subject_id(sanitize_memory_content(str(item))[:120])
        if (
            subject
            and subject.lower() != "none"
            and subject not in {"current_sender", "group", "conversation"}
        ):
            subjects.append(subject)
    return list(dict.fromkeys(subjects))


def strip_json_fence(text: str) -> str:
    """移除 markdown JSON 围栏"""
    text = text.strip()
    if not text.startswith("```"):
        return text
    text = re.sub(r"^```\w*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def validate_extracted_memories(
    data: Any, session_type: str = "private"
) -> list[dict[str, Any]]:
    """校验并规范化 LLM 提取的记忆列表，带数量上限与注入清理。"""
    if not isinstance(data, list):
        return []

    validated = []
    for item in data[:MAX_EXTRACTED_MEMORIES]:  # 限制数量
        if not isinstance(item, dict):
            continue

        # 校验必需字段
        content = item.get("content", "")
        if not content or not isinstance(content, str):
            continue

        # 清理内容，防止 Prompt Injection
        content = sanitize_memory_content(content)
        if not content:
            continue

        # 校验并规范化字段
        mem_type = str(item.get("type", "fact")).lower()
        if mem_type not in ALLOWED_MEMORY_TYPES:
            mem_type = "fact"

        scope = normalize_extracted_scope(
            str(item.get("scope", "personal")), session_type
        )
        subjects = normalize_subject_ids(item.get("subjects"))
        if not subjects:
            subjects = normalize_subject_ids(item.get("subject", ""))
        subject = subjects[0] if subjects else ""
        if session_type == "group" and scope == MemoryScope.PERSONAL:
            if not subjects:
                continue
        entities = sanitize_string_list(item.get("entities", []))
        topics = sanitize_string_list(item.get("topics", []))
        disclosure = sanitize_memory_content(str(item.get("disclosure", "")))[:200]

        try:
            importance = int(item.get("importance", 3))
            importance = max(1, min(5, importance))
        except (TypeError, ValueError):
            importance = 3

        validated.append(
            {
                "scope": scope,
                "type": mem_type,
                "content": content,
                "subject": subject,
                "subjects": subjects,
                "entities": entities,
                "topics": topics,
                "disclosure": disclosure,
                "importance": importance,
            }
        )

    return validated


def parse_extracted_memories(
    text: str, session_type: str = "private"
) -> list[dict[str, Any]]:
    """解析 LLM 返回的记忆 JSON 文本，带校验和上限。"""
    text = strip_json_fence(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    return validate_extracted_memories(data, session_type)
