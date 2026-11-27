"""
AstrBot 长期记忆插件主入口

功能:
- LLM 请求前注入记忆上下文
- LLM 响应后自动提取记忆
- 提供记忆管理命令
- 提供 LLM 工具供 AI 主动操作记忆
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import re
import shlex
import time
import weakref
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.api.util import SessionController, SessionWaiter

from .config_compat import build_flat_config_view, migrate_maintenance_config
from .maintenance.inbox import (
    INBOX_ASSISTANT_BRIEF_MAX_CHARS,
    INBOX_USER_TEXT_MAX_CHARS,
    InboxRecord,
)
from .maintenance.memory_gate import (
    GateCandidate,
    GateOutcome,
    LlmGenerateJudge,
    MemoryGate,
)
from .maintenance.scheduler import MaintenanceScheduler
from .maintenance.version_gate import (
    maintenance_supported,
    unsupported_maintenance_reason,
)
from .memory_manager import (
    MemoryManager,
    _debug_content_summary,
    _debug_memory_summary,
    normalize_domain,
)
from .memory_protocol import (
    MemoryScope,
    MemoryType,
    MemoryURI,
    UMOInfo,
    build_user_id,
    format_memory_for_injection,
    format_memory_for_user,
    validate_owner_ids,
)
from .recall_utils import bound_query, query_max_chars, recall_timeout

if TYPE_CHECKING:
    from .memory_manager import MemoryManager

# 提取解析共享逻辑已抽至 extraction_utils（maintenance.agents.curator 共用），
# 以下同名别名仅为兼容外部引用而保留（re-export），本文件内不再直接使用
from .extraction_utils import (
    normalize_extracted_scope as _normalize_extracted_scope,  # noqa: F401
)
from .extraction_utils import (
    normalize_subject_id as _normalize_subject_id,  # noqa: F401
)
from .extraction_utils import (
    normalize_subject_ids as _normalize_subject_ids,  # noqa: F401
)
from .extraction_utils import (
    parse_extracted_memories as _shared_parse_extracted_memories,
)
from .extraction_utils import (
    sanitize_string_list as _sanitize_string_list,  # noqa: F401
)
from .extraction_utils import (
    strip_json_fence as _strip_json_fence_text,
)
from .prompts import (
    ALLOWED_MEMORY_TYPES,
    MEMORY_CONSOLIDATION_PROMPT,
    MEMORY_EXTRACTION_PROMPT,
    RECALL_QUERY_PROMPT,
    VALID_TOOL_DOMAINS,
    VALID_TOOL_SCOPES,
)
from .prompts import (
    sanitize_memory_content as _sanitize_memory_content,
)

DEFAULT_RECALL_QUERY_OPTIMIZATION_TIMEOUT = 10
DEFAULT_MAX_SESSION_SNAPSHOTS = 20

# 命令轮次排除：本插件命令组与核心管理命令（核心会剥离唤醒前缀，两种形式等价）
INBOX_COMMAND_TOKENS = frozenset({"memory", "reset"})
INBOX_MIN_USER_TEXT_CHARS = 2
"""低于该用户文本长度的轮次视为无有效用户输入，不入缓冲表。"""
INBOX_LOW_INFO_ASSISTANT_CHARS = 200
"""用户文本极短且助手输出超过该长度时视为单向回显，不入缓冲表。"""
INBOX_LOW_INFO_USER_CHARS = 8
"""判定单向回显时用户文本的极短阈值。"""
GATE_TOOL_TIMEOUT_SECONDS = 20
"""memory_store 工具与实时提取的闸门 Chat 判定超时。"""
DEFAULT_MAX_SNAPSHOT_CHARS = 8000


def _current_speaker_subject(event: AstrMessageEvent, scope: str) -> str:
    parsed = UMOInfo.parse(event.unified_msg_origin)
    if scope == MemoryScope.GROUP:
        return parsed.session_id
    if scope == MemoryScope.CONVERSATION:
        return event.unified_msg_origin
    return event.get_sender_id()


def _flatten_content(content: Any) -> str:
    """将内容转换为字符串，跳过临时块与非纯文本块。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(part for item in content if (part := _flatten_content(item)))
    if isinstance(content, dict):
        if content.get("_no_save"):
            return ""
        part_type = content.get("type")
        if part_type and part_type != "text":
            return ""
        return str(content.get("text") or content.get("content") or "")
    if getattr(content, "_no_save", False):
        return ""
    part_type = getattr(content, "type", None)
    if part_type is not None:
        if part_type != "text":
            return ""
        return str(getattr(content, "text", "") or "")
    return str(content)


def _noop_decorator(*args: Any, **kwargs: Any):
    def decorator(func: Any) -> Any:
        return func

    return decorator


def _on_agent_begin_decorator(*args: Any, **kwargs: Any):
    factory = getattr(filter, "on_agent_begin", None)
    if callable(factory):
        return factory(*args, **kwargs)
    return _noop_decorator(*args, **kwargs)


def _on_agent_done_decorator(*args: Any, **kwargs: Any):
    factory = getattr(filter, "on_agent_done", None)
    if callable(factory):
        return factory(*args, **kwargs)
    return _noop_decorator(*args, **kwargs)


def _is_wakeup_event(event: Any) -> bool:
    """判断是否为 AstrBot 4.30+ 程序唤醒事件（定时任务、后台任务、run_turn）。"""
    return getattr(event, "event_kind", "message") == "agent_wakeup"


def _has_real_wakeup_sender(event: Any) -> bool:
    """判断唤醒事件是否携带真实发送者，而非框架默认的 System/astrbot 占位身份。"""
    get_sender_id = getattr(event, "get_sender_id", None)
    sender_id = str((get_sender_id() if callable(get_sender_id) else "") or "").strip()
    if not sender_id:
        return False
    get_sender_name = getattr(event, "get_sender_name", None)
    sender_name = str(
        (get_sender_name() if callable(get_sender_name) else "") or ""
    ).strip()
    if sender_id == "astrbot" and sender_name in ("", "System"):
        return False
    get_self_id = getattr(event, "get_self_id", None)
    self_id = str((get_self_id() if callable(get_self_id) else "") or "").strip()
    if self_id and sender_id == self_id:
        return False
    return True


def _is_subagent_run(run_context: Any) -> bool:
    """判断当前 Agent 运行上下文是否属于原生子代理。"""
    return bool(getattr(getattr(run_context, "context", None), "is_subagent", False))


def _read_astrbot_config(context: Any, umo: str | None = None) -> Any:
    """安全读取 AstrBot 全局或会话级配置，兼容仅暴露 astrbot_config 的测试桩。"""
    if context is None:
        return None
    get_config = getattr(context, "get_config", None)
    if callable(get_config):
        if umo:
            try:
                return get_config(umo=umo)
            except TypeError:
                pass
        return get_config()
    return getattr(context, "astrbot_config", None)


def _extract_visible_message_text(message: Any) -> str:
    """从 AstrBot Message 对象或字典中提取可持久化的纯文本内容。"""
    if isinstance(message, dict):
        if message.get("_no_save"):
            return ""
        content = message.get("content")
    else:
        if getattr(message, "_no_save", False):
            return ""
        content = getattr(message, "content", None)

    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            if part:
                parts.append(part)
            continue
        if isinstance(part, dict):
            if part.get("_no_save") or part.get("type") != "text":
                continue
            text = str(part.get("text") or "")
            if text:
                parts.append(text)
            continue
        if getattr(part, "_no_save", False) or getattr(part, "type", None) != "text":
            continue
        text = str(getattr(part, "text", "") or "")
        if text:
            parts.append(text)
    return "".join(parts).strip()


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or "")
    return str(getattr(message, "role", "") or "")


def _normalize_contexts(contexts: Any) -> list[dict[str, Any]]:
    """标准化 contexts 为列表"""
    return list(contexts) if isinstance(contexts, list) else []


def _build_recall_query(
    prompt: str, contexts: list[dict[str, Any]], max_chars: int = 1200
) -> str:
    """优先保留当前提问，用剩余预算补最近的自然语言上下文。"""
    query = bound_query(prompt, max_chars)
    for ctx in reversed(contexts[-3:]):
        if not isinstance(ctx, dict) or ctx.get("role") not in {"user", "assistant"}:
            continue
        remaining = max_chars - len(query)
        if remaining <= 15:
            break
        content = _flatten_content(ctx.get("content", ""))
        if content:
            query += f"\n[{ctx['role']}]: {content[:400]}"[:remaining]
    return query.strip()


def _make_memory_text_part(content: str) -> Any | None:
    """构造 AstrBot 用户内容块；新版支持标记为本轮临时内容。"""
    try:
        from astrbot.api.agents import TextPart
    except Exception:
        try:
            from astrbot.core.agent.message import TextPart
        except Exception:
            return None

    part = TextPart(text=content)
    mark_as_temp = getattr(part, "mark_as_temp", None)
    if callable(mark_as_temp):
        marked_part = mark_as_temp()
        if marked_part is not None:
            part = marked_part
    return part


def _append_extra_user_content(request: ProviderRequest, content: str) -> bool:
    """优先使用 AstrBot 动态用户内容块注入记忆。"""
    if not hasattr(request, "extra_user_content_parts"):
        return False

    parts = getattr(request, "extra_user_content_parts", None)
    if parts is None:
        parts = []
        request.extra_user_content_parts = parts
    if not hasattr(parts, "append"):
        return False

    part = _make_memory_text_part(content)
    if part is None:
        return False

    parts.append(part)
    return True


def _inject_memory_context(request: ProviderRequest, content: str) -> str:
    """注入记忆上下文，返回实际使用的注入位置。"""
    if _append_extra_user_content(request, content):
        return "extra_user_content_parts"

    contexts = _normalize_contexts(getattr(request, "contexts", None))
    memory_msg = {"role": "user", "content": content}
    # 旧版回退：放在 contexts 最前面，使当前 prompt 仍保持最后、优先级更高。
    request.contexts = [memory_msg] + contexts
    return "contexts 顶部"


def _clamp_timeout(
    value: Any, default: int = DEFAULT_RECALL_QUERY_OPTIMIZATION_TIMEOUT
) -> int:
    try:
        timeout = int(value)
    except (TypeError, ValueError):
        timeout = default
    return max(1, min(20, timeout))


def _read_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0, parsed)


def _confirmation_code(action: str, target: str = "") -> str:
    seed = f"{action}:{target}".encode()
    digest = hashlib.sha256(seed).hexdigest()[:8]
    return f"{action}-{digest}"


CONFIRMATION_TIMEOUT_SECONDS = 60
CONFIRMATION_WORD = "确认"
CONFIRMATION_CANCEL_WORDS = frozenset({"取消", "退出"})


@dataclass(frozen=True)
class ConfirmationResult:
    status: str
    event: AstrMessageEvent | None = None


class _ConfirmationSessionFilter:
    """Bind a confirmation session to one UMO and its original sender."""

    def __init__(self, event: AstrMessageEvent):
        self._umo = event.unified_msg_origin
        self._sender_id = str(event.get_sender_id())
        self.session_id = (
            f"simple_long_memory:confirmation:{self._umo}:{self._sender_id}"
        )

    def filter(self, event: AstrMessageEvent) -> str:
        if (
            event.unified_msg_origin == self._umo
            and str(event.get_sender_id()) == self._sender_id
        ):
            return self.session_id
        return f"{self.session_id}:unmatched:{id(self)}"


async def _wait_for_confirmation(
    event: AstrMessageEvent,
    waiters: set[SessionWaiter],
    *,
    timeout: int = CONFIRMATION_TIMEOUT_SECONDS,
) -> ConfirmationResult:
    """Wait for an exact confirmation message from the original sender."""
    session_filter = _ConfirmationSessionFilter(event)
    if any(waiter.session_id == session_filter.session_id for waiter in waiters):
        return ConfirmationResult("busy")

    waiter = SessionWaiter(session_filter, session_filter.session_id, False)
    waiters.add(waiter)
    result: ConfirmationResult | None = None

    async def handle(controller: SessionController, incoming: AstrMessageEvent) -> None:
        nonlocal result
        text = (incoming.message_str or "").strip()
        if text == CONFIRMATION_WORD:
            result = ConfirmationResult("confirmed", incoming)
            controller.stop()
            return
        if text in CONFIRMATION_CANCEL_WORDS:
            result = ConfirmationResult("cancelled", incoming)
            controller.stop()
            return
        await incoming.send(incoming.plain_result("请输入“确认”执行，发送“取消”放弃。"))

    try:
        await waiter.register_wait(handle, timeout=timeout)
    except TimeoutError:
        return ConfirmationResult("timeout")
    finally:
        waiters.discard(waiter)
    return result or ConfirmationResult("cancelled")


ADMIN_MANAGED_SCOPES = {
    MemoryScope.GLOBAL,
    MemoryScope.PERSONAL,
    MemoryScope.GROUP,
    MemoryScope.CONVERSATION,
}
ADMIN_SCOPE_OWNER_FIELDS = {
    MemoryScope.PERSONAL: "owner_user_id",
    MemoryScope.GROUP: "owner_session_id",
    MemoryScope.CONVERSATION: "umo",
}


def _format_admin_removal_preview(
    uri: str,
    scope: str,
    owner_id: str,
    preview: dict[str, Any],
    confirmation_code: str,
    owner_field: str = "",
) -> str:
    """渲染管理员记忆删除预览，不输出完整正文。"""
    records = preview.get("records", [])
    lines = [
        "[Administrator memory removal confirmation]",
        f"URI: {uri}",
        f"Scope: {scope}",
        f"Owner: {owner_id or 'global'}",
        f"Matching active records: {preview.get('count', 0)}",
    ]
    for index, record in enumerate(records[:10], start=1):
        lines.append(
            f"{index}. len={record.get('content_len', 0)}, "
            f"sha256={record.get('content_sha256', '')}, "
            f"preview={record.get('preview', '')}"
        )
    if len(records) > 10:
        lines.append(f"... {len(records) - 10} more matching records")
    target_args = f"uri='{uri}', scope='{scope}'"
    if owner_field:
        target_args += f", owner_id='{owner_id}'"
    lines.append(
        "No deletion has occurred. To confirm the exact current records, call "
        f"memory_remove_admin({target_args}, confirm='{confirmation_code}')."
    )
    return "\n".join(lines)


def _render_cmd_prefix(prefix: str) -> str:
    return prefix or "/"


def _parse_command_args(event: AstrMessageEvent, full_cmd: str) -> str:
    """从 event.message_str 提取命令名之后的原始参数文本

    AstrBot 在进入 handler 前已剥离 wake prefix（如 /），
    因此 event.message_str 实际格式为 "memory list 1" 而非 "/memory list 1"。
    内部先做与 AstrBot 一致的空白规范化再截断命令名。
    """
    msg = re.sub(r"\s+", " ", event.message_str.strip())
    if msg.startswith(full_cmd):
        remainder = msg[len(full_cmd) :].strip()
        return remainder
    return msg


def _parse_memory_flags(args_text: str) -> dict[str, Any]:
    """解析 --all / --user <id> / --to <name> / --clear-cache 标志

    Returns:
        {"all": bool, "user": str, "to": str, "clear_cache": bool,
         "positional": str, "user_missing_value": bool,
         "to_missing_value": bool, "unknown_flags": list[str]}
    """
    result: dict[str, Any] = {
        "all": False,
        "user": "",
        "to": "",
        "clear_cache": False,
        "positional": "",
        "user_missing_value": False,
        "to_missing_value": False,
        "parse_error": "",
        "unknown_flags": [],
    }
    try:
        tokens = shlex.split(args_text)
    except ValueError as e:
        result["parse_error"] = str(e)
        return result
    i = 0
    positional_parts = []
    while i < len(tokens):
        token = tokens[i]
        if token == "--all":
            result["all"] = True
        elif token == "--user":
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                i += 1
                result["user"] = tokens[i]
            else:
                result["user_missing_value"] = True
        elif token == "--to":
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                i += 1
                result["to"] = tokens[i]
            else:
                result["to_missing_value"] = True
        elif token == "--clear-cache":
            result["clear_cache"] = True
        elif token.startswith("--"):
            result["unknown_flags"].append(token)
        else:
            positional_parts.append(token)
        i += 1
    result["positional"] = " ".join(positional_parts).strip()
    return result


def _ensure_initialized(memory_mgr: MemoryManager | None) -> str | None:
    """检查记忆管理器是否就绪，返回错误消息或 None"""
    if not memory_mgr:
        return "长期记忆插件未正确初始化，请检查配置"
    return None


def _get_confirmation_waiters(plugin: Any) -> set[SessionWaiter]:
    waiters = getattr(plugin, "_confirmation_waiters", None)
    if waiters is None:
        waiters = set()
        setattr(plugin, "_confirmation_waiters", waiters)
    return waiters


_SUBAGENT_RESULTS_REMINDER_RE = re.compile(
    r"<system_reminder>\s*\[Subagent results\][\s\S]*?</system_reminder>",
    re.IGNORECASE,
)
_ASTRBOT_INTERRUPTION_REQUEST = "Stop output."
_ASTRBOT_INTERRUPTION_MESSAGE = "Output stopped."


def _is_turn_continuation(message: Any) -> bool:
    """判断消息是否带有 AstrBot Steering 续轮标记（兼容对象与字典）。"""
    if isinstance(message, dict):
        return bool(message.get("_turn_continuation", False))
    return bool(getattr(message, "_turn_continuation", False))


def _sanitize_turn_user_text(text: str) -> str:
    """去除框架通过 Steering 注入的系统提醒与中断控制消息，保留真实用户输入。"""
    cleaned = _SUBAGENT_RESULTS_REMINDER_RE.sub("", text).strip()
    if not cleaned or cleaned == _ASTRBOT_INTERRUPTION_REQUEST:
        return ""
    return cleaned


_MAX_UNWEAKREFABLE_EVENTS = 512


def _get_agent_tracking(
    plugin: Any,
) -> tuple[
    weakref.WeakSet[Any],
    list[Any],
    weakref.WeakKeyDictionary[Any, dict[str, Any]],
    list[tuple[Any, dict[str, Any]]],
    weakref.WeakKeyDictionary[Any, dict[str, str]],
    list[tuple[Any, dict[str, str]]],
]:
    managed_events = getattr(plugin, "_agent_managed_events", None)
    if managed_events is None:
        managed_events = weakref.WeakSet()
        setattr(plugin, "_agent_managed_events", managed_events)
    managed_fallback = getattr(plugin, "_agent_managed_events_fallback", None)
    if managed_fallback is None:
        managed_fallback = []
        setattr(plugin, "_agent_managed_events_fallback", managed_fallback)
    run_states = getattr(plugin, "_agent_run_states", None)
    if run_states is None:
        run_states = weakref.WeakKeyDictionary()
        setattr(plugin, "_agent_run_states", run_states)
    run_states_fallback = getattr(plugin, "_agent_run_states_fallback", None)
    if run_states_fallback is None:
        run_states_fallback = []
        setattr(plugin, "_agent_run_states_fallback", run_states_fallback)
    pending_snapshots = getattr(plugin, "_pending_event_snapshots", None)
    if pending_snapshots is None:
        pending_snapshots = weakref.WeakKeyDictionary()
        setattr(plugin, "_pending_event_snapshots", pending_snapshots)
    pending_fallback = getattr(plugin, "_pending_event_snapshots_fallback", None)
    if pending_fallback is None:
        pending_fallback = []
        setattr(plugin, "_pending_event_snapshots_fallback", pending_fallback)
    return (
        managed_events,
        managed_fallback,
        run_states,
        run_states_fallback,
        pending_snapshots,
        pending_fallback,
    )


def _mark_agent_managed_event(plugin: Any, event: Any) -> None:
    managed_events, managed_fallback, _, _, _, _ = _get_agent_tracking(plugin)
    try:
        managed_events.add(event)
    except TypeError:
        if not any(item is event for item in managed_fallback):
            managed_fallback.append(event)
            if len(managed_fallback) > _MAX_UNWEAKREFABLE_EVENTS:
                del managed_fallback[:-_MAX_UNWEAKREFABLE_EVENTS]


def _is_agent_managed_event(plugin: Any, event: Any) -> bool:
    managed_events, managed_fallback, _, _, _, _ = _get_agent_tracking(plugin)
    try:
        if event in managed_events:
            return True
    except TypeError:
        pass
    return any(item is event for item in managed_fallback)


def _set_agent_run_state(plugin: Any, event: Any, state: dict[str, Any]) -> None:
    _, _, run_states, run_states_fallback, _, _ = _get_agent_tracking(plugin)
    try:
        run_states[event] = state
    except TypeError:
        for idx, (existing, _) in enumerate(run_states_fallback):
            if existing is event:
                run_states_fallback[idx] = (event, state)
                return
        run_states_fallback.append((event, state))
        if len(run_states_fallback) > _MAX_UNWEAKREFABLE_EVENTS:
            del run_states_fallback[:-_MAX_UNWEAKREFABLE_EVENTS]


def _pop_agent_run_state(plugin: Any, event: Any) -> dict[str, Any] | None:
    _, _, run_states, run_states_fallback, _, _ = _get_agent_tracking(plugin)
    try:
        state = run_states.pop(event, None)
        if state is not None:
            return state
    except TypeError:
        pass
    for idx, (existing, state) in enumerate(run_states_fallback):
        if existing is event:
            run_states_fallback.pop(idx)
            return state
    return None


def _set_pending_event_snapshot(
    plugin: Any, event: Any, pending: dict[str, str]
) -> None:
    _, _, _, _, pending_snapshots, pending_fallback = _get_agent_tracking(plugin)
    try:
        pending_snapshots[event] = pending
    except TypeError:
        for idx, (existing, _) in enumerate(pending_fallback):
            if existing is event:
                pending_fallback[idx] = (event, pending)
                return
        pending_fallback.append((event, pending))
        if len(pending_fallback) > _MAX_UNWEAKREFABLE_EVENTS:
            del pending_fallback[:-_MAX_UNWEAKREFABLE_EVENTS]


def _pop_pending_event_snapshot(plugin: Any, event: Any) -> dict[str, str] | None:
    _, _, _, _, pending_snapshots, pending_fallback = _get_agent_tracking(plugin)
    try:
        pending = pending_snapshots.pop(event, None)
        if pending is not None:
            return pending
    except TypeError:
        pass
    for idx, (existing, pending) in enumerate(pending_fallback):
        if existing is event:
            pending_fallback.pop(idx)
            return pending
    return None


def _make_event_ref(event: Any) -> Any:
    try:
        return weakref.ref(event)
    except TypeError:
        return event


def _resolve_event_ref(ref: Any) -> Any:
    if isinstance(ref, weakref.ReferenceType):
        return ref()
    return ref


def _collect_turn_messages(
    run_context: Any, run_state: dict[str, Any] | None
) -> list[tuple[str, str]]:
    """按时间顺序提取本轮新增的真实用户追问与可见助手回复。"""
    raw_messages = getattr(run_context, "messages", None)
    if not isinstance(raw_messages, list) or not raw_messages:
        return []

    new_messages: list[Any] | None = None
    if run_state:
        anchor = run_state.get("anchor")
        start_len = int(run_state.get("start_len", 0) or 0)
        if anchor is not None:
            for idx in range(len(raw_messages) - 1, -1, -1):
                if raw_messages[idx] is anchor:
                    new_messages = raw_messages[idx + 1 :]
                    break
        elif start_len <= len(raw_messages):
            new_messages = raw_messages[start_len:]

    if new_messages is None:
        # 轮内发生上下文压缩导致起始锚点变化时，从最后一个非 Steering 用户消息之后恢复本轮消息。
        last_turn_start = -1
        for idx in range(len(raw_messages) - 1, -1, -1):
            msg = raw_messages[idx]
            if _message_role(msg) == "user" and not _is_turn_continuation(msg):
                last_turn_start = idx
                break
        new_messages = (
            raw_messages[last_turn_start + 1 :] if last_turn_start >= 0 else []
        )

    end_reason = getattr(run_context, "run_end_reason", None)
    turn_messages: list[tuple[str, str]] = []
    for msg in new_messages:
        role = _message_role(msg)
        if role not in ("user", "assistant"):
            continue
        text = _extract_visible_message_text(msg)
        if not text:
            continue
        if role == "user":
            text = _sanitize_turn_user_text(text)
            if not text:
                continue
        elif end_reason == "cancelled" and text == _ASTRBOT_INTERRUPTION_MESSAGE:
            continue
        turn_messages.append((role, text))
    return turn_messages


def _is_inbox_command_turn(message_str: str) -> bool:
    """命令轮次不入缓冲表；核心会剥离唤醒前缀，两种形式等价处理。"""
    text = str(message_str or "").strip()
    if not text:
        return False
    if text.startswith("/"):
        return True
    first_token = text.split(None, 1)[0].strip("/")
    return first_token in INBOX_COMMAND_TOKENS


def _is_inbox_command_event(event: Any) -> bool:
    """核心消费过唤醒前缀或原生标记为命令的轮次视为命令调用。

    唤醒检查阶段消费斜杠前缀后 message_str 不再带 "/"，需要靠
    _consumed_wake_prefix 标记识别；Discord 等平台另有原生 is_slash_command。
    """
    is_slash_command = getattr(event, "is_slash_command", None)
    if callable(is_slash_command):
        try:
            if is_slash_command():
                return True
        except Exception:
            pass
    try:
        return bool(event.get_extra("_consumed_wake_prefix"))
    except Exception:
        return False


def _build_inbox_user_text(
    run_state: dict[str, Any] | None, turn_messages: list[tuple[str, str]]
) -> str:
    """初始用户文本 + Steering 追问按时间顺序拼接。"""
    parts: list[str] = []
    initial = str((run_state or {}).get("message_str", "") or "").strip()
    if initial:
        parts.append(initial)
    parts.extend(text for role, text in turn_messages or [] if role == "user")
    return "\n".join(parts)


def authorize_memory_mutation(
    event: AstrMessageEvent,
    *,
    target_user_id: str | None = None,
    all_users: bool = False,
) -> str | None:
    """校验记忆修改目标，普通用户只能操作自己的记忆。"""
    if all_users and not event.is_admin():
        return "该操作需要管理员权限"
    if event.is_admin() or not target_user_id:
        return None
    current_user_id = build_user_id(event.get_platform_id(), event.get_sender_id())
    if target_user_id != current_user_id:
        return "普通用户只能修改自己的记忆"
    return None


def _validate_command(
    event: AstrMessageEvent,
    args: dict[str, Any],
    *,
    cmd_name: str,
    require_admin: bool = False,
    allow_all: bool = False,
    allow_user: bool = False,
    allow_to: bool = False,
    allow_clear_cache: bool = False,
    allow_positional: bool = True,
) -> str | None:
    """统一的命令参数校验，返回首个错误消息或 None。

    校验顺序与原各命令保持一致：未知 flag → user 缺值 → to 缺值 →
    各 flag 是否被允许 → positional → 管理员权限 → --all 管理员权限。
    """
    if args["unknown_flags"]:
        return f"未知参数: {', '.join(args['unknown_flags'])}"
    if args["parse_error"]:
        return f"参数解析失败: {args['parse_error']}"
    if args["user_missing_value"]:
        return "--user 需要指定用户 ID"
    if args["to_missing_value"]:
        if not allow_to:
            return f"{cmd_name} 命令不支持 --to 参数"
        return "需要指定知识库名称，用法: /memory rebuild --to <知识库名>"
    if not allow_user and args["user"]:
        return f"{cmd_name} 命令不支持 --user 参数"
    if not allow_to and args["to"]:
        return f"{cmd_name} 命令不支持 --to 参数"
    if not allow_clear_cache and args["clear_cache"]:
        return f"{cmd_name} 命令不支持 --clear-cache 参数"
    if not allow_all and args["all"]:
        return f"{cmd_name} 命令不支持 --all 参数"
    if not allow_positional and args["positional"]:
        return f"未知参数: {args['positional']}"
    if require_admin and not event.is_admin():
        return "该操作需要管理员权限"
    if args["all"] and not event.is_admin():
        return "--all 标志需要管理员权限"
    return None


REVIEW_VIEW_NAME = "memory_review"
"""ChatUI 审核卡片视图名，入口为 views/memory_review/index.html。"""

REVIEW_CARD_SUMMARY_MAX_CHARS = 120
"""卡片与文本回退中单条待审摘要的截断长度。"""

_UI_ACTION_FILTER = getattr(filter, "ui_action", None)
"""4.30 的 ui_action 装饰器；4.29 无此 API 时回退为不注册。"""


def _ui_action(action_id: str):
    """按宿主能力注册视图动作回调，低版本直接保留原函数。"""
    if _UI_ACTION_FILTER is not None:
        return _UI_ACTION_FILTER(action_id)
    return lambda awaitable: awaitable


def _chatui_view_supported(event: AstrMessageEvent) -> bool:
    """当前平台是否支持 ChatUI 卡片，且事件带 4.30 的 view_result 方法。"""
    return callable(getattr(event, "view_result", None)) and bool(
        getattr(event.platform_meta, "support_ui_interaction", False)
    )


def _review_item_summary(op: dict[str, Any]) -> str:
    """从待审操作提取一句安全摘要，覆盖 create/update/merge/link 等操作字段。"""
    for key in ("content", "new_content", "merged_content"):
        value = str(op.get(key) or "").strip()
        if value:
            return value[:REVIEW_CARD_SUMMARY_MAX_CHARS]
    parts = [
        str(op.get(key) or "").strip()
        for key in ("uri", "old_uri", "new_uri", "source", "target")
    ]
    uris = op.get("uris")
    if isinstance(uris, list):
        parts.extend(str(uri).strip() for uri in uris if str(uri).strip())
    parts.append(str(op.get("reason") or "").strip()[:60])
    return " ".join(part for part in parts if part)[:REVIEW_CARD_SUMMARY_MAX_CHARS]


def _review_card_data(
    pending: list[dict[str, Any]],
    *,
    notice: str = "",
    error: str = "",
) -> dict[str, Any]:
    """构造审核卡片数据：待审条目列表加可选反馈信息。"""
    data: dict[str, Any] = {
        "items": [
            {
                "id": item.get("id", 0),
                "op_type": str(item.get("op_type", "")),
                "reason": str(item.get("verdict_reason", "")),
                "summary": _review_item_summary(item.get("op") or {}),
                "created_at": str(item.get("created_at", "")),
            }
            for item in pending
        ]
    }
    if notice:
        data["notice"] = notice
    if error:
        data["error"] = error
    return data


class MemoryPlugin(Star):
    """长期记忆插件"""

    # 请求快照过期时间（秒）
    SNAPSHOT_TTL = 300  # 5 分钟
    SNAPSHOT_CLEANUP_INTERVAL = 60
    TTL_EXPIRE_INTERVAL = 3600  # TTL 过期扫描间隔（秒）
    CONSOLIDATION_INTERVAL = 7200  # 记忆巩固扫描间隔（秒）
    CONTACT_WRITE_INTERVAL = 60
    CONTACT_WRITE_CACHE_MAX = 10000
    MAX_SESSION_SNAPSHOTS = DEFAULT_MAX_SESSION_SNAPSHOTS
    MAX_SNAPSHOT_CHARS = DEFAULT_MAX_SNAPSHOT_CHARS
    EXTRACTION_ASSISTANT_MAX_CHARS = 300  # 实时提取对话中助手回复的截断长度

    def __init__(self, context: Context, config=None):
        super().__init__(context, config)
        raw_config = config or {}
        migrate_maintenance_config(raw_config)
        # 迁移后 self.config 即单一平面视图，卡片/命令/后台与 MemoryManager 读取同一来源
        self.config = build_flat_config_view(raw_config)
        self._flat_config = self.config
        # 命令与审核卡片的待审裁决互斥锁，覆盖读队列→执行→写回
        self._review_lock = asyncio.Lock()
        self.memory_mgr: MemoryManager | None = None
        # 实例级请求快照字典，按 session_key 键控，避免跨用户污染
        # 结构: {session_key: {"snapshots": [...], "timestamp": float}}
        # snapshots 是一个列表，累积多轮对话
        self._request_snapshots: dict[str, dict[str, Any]] = {}
        # 每个会话的对话计数器
        self._session_counters: dict[str, int] = {}
        self._contact_write_times: dict[tuple[str, ...], float] = {}
        self._last_snapshot_cleanup = 0.0
        self._last_ttl_expire = 0.0
        self._last_consolidation = 0.0
        self._background_tasks: set[asyncio.Task] = set()
        self._confirmation_waiters: set[SessionWaiter] = set()
        self._agent_managed_events: weakref.WeakSet[Any] = weakref.WeakSet()
        self._agent_managed_events_fallback: list[Any] = []
        self._agent_run_states: weakref.WeakKeyDictionary[Any, dict[str, Any]] = (
            weakref.WeakKeyDictionary()
        )
        self._agent_run_states_fallback: list[tuple[Any, dict[str, Any]]] = []
        self._pending_event_snapshots: weakref.WeakKeyDictionary[
            Any, dict[str, str]
        ] = weakref.WeakKeyDictionary()
        self._pending_event_snapshots_fallback: list[tuple[Any, dict[str, str]]] = []
        self._scheduler: MaintenanceScheduler | None = None
        self._memory_gate: MemoryGate | None = None

    async def initialize(self):
        """插件初始化：校验配置，并尝试立即连接 KB（重载场景）"""
        try:
            self.memory_mgr = MemoryManager(
                kb_mgr=self.context.kb_manager,
                config=self.config,
                kv_put=self.put_kv_data,
                kv_get=self.get_kv_data,
                kv_delete=self.delete_kv_data,
            )
            self.memory_mgr.initialize()
        except Exception as e:
            logger.error(f"[简单长期记忆] 配置校验失败: {e}")
            self.memory_mgr = None
            return

        if (
            self.config.get("maintenance_enabled", False)
            and not maintenance_supported()
        ):
            logger.warning(
                "[简单长期记忆] %s，已跳过后台整理调度；基础记忆读写与召回不受影响",
                unsupported_maintenance_reason(),
            )
        # 尝试立即连接 KB（热重载时 KB 已就绪）
        try:
            await self.memory_mgr.connect_kb()
            logger.info("[简单长期记忆] 插件初始化成功")
            await self._start_scheduler()
        except Exception:
            # 首次启动时 KB 尚未就绪，由 on_astrbot_loaded 钩子处理
            logger.info("[简单长期记忆] 配置校验通过，等待知识库就绪")

    @filter.on_astrbot_loaded()
    async def on_loaded(self):
        if not self.memory_mgr or self.memory_mgr.is_kb_connected:
            # KB 已连接（热重载场景），但仍需检查中断恢复
            if self.memory_mgr:
                await self._recover_interrupted_rebuild()
                await self.memory_mgr.cleanup_orphan_links()
            return
        connected = False
        try:
            await self.memory_mgr.connect_kb()
            connected = True
            logger.info("[简单长期记忆] 插件初始化成功")
        except Exception as e:
            logger.error(f"[简单长期记忆] 连接知识库失败，将先尝试恢复重建缓存: {e}")

        # 检测上次中断的重建：主快照可按缓存中的目标 KB ID 恢复，
        # 不强依赖当前配置的源 KB 仍然可连接。
        await self._recover_interrupted_rebuild()

        if not connected:
            self.memory_mgr = None
            return

        await self.memory_mgr.cleanup_orphan_links()
        await self._start_scheduler()

    async def _recover_interrupted_rebuild(self) -> None:
        """启动时检测并恢复上次中断的重建

        恢复优先级：
        1. 主数据快照 (rebuild_memory_records) — 从中断点继续重建
        2. 缓冲写入 (rebuild_pending_writes) — flush 未处理的写入
        无论主数据快照是否恢复成功，都继续处理缓冲写入。
        """
        if not self.memory_mgr:
            return

        # 优先恢复主数据快照（更严重的崩溃场景）
        try:
            rebuild_status = await self.get_kv_data("rebuild_status", None)
            memory_records = await self.get_kv_data("rebuild_memory_records", None)
            rebuild_context = await self.get_kv_data("rebuild_context", None)

            if (
                rebuild_status == "in_progress"
                and not memory_records
                and not self.memory_mgr.is_rebuilding
            ):
                await self.put_kv_data("rebuild_status", "interrupted")
                rebuild_status = "interrupted"

            if (
                rebuild_status in {"in_progress", "interrupted"}
                and memory_records
                and isinstance(memory_records, list)
            ):
                logger.warning(
                    f"[简单长期记忆] 检测到未完成的重建，"
                    f"状态: {rebuild_status}, "
                    f"主数据快照 {len(memory_records)} 条，正在恢复..."
                )
                # 从 KV 快照继续重建（不重新拉取源 KB，因为可能已被清空）
                recovery_result = await self.memory_mgr._resume_rebuild_from_snapshot(
                    memory_records,
                    rebuild_context if isinstance(rebuild_context, dict) else None,
                )
                recovered = recovery_result["success"]
                remaining_records = recovery_result["remaining_records"]

                if recovered:
                    logger.info(f"[简单长期记忆] 主数据快照恢复完成: {recovered} 条")

                if remaining_records:
                    logger.warning(
                        f"[简单长期记忆] 仍有 {len(remaining_records)} 条快照"
                        "未恢复，已保留到 KV，等待下次继续恢复"
                    )
                    await self.put_kv_data("rebuild_memory_records", remaining_records)
                    await self.put_kv_data("rebuild_status", "interrupted")
                else:
                    await self.delete_kv_data("rebuild_memory_records")
                    await self.put_kv_data("rebuild_status", "completed")
        except Exception as e:
            logger.warning(f"[简单长期记忆] 恢复主数据快照失败: {e}")
            try:
                await self.put_kv_data("rebuild_status", "interrupted")
            except Exception:
                pass

        # 无论主数据快照是否恢复成功，都继续恢复缓冲写入
        try:
            pending = await self.get_kv_data("rebuild_pending_writes", None)
            if not pending or not isinstance(pending, list):
                return
            self.memory_mgr.load_pending_writes(pending)
            flushed = await self.memory_mgr._flush_pending_writes()
            if flushed:
                logger.info(
                    f"[简单长期记忆] 已恢复上次中断的缓冲写入: "
                    f"{len(pending)} 条中写入 {flushed} 条"
                )
        except Exception as e:
            logger.warning(f"[简单长期记忆] 恢复缓冲写入失败: {e}")

    async def terminate(self):
        """插件销毁"""
        for waiter in tuple(self._confirmation_waiters):
            waiter.session_controller.stop()
        self._confirmation_waiters.clear()
        if self._scheduler:
            await self._scheduler.stop()
            self._scheduler = None
        self._request_snapshots.clear()
        self._session_counters.clear()
        self._agent_managed_events.clear()
        self._agent_managed_events_fallback.clear()
        self._agent_run_states.clear()
        self._agent_run_states_fallback.clear()
        self._pending_event_snapshots.clear()
        self._pending_event_snapshots_fallback.clear()
        logger.info("[简单长期记忆] 插件已卸载")

    async def _start_scheduler(self) -> None:
        """KB 连接后启动后台整理调度器。"""
        if not self.memory_mgr or not self.memory_mgr.is_kb_connected:
            return
        if self._scheduler:
            return  # 已启动
        if (
            self.config.get("maintenance_enabled", False)
            and not maintenance_supported()
        ):
            # maintenance_enabled 仅允许 AstrBot 4.30+；版本不足时不创建调度
            logger.warning(
                "[简单长期记忆] %s，本次不创建后台整理调度器",
                unsupported_maintenance_reason(),
            )
            return
        self._scheduler = MaintenanceScheduler(
            context=self.context,
            memory_mgr=self.memory_mgr,
            config=self.config,
            kv_put=self.put_kv_data,
            kv_get=self.get_kv_data,
        )
        await self._scheduler.start()

    def _get_display_timezone(self, umo: str | None = None) -> str | None:
        """读取 AstrBot 全局或会话时区，用于注入内容的时间展示。"""
        try:
            cfg = _read_astrbot_config(self.context, umo)
            if isinstance(cfg, dict):
                return cfg.get("timezone") or None
        except Exception:
            pass
        return None

    def _get_viewer_user_id(self, event: AstrMessageEvent) -> str | None:
        """获取当前会话用户 ID，用于标注共享记忆归属。"""
        if not self.memory_mgr:
            return None
        try:
            return self.memory_mgr._current_owner_user_id(event)
        except Exception:
            return None

    def _get_cmd_prefix(self, umo: str | None = None) -> str:
        """从 AstrBot 配置读取命令前缀，默认 /"""
        try:
            cfg = _read_astrbot_config(self.context, umo)
            prefixes = cfg.get("wake_prefix", []) if isinstance(cfg, dict) else []
            if prefixes and isinstance(prefixes, list):
                return prefixes[0]
        except Exception:
            pass
        return "/"

    async def _get_llm_provider_id(
        self, event: AstrMessageEvent, provider_type: str
    ) -> str | None:
        """获取 LLM Provider ID

        优先使用配置中指定的 Provider，否则使用会话主 LLM

        Args:
            event: 消息事件
            provider_type: 'extraction' 或 'summarization'

        Returns:
            Provider ID 或 None
        """
        config_key = f"{provider_type}_provider_id"
        provider_id = self.config.get(config_key, "")

        if provider_id:
            return provider_id

        # 使用会话主 LLM
        try:
            return await self.context.get_current_chat_provider_id(
                event.unified_msg_origin
            )
        except Exception:
            return None

    # ==================== 请求快照管理 ====================

    def _get_session_key(self, event: AstrMessageEvent) -> str:
        """获取会话唯一标识，用于请求-响应关联

        使用 unified_msg_origin 作为会话键，确保同一会话的请求和响应正确匹配
        """
        return event.unified_msg_origin

    def _accumulate_request_snapshot(
        self, event: AstrMessageEvent, request: ProviderRequest
    ) -> None:
        """累积请求快照（在请求阶段调用）"""
        if _read_positive_int(self.config.get("extraction_interval", 20), 20) <= 0:
            self._discard_session_snapshots(event)
            return

        session_key = self._get_session_key(event)
        current_time = time.time()

        if session_key not in self._request_snapshots:
            self._request_snapshots[session_key] = {
                "snapshots": [],
                "pending_prompt": None,
                "timestamp": current_time,
            }

        prompt_text = (request.prompt or "")[: self.MAX_SNAPSHOT_CHARS]
        sender_id = event.get_sender_id()

        # 按事件保存待匹配请求，避免同会话重叠轮次互相覆盖或误清
        _set_pending_event_snapshot(
            self,
            event,
            {
                "prompt": prompt_text,
                "sender_id": sender_id,
            },
        )
        self._request_snapshots[session_key]["pending_prompt"] = prompt_text
        self._request_snapshots[session_key]["pending_sender_id"] = sender_id
        self._request_snapshots[session_key]["pending_event_ref"] = _make_event_ref(
            event
        )
        self._request_snapshots[session_key]["timestamp"] = current_time

        self._cleanup_expired_snapshots(current_time)

    def _complete_snapshot_with_response(
        self,
        event: AstrMessageEvent,
        response_text: str,
        *,
        turn_messages: list[tuple[str, str]] | None = None,
    ) -> None:
        """用响应完成快照（在响应阶段调用）"""
        session_key = self._get_session_key(event)
        entry = self._request_snapshots.get(session_key)
        pending = _pop_pending_event_snapshot(self, event)

        if not entry:
            return

        pending_owner = _resolve_event_ref(entry.get("pending_event_ref"))
        if pending is not None:
            prompt_text = pending.get("prompt") or ""
            sender_id = pending.get("sender_id", event.get_sender_id())
            if pending_owner is None or pending_owner is event:
                entry["pending_prompt"] = None
                entry["pending_sender_id"] = ""
                entry["pending_event_ref"] = None
        else:
            if pending_owner is not None and pending_owner is not event:
                return
            prompt_text = entry.get("pending_prompt") or ""
            sender_id = entry.get("pending_sender_id", event.get_sender_id())
            entry["pending_prompt"] = None
            entry["pending_sender_id"] = ""
            entry["pending_event_ref"] = None

        if not prompt_text:
            return

        snapshot: dict[str, Any] = {
            "prompt": prompt_text,
            "response": response_text[: self.MAX_SNAPSHOT_CHARS],
            "sender_id": sender_id,
        }

        if turn_messages:
            parsed_umo = UMOInfo.parse(event.unified_msg_origin)
            keep_steered_sender = (
                parsed_umo.is_valid and parsed_umo.session_type == "private"
            )
            ordered_messages: list[dict[str, str]] = [
                {
                    "role": "user",
                    "text": prompt_text,
                    "sender_id": str(sender_id or ""),
                }
            ]
            remaining_user = max(0, self.MAX_SNAPSHOT_CHARS - len(prompt_text))
            remaining_assistant = self.MAX_SNAPSHOT_CHARS
            for role, text in turn_messages:
                if role == "user":
                    if remaining_user <= 0:
                        continue
                    clipped = text[:remaining_user]
                    remaining_user -= len(clipped)
                    ordered_messages.append(
                        {
                            "role": "user",
                            "text": clipped,
                            "sender_id": (
                                str(sender_id or "") if keep_steered_sender else ""
                            ),
                        }
                    )
                elif role == "assistant":
                    if remaining_assistant <= 0:
                        continue
                    clipped = text[:remaining_assistant]
                    remaining_assistant -= len(clipped)
                    ordered_messages.append({"role": "assistant", "text": clipped})
            snapshot["messages"] = ordered_messages

        entry["snapshots"].append(snapshot)
        if len(entry["snapshots"]) > self.MAX_SESSION_SNAPSHOTS:
            del entry["snapshots"][: -self.MAX_SESSION_SNAPSHOTS]

    def _get_session_snapshot_count(self, event: AstrMessageEvent) -> int:
        """获取会话的快照数量"""
        session_key = self._get_session_key(event)
        entry = self._request_snapshots.get(session_key)
        if not entry:
            return 0
        return len(entry.get("snapshots", []))

    def _get_and_clear_session_snapshots(
        self, event: AstrMessageEvent
    ) -> list[dict[str, Any]]:
        """获取并清空会话的快照列表"""
        session_key = self._get_session_key(event)
        entry = self._request_snapshots.get(session_key)
        if not entry:
            return []

        snapshots = entry.get("snapshots", [])
        # 清空快照列表但保留会话条目
        entry["snapshots"] = []
        return snapshots

    def _discard_session_snapshots(self, event: AstrMessageEvent) -> None:
        """清理当前会话快照和计数器"""
        _pop_pending_event_snapshot(self, event)
        session_key = self._get_session_key(event)
        self._request_snapshots.pop(session_key, None)
        self._session_counters.pop(session_key, None)

    def _increment_session_counter(self, event: AstrMessageEvent) -> int:
        """递增会话对话计数器并返回当前值"""
        key = self._get_session_key(event)
        self._session_counters[key] = self._session_counters.get(key, 0) + 1
        return self._session_counters[key]

    async def _maybe_expire_stale_memories(self) -> None:
        """节流触发 TTL 过期（P1.1）。仿 _cleanup_expired_snapshots 的间隔模式。"""
        now = time.time()
        if now - self._last_ttl_expire < self.TTL_EXPIRE_INTERVAL:
            return
        self._last_ttl_expire = now
        mgr = self.memory_mgr
        if not mgr or not mgr.is_kb_connected:
            return
        ttl_days = _read_positive_int(self.config.get("memory_ttl_days", 30), 30)
        if ttl_days <= 0:
            return
        try:
            expired = await mgr.expire_stale_memories(ttl_days)
            if expired:
                logger.info(f"[简单长期记忆] TTL 过期标记 {expired} 条记忆")
        except Exception as e:
            logger.debug(f"[简单长期记忆] TTL 过期任务失败: {e}")

    async def _maybe_consolidate_memories(self, event: AstrMessageEvent) -> None:
        """节流触发记忆巩固（P1.2）：低频老记忆 → LLM 总结 → 摘要记忆 + 原文标记。"""
        now = time.time()
        if now - self._last_consolidation < self.CONSOLIDATION_INTERVAL:
            return
        self._last_consolidation = now
        mgr = self.memory_mgr
        if not mgr or not mgr.is_kb_connected:
            return
        if not self.config.get("memory_consolidation_enabled", True):
            return
        # 后台整理启用时禁用旧巩固，避免并发竞态（spec: 旧 consolidation 迁移）
        if self.config.get("maintenance_enabled", False):
            return
        min_age_days = _read_positive_int(
            self.config.get("consolidation_min_age_days", 14), 14
        )
        max_recall = _read_positive_int(
            self.config.get("consolidation_max_recall", 1), 1
        )
        limit = _read_positive_int(self.config.get("consolidation_batch_size", 30), 30)
        try:
            candidates = await mgr.fetch_consolidation_candidates(
                event, min_age_days, max_recall, limit
            )
        except Exception as e:
            logger.debug(f"[简单长期记忆] 读取巩固候选失败: {e}")
            return
        if not candidates:
            return
        provider_id = await self._get_llm_provider_id(event, "summarization")
        if not provider_id:
            provider_id = await self._get_llm_provider_id(event, "extraction")
        if not provider_id:
            return
        joined = "\n".join(
            f"- {c.get('text', '').strip()}" for c in candidates if c.get("text")
        )
        if not joined.strip():
            return
        try:
            prompt = MEMORY_CONSOLIDATION_PROMPT.substitute(memories=joined)
            llm_response = await self.context.llm_generate(
                chat_provider_id=provider_id, prompt=prompt
            )
            summary = (getattr(llm_response, "completion_text", "") or "").strip()
        except Exception as e:
            logger.debug(f"[简单长期记忆] 巩固 LLM 调用失败: {e}")
            return
        if not summary:
            return
        source_uris = [
            c.get("metadata", {}).get("uri")
            for c in candidates
            if c.get("metadata", {}).get("uri")
        ]
        # 先写摘要再标记源（避免 store 失败后源已 deprecated 但无替代）
        try:
            await mgr.store_memory(
                event=event,
                content=summary,
                domain="context",
                memory_type=MemoryType.NORMAL,
                disclosure="由历史低频记忆巩固而来",
                importance=4,
                memory_scope=MemoryScope.PERSONAL,
            )
        except Exception as e:
            logger.warning(f"[简单长期记忆] 巩固摘要写入失败，保留原文: {e}")
            return
        marked = await mgr.mark_consolidated(source_uris)
        if marked <= 0:
            logger.debug("[简单长期记忆] 巩固：原文标记 0 条（摘要已写入）")
        logger.info(
            f"[简单长期记忆] 巩固 {len(candidates)} 条 → 1 条摘要，标记原文 {marked} 条"
        )

    def _cleanup_expired_snapshots(self, current_time: float | None = None) -> None:
        """清理过期的请求快照"""
        current_time = current_time or time.time()
        if current_time - self._last_snapshot_cleanup < self.SNAPSHOT_CLEANUP_INTERVAL:
            return
        self._last_snapshot_cleanup = current_time

        expired_keys = [
            key
            for key, entry in self._request_snapshots.items()
            if current_time - entry.get("timestamp", 0) > self.SNAPSHOT_TTL
        ]
        for key in expired_keys:
            del self._request_snapshots[key]
            self._session_counters.pop(key, None)
        for key in list(self._session_counters):
            if key not in self._request_snapshots:
                self._session_counters.pop(key, None)

    # ==================== JSON 解析辅助 ====================

    def _strip_json_fence(self, text: str) -> str:
        """移除 markdown JSON 围栏"""
        return _strip_json_fence_text(text)

    def _parse_extracted_memories(
        self, text: str, session_type: str = "private"
    ) -> list[dict[str, Any]]:
        """解析 LLM 返回的记忆 JSON，带校验和上限"""
        return _shared_parse_extracted_memories(text, session_type)

    def _build_conversation_from_snapshots(
        self, snapshots: list[dict[str, Any]]
    ) -> str:
        """从快照列表构建对话文本（用户/助手标签隔离，助手长回复截断）"""
        assistant_limit = self.EXTRACTION_ASSISTANT_MAX_CHARS

        def _clip_assistant(text: str) -> str:
            if len(text) <= assistant_limit:
                return text
            return text[:assistant_limit] + "…"

        lines = []
        for snapshot in snapshots:
            ordered_messages = snapshot.get("messages")
            if isinstance(ordered_messages, list) and ordered_messages:
                for item in ordered_messages:
                    role = item.get("role", "")
                    text = item.get("text", "")
                    if not text:
                        continue
                    if role == "user":
                        sid = item.get("sender_id", "")
                        sender_label = f"用户:{sid}" if sid else "用户"
                        lines.append(f"[{sender_label}]: {text}")
                    elif role == "assistant":
                        lines.append(f"[助手]: {_clip_assistant(text)}")
                continue
            prompt = snapshot.get("prompt", "")
            response = snapshot.get("response", "")
            sender_id = snapshot.get("sender_id", "")
            if prompt:
                sender_label = f"用户:{sender_id}" if sender_id else "用户"
                lines.append(f"[{sender_label}]: {prompt}")
            if response:
                lines.append(f"[助手]: {_clip_assistant(response)}")
        return "\n".join(lines)

    @staticmethod
    def _snapshot_user_text_length(snapshots: list[dict[str, Any]]) -> int:
        """统计快照中的有效用户文本长度（不含角色标签与助手输出）。"""
        total = 0
        for snapshot in snapshots:
            ordered_messages = snapshot.get("messages")
            if isinstance(ordered_messages, list) and ordered_messages:
                for item in ordered_messages:
                    if item.get("role") == "user":
                        total += len(item.get("text", "") or "")
                continue
            total += len(snapshot.get("prompt", "") or "")
        return total

    # ==================== 检索优化 ====================

    async def _optimize_recall_query(
        self, event: AstrMessageEvent, raw_query: str
    ) -> str:
        """调用 LLM 从对话上下文中提炼检索关键词"""
        provider_id = await self._get_llm_provider_id(event, "extraction")
        if not provider_id:
            return raw_query

        prompt = RECALL_QUERY_PROMPT.substitute(context=raw_query[:1000])
        try:
            timeout = _clamp_timeout(
                self.config.get(
                    "optimize_recall_query_timeout",
                    DEFAULT_RECALL_QUERY_OPTIMIZATION_TIMEOUT,
                )
            )
            llm_response = await asyncio.wait_for(
                self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=prompt,
                ),
                timeout=timeout,
            )
            result = getattr(llm_response, "completion_text", "") or ""
            result = self._strip_json_fence(result).strip()
            keywords = json.loads(result)
            if isinstance(keywords, list) and keywords:
                optimized = " ".join(str(k) for k in keywords[:5])
                logger.debug(f"[简单长期记忆] 检索优化: {optimized}")
                return optimized
        except asyncio.TimeoutError:
            logger.debug("[简单长期记忆] 检索优化超时，使用原始查询")
        except Exception as e:
            logger.debug(f"[简单长期记忆] 检索优化失败，使用原始查询: {e}")

        return raw_query

    # ==================== LLM 钩子 ====================

    async def _maybe_observe_contact(self, event: AstrMessageEvent) -> None:
        """节流记录当前发送者的接触事实。"""
        if not self.memory_mgr:
            return
        try:
            key = (
                str(event.get_platform_id() or ""),
                str(event.get_self_id() or ""),
                str(event.get_sender_id() or ""),
                event.unified_msg_origin,
            )
            now = time.monotonic()
            if (
                now - self._contact_write_times.get(key, 0.0)
                < self.CONTACT_WRITE_INTERVAL
            ):
                return
            if len(self._contact_write_times) >= self.CONTACT_WRITE_CACHE_MAX:
                self._contact_write_times.clear()
            if await self.memory_mgr.observe_contact(event):
                self._contact_write_times[key] = now
        except Exception as e:
            logger.debug(f"[简单长期记忆] 接触记录节流处理失败: {e}")

    @filter.on_llm_request()
    async def inject_memories(self, event: AstrMessageEvent, request: ProviderRequest):
        if not self.memory_mgr:
            return

        if not self.config.get("auto_memorize", True):
            return

        is_wakeup = _is_wakeup_event(event)
        if is_wakeup:
            if not _has_real_wakeup_sender(event):
                return
        else:
            await self._maybe_observe_contact(event)
            # P1 生命周期维护（节流；expire 轻量同步，consolidate 含 LLM 调用后台化以免阻塞用户首轮）
            await self._maybe_expire_stale_memories()
            task = asyncio.create_task(self._maybe_consolidate_memories(event))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        try:
            # 累积请求快照（等待响应后完成；程序唤醒不参与自动提取）
            if not is_wakeup:
                self._accumulate_request_snapshot(event, request)

            # 构建召回查询
            contexts = _normalize_contexts(request.contexts)
            deadline = time.monotonic() + recall_timeout(self.config)
            query = _build_recall_query(
                request.prompt or "", contexts, query_max_chars(self.config)
            )

            contact_summary = await self.memory_mgr.get_contact_summary(event)
            logger.debug(
                "[简单长期记忆] 自动注入查询已构建: contexts=%s, query=%s",
                len(contexts),
                _debug_content_summary(query, preview_limit=0),
            )

            # 池空短路：无活跃记忆时跳过查询优化与召回，避免空库白烧 LLM/embedding
            if not await asyncio.wait_for(
                self.memory_mgr.has_any_active_memory(),
                timeout=max(0.001, deadline - time.monotonic()),
            ):
                logger.debug("[简单长期记忆] 召回短路: 无活跃记忆, source=auto_inject")
                if contact_summary:
                    _inject_memory_context(request, contact_summary)
                return

            # 检索优化：调用 LLM 提炼关键词
            if self.config.get("optimize_recall_query", False):
                try:
                    query = await asyncio.wait_for(
                        self._optimize_recall_query(event, query),
                        timeout=min(
                            recall_timeout(self.config) * 0.4,
                            max(0.001, deadline - time.monotonic()),
                        ),
                    )
                except Exception as e:
                    logger.debug(
                        "[简单长期记忆] 查询优化降级: error_type=%s", type(e).__name__
                    )

            # 召回相关记忆（普通对话计入 recall_count，唤醒路径只读注入）
            memories = await self.memory_mgr.recall_memories(
                event=event,
                query=query,
                top_k=self.config.get("max_memories_per_inject", 5),
                bump=not is_wakeup,
                source="auto_inject",
                deadline=deadline,
            )

            safe_memory_context = ""
            if memories:
                # format_memory_for_injection 已返回含 <user_context_reference> 包装的完整字符串
                safe_memory_context = format_memory_for_injection(
                    memories,
                    timezone_name=self._get_display_timezone(event.unified_msg_origin),
                    viewer_user_id=self._get_viewer_user_id(event),
                )
            injection_content = "\n".join(
                part for part in (safe_memory_context, contact_summary) if part
            )
            if injection_content:
                inject_target = _inject_memory_context(request, injection_content)
                logger.debug(
                    "[简单长期记忆] 自动注入完成: target=%s, memory_count=%s, "
                    "has_contact_summary=%s, context_len=%s, memories=%s",
                    inject_target,
                    len(memories),
                    bool(contact_summary),
                    len(injection_content),
                    [_debug_memory_summary(memory) for memory in memories],
                )
        except Exception as e:
            logger.warning(f"[简单长期记忆] 注入记忆失败: {e}")

    @_on_agent_begin_decorator()
    async def on_agent_begin(self, event: AstrMessageEvent, run_context: Any):
        """标记由 Agent 生命周期收口的事件，避免子代理与 Steering 重复触发提取。"""
        if not self.memory_mgr:
            return
        if not (
            self.config.get("auto_memorize", True) or self._inbox_collection_active()
        ):
            return
        if _is_wakeup_event(event):
            return
        if getattr(event, "_is_maintenance_internal", False):
            return
        _mark_agent_managed_event(self, event)
        if _is_subagent_run(run_context):
            return

        raw_messages = getattr(run_context, "messages", None)
        messages = raw_messages if isinstance(raw_messages, list) else []
        _set_agent_run_state(
            self,
            event,
            {
                "start_len": len(messages),
                "anchor": messages[-1] if messages else None,
                "message_str": str(getattr(event, "message_str", "") or "").strip(),
            },
        )

    @_on_agent_done_decorator()
    async def on_agent_done(
        self,
        event: AstrMessageEvent,
        run_context: Any,
        response: LLMResponse | None = None,
    ):
        """主 Agent 单轮收口：快照提取（auto_memorize）与缓冲表采集相互独立。"""
        if not self.memory_mgr:
            return
        inbox_active = self._inbox_collection_active()
        if not (self.config.get("auto_memorize", True) or inbox_active):
            return
        if _is_wakeup_event(event) or _is_subagent_run(run_context):
            return
        if getattr(event, "_is_maintenance_internal", False):
            _pop_agent_run_state(self, event)
            return

        _mark_agent_managed_event(self, event)
        run_state = _pop_agent_run_state(self, event)
        end_reason = getattr(run_context, "run_end_reason", None)
        resp_role = getattr(response, "role", None)
        turn_messages = _collect_turn_messages(run_context, run_state)

        final_text = ""
        if resp_role != "err":
            candidate = (
                getattr(response, "completion_text", "")
                or getattr(response, "result", "")
                or ""
            ).strip()
            if not (
                end_reason == "cancelled" and candidate == _ASTRBOT_INTERRUPTION_MESSAGE
            ):
                final_text = candidate

        if final_text:
            last_assistant = next(
                (text for role, text in reversed(turn_messages) if role == "assistant"),
                "",
            )
            if final_text != last_assistant:
                turn_messages.append(("assistant", final_text))

        assistant_replies = [
            text for role, text in turn_messages if role == "assistant"
        ]
        assistant_output = "\n".join(assistant_replies)
        if not assistant_output and (
            end_reason in {"cancelled", "error"} or resp_role == "err"
        ):
            self._clear_pending_snapshot(event)
            return

        inbox_row_id: int | None = None
        if (
            inbox_active
            and end_reason not in {"cancelled", "error"}
            and resp_role != "err"
        ):
            inbox_row_id = await self._append_inbox_turn(
                # response 为空时回退到本轮已收集的助手回复，避免 brief 无故为空
                event,
                run_state,
                turn_messages,
                final_text or assistant_output,
            )

        # 实时提取只受 auto_memorize 控制；缓冲采集独立于该开关
        extraction: dict[str, Any] = {"extraction_performed": False}
        if self.config.get("auto_memorize", True):
            extraction = await self._record_response_and_maybe_extract(
                event, assistant_output, turn_messages=turn_messages
            )
        if (
            inbox_row_id is not None
            and extraction.get("extraction_performed")
            and self.memory_mgr
        ):
            inbox = getattr(self.memory_mgr, "inbox", None)
            if inbox is not None:
                await inbox.mark_realtime_extracted(inbox_row_id)

    def _clear_pending_snapshot(self, event: AstrMessageEvent) -> None:
        """取消或报错且无有效回复时仅清除当前事件对应的待匹配请求。"""
        _pop_pending_event_snapshot(self, event)
        snapshots = getattr(self, "_request_snapshots", None)
        if not isinstance(snapshots, dict):
            return
        entry = snapshots.get(self._get_session_key(event))
        if not entry:
            return
        pending_owner = _resolve_event_ref(entry.get("pending_event_ref"))
        if pending_owner is None or pending_owner is event:
            entry["pending_prompt"] = None
            entry["pending_sender_id"] = ""
            entry["pending_event_ref"] = None

    def _inbox_collection_active(self) -> bool:
        """后台提取缓冲采集是否开启：maintenance_enabled 且 AstrBot 4.30+。"""
        if not self.config.get("maintenance_enabled", False):
            return False
        if not maintenance_supported():
            return False
        inbox = getattr(self.memory_mgr, "inbox", None) if self.memory_mgr else None
        return inbox is not None and inbox.ready

    async def _get_current_conversation_id(self, umo: str) -> str:
        """从真实 ConversationManager 读取当前会话 cid，不额外创建会话。"""
        manager = getattr(self.context, "conversation_manager", None)
        get_curr = getattr(manager, "get_curr_conversation_id", None)
        if not callable(get_curr):
            return ""
        try:
            cid = await get_curr(umo)
        except Exception as e:
            logger.debug(
                "[简单长期记忆] 读取当前会话 cid 失败: error_type=%s", type(e).__name__
            )
            return ""
        return str(cid or "")

    async def _append_inbox_turn(
        self,
        event: AstrMessageEvent,
        run_state: dict[str, Any] | None,
        turn_messages: list[tuple[str, str]],
        assistant_text: str,
    ) -> int | None:
        """把一轮正常收口的主轮次写入后台提取缓冲表，返回行 id。

        sender_id 只取真实事件字段，不从正文推定；日志不输出对话正文。
        """
        inbox = getattr(self.memory_mgr, "inbox", None) if self.memory_mgr else None
        if inbox is None or not inbox.ready:
            return None
        parsed_umo = UMOInfo.parse(event.unified_msg_origin)
        if not parsed_umo.is_valid:
            return None
        initial_text = str((run_state or {}).get("message_str", "") or "").strip()
        if _is_inbox_command_event(event) or _is_inbox_command_turn(initial_text):
            return None
        user_text = _build_inbox_user_text(run_state, turn_messages)
        user_text = "\n".join(line.strip() for line in user_text.splitlines())
        user_text = user_text[:INBOX_USER_TEXT_MAX_CHARS]
        if len(user_text) < INBOX_MIN_USER_TEXT_CHARS:
            return None
        assistant_brief = (assistant_text or "").strip()[
            :INBOX_ASSISTANT_BRIEF_MAX_CHARS
        ]
        if (
            len(user_text) < INBOX_LOW_INFO_USER_CHARS
            and len(assistant_brief) > INBOX_LOW_INFO_ASSISTANT_CHARS
        ):
            # 用户输入极短而内容几乎全在助手输出：单向回显不入缓冲
            return None
        conversation_id = await self._get_current_conversation_id(
            event.unified_msg_origin
        )
        record = InboxRecord(
            umo=event.unified_msg_origin,
            platform_id=parsed_umo.platform_id,
            session_type=parsed_umo.session_type,
            session_id=parsed_umo.session_id,
            conversation_id=conversation_id,
            sender_id=event.get_sender_id(),
            sender_name=event.get_sender_name(),
            user_text=user_text,
            assistant_brief=assistant_brief,
        )
        return await inbox.append(record)

    def _plugin_gate(self) -> MemoryGate:
        """插件级长生命周期闸门，保留 Decision 未配置的探测缓存。"""
        gate = getattr(self, "_memory_gate", None)
        if gate is None:
            gate = MemoryGate(
                decision_service=getattr(self.context, "decision", None),
                provider_manager=getattr(self.context, "provider_manager", None),
            )
            self._memory_gate = gate
        return gate

    @filter.on_llm_response()
    async def extract_memories(self, event: AstrMessageEvent, response: LLMResponse):
        if not self.memory_mgr:
            return

        if not self.config.get("auto_memorize", True):
            return

        if _is_wakeup_event(event) or _is_agent_managed_event(self, event):
            return

        assistant_output = (
            getattr(response, "completion_text", "")
            or getattr(response, "result", "")
            or ""
        )
        await self._record_response_and_maybe_extract(event, assistant_output)

    async def _record_response_and_maybe_extract(
        self,
        event: AstrMessageEvent,
        assistant_output: str,
        *,
        turn_messages: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        """按间隔提取记忆并返回明确结果，供调用方判断是否标记实时提取。"""

        def _result(performed: bool = False, **extra: Any) -> dict[str, Any]:
            base: dict[str, Any] = {
                "extraction_performed": performed,
                "stored": 0,
                "gate_rejected": 0,
                "gate_unavailable": 0,
            }
            base.update(extra)
            return base

        try:
            extraction_interval = _read_positive_int(
                self.config.get("extraction_interval", 20), 20
            )
            if extraction_interval <= 0:
                self._discard_session_snapshots(event)
                return _result()

            # 用响应完成快照（仅在具备轮内有序消息时传关键字参数，兼容旧测试桩）
            if turn_messages:
                self._complete_snapshot_with_response(
                    event, assistant_output, turn_messages=turn_messages
                )
            else:
                self._complete_snapshot_with_response(event, assistant_output)

            # 递增会话对话计数器
            current_count = self._increment_session_counter(event)

            # 检查是否达到提取间隔
            if current_count % extraction_interval != 0:
                return _result()

            # 先校验 UMO 再清空快照缓冲，无效 UMO 保留累积对话等下个间隔重试
            parsed_umo = UMOInfo.parse(event.unified_msg_origin)
            if not parsed_umo.is_valid:
                logger.warning("[简单长期记忆] 自动提取跳过: invalid_umo")
                return _result()

            # 获取累积的快照列表
            snapshots = self._get_and_clear_session_snapshots(event)
            if not snapshots:
                return _result()

            # 构建对话文本（包含所有累积的对话）
            conversation = self._build_conversation_from_snapshots(snapshots)

            # 检查有效用户文本最小长度（不含角色标签与助手输出）
            min_length = self.config.get("extraction_min_content_length", 150)
            user_text_length = self._snapshot_user_text_length(snapshots)
            if user_text_length < min_length:
                logger.debug(
                    f"[简单长期记忆] 有效用户文本长度 {user_text_length} < {min_length}，跳过提取"
                )
                return _result()

            if not conversation:
                return _result()

            # 获取提取模型
            provider_id = await self._get_llm_provider_id(event, "extraction")
            if not provider_id:
                logger.debug("[简单长期记忆] 未配置提取模型，跳过记忆提取")
                return _result()

            allowed_sender_ids = list(
                dict.fromkeys(
                    str(snapshot["sender_id"])
                    for snapshot in snapshots
                    if snapshot.get("sender_id")
                )
            )

            logger.debug(
                "[简单长期记忆] 自动提取开始: snapshots=%s, conversation=%s, "
                "interval=%s, counter=%s, provider_configured=%s",
                len(snapshots),
                _debug_content_summary(conversation, preview_limit=0),
                extraction_interval,
                current_count,
                bool(provider_id),
            )
            extraction_started_at = time.monotonic()
            # 调用 LLM 提取记忆
            prompt = MEMORY_EXTRACTION_PROMPT.substitute(
                platform_id=parsed_umo.platform_id,
                session_type=parsed_umo.session_type,
                session_id=parsed_umo.session_id,
                sender_id=event.get_sender_id(),
                known_sender_ids=json.dumps(allowed_sender_ids, ensure_ascii=False),
                conversation=conversation,
            )
            try:
                llm_response = await self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=prompt,
                )
                extraction_result = getattr(llm_response, "completion_text", "") or ""

                logger.debug(
                    "[简单长期记忆] 自动提取 LLM 完成: elapsed_ms=%s, result=%s",
                    round((time.monotonic() - extraction_started_at) * 1000),
                    _debug_content_summary(extraction_result, preview_limit=0),
                )

            except Exception as e:
                logger.warning(f"[简单长期记忆] LLM 提取调用失败: {e}")
                return _result()

            # 解析提取结果
            memories = self._parse_extracted_memories(
                extraction_result, parsed_umo.session_type
            )
            logger.debug(
                "[简单长期记忆] 自动提取解析完成: parsed=%s",
                len(memories),
            )
            # LLM 已实际调用并返回：即使候选为空也算完成一次提取
            extraction_done = _result(performed=True)
            if not memories:
                return extraction_done

            # 准入闸门：解析后的候选在查重与写入之前批量判定
            gate_outcomes = await self._evaluate_store_gate(
                memories, conversation, provider_id
            )

            # 存储提取的记忆
            stored_count = 0
            skipped_empty_count = 0
            skipped_duplicate_count = 0
            store_failures = 0
            for idx, mem in enumerate(memories):
                outcome = gate_outcomes.get(str(idx))
                if outcome is not None and outcome.status == "rejected":
                    extraction_done["gate_rejected"] += 1
                    continue
                if outcome is None or outcome.status == "unavailable":
                    # 判定失败 fail closed：本条不入库，不中断其余候选
                    extraction_done["gate_unavailable"] += 1
                    continue
                memory_domain = mem.get("type", "fact")
                scope = mem.get("scope", MemoryScope.PERSONAL)
                content = mem.get("content", "")
                subject = mem.get("subject", "")
                subjects = mem.get("subjects", []) or ([subject] if subject else [])
                if scope == MemoryScope.PERSONAL:
                    try:
                        subjects = validate_owner_ids(subjects, allowed_sender_ids)
                    except ValueError:
                        logger.warning(
                            "[简单长期记忆] 自动提取拒绝个人记忆: untrusted_subject"
                        )
                        continue
                else:
                    subject = _current_speaker_subject(event, scope)
                entities = mem.get("entities", [])
                topics = mem.get("topics", [])
                disclosure = mem.get("disclosure", "")
                importance = mem.get("importance", 3)
                owner_sender_ids = subjects if scope == MemoryScope.PERSONAL else []

                if not content:
                    skipped_empty_count += 1
                    continue

                domain = normalize_domain(memory_domain)

                duplicate_uri = await self.memory_mgr.find_scoped_duplicate(
                    content=content,
                    memory_scope=scope,
                    platform_id=parsed_umo.platform_id,
                    session_id=parsed_umo.session_id,
                    umo=event.unified_msg_origin,
                    owner_user_id=(
                        build_user_id(parsed_umo.platform_id, subjects[0])
                        if scope == MemoryScope.PERSONAL and subjects
                        else ""
                    ),
                )
                if duplicate_uri:
                    skipped_duplicate_count += 1
                    logger.debug(
                        f"[简单长期记忆] 自动提取查重命中，跳过写入: {duplicate_uri}"
                    )
                    continue
                try:
                    uri = await self.memory_mgr.store_memory(
                        event=event,
                        content=content,
                        domain=domain,
                        memory_type=MemoryType.NORMAL,
                        disclosure=disclosure,
                        importance=importance,
                        memory_scope=scope,
                        subject=subject,
                        entities=entities,
                        topics=topics,
                        owner_sender_ids=owner_sender_ids,
                        allowed_sender_ids=allowed_sender_ids,
                    )
                    stored_count += 1
                    logger.debug(
                        "[简单长期记忆] 自动提取写入完成: uri=%s, domain=%s, "
                        "scope=%s, content=%s",
                        uri,
                        domain,
                        scope,
                        _debug_content_summary(content),
                    )
                except Exception as e:
                    store_failures += 1
                    logger.warning(
                        "[简单长期记忆] 自动提取写入失败: domain=%s, scope=%s, "
                        "content=%s, error_type=%s, error=%s",
                        domain,
                        scope,
                        _debug_content_summary(content),
                        type(e).__name__,
                        e,
                    )

            extraction_done["stored"] = stored_count
            logger.debug(
                "[简单长期记忆] 自动提取完成: snapshots=%s, parsed=%s, "
                "stored=%s, skipped_empty=%s, skipped_duplicate=%s, "
                "gate_rejected=%s, gate_unavailable=%s, store_failures=%s, "
                "elapsed_ms=%s",
                len(snapshots),
                len(memories),
                stored_count,
                skipped_empty_count,
                skipped_duplicate_count,
                extraction_done["gate_rejected"],
                extraction_done["gate_unavailable"],
                store_failures,
                round((time.monotonic() - extraction_started_at) * 1000),
            )
            logger.info(
                f"[简单长期记忆] 已从 {len(snapshots)} 轮对话中存储 {stored_count} 条记忆（候选 {len(memories)} 条）"
            )
            return extraction_done

        except Exception as e:
            logger.warning(f"[简单长期记忆] 提取记忆失败: {e}")
            return _result()

    async def _evaluate_store_gate(
        self,
        memories: list[dict[str, Any]],
        conversation: str,
        provider_id: str | None,
    ) -> dict[str, GateOutcome]:
        """对实时提取候选批量执行准入闸门；异常时全部按 unavailable fail closed。"""
        candidates = [
            GateCandidate(
                candidate_id=str(idx),
                content=str(mem.get("content", "")),
                evidence=conversation,
            )
            for idx, mem in enumerate(memories)
        ]
        try:
            gate = self._plugin_gate()
            judge = (
                LlmGenerateJudge(self.context, provider_id, GATE_TOOL_TIMEOUT_SECONDS)
                if provider_id
                else None
            )
            return await gate.evaluate(candidates, chat_judge=judge)
        except Exception as e:
            logger.warning(
                "[简单长期记忆] 闸门判定异常，候选全部 fail closed: error_type=%s",
                type(e).__name__,
            )
            return {
                candidate.candidate_id: GateOutcome(
                    "unavailable", "gate evaluation error"
                )
                for candidate in candidates
            }

    async def _gate_tool_content(
        self, event: AstrMessageEvent, content: str
    ) -> str | None:
        """memory_store 工具的准入闸门；返回错误文案或 None（放行）。"""
        provider_id = await self._get_llm_provider_id(event, "extraction")
        outcome: GateOutcome | None = None
        try:
            gate = self._plugin_gate()
            judge = (
                LlmGenerateJudge(self.context, provider_id, GATE_TOOL_TIMEOUT_SECONDS)
                if provider_id
                else None
            )
            outcomes = await gate.evaluate(
                [GateCandidate(candidate_id="tool", content=content)],
                chat_judge=judge,
            )
            outcome = outcomes.get("tool")
        except Exception as e:
            logger.warning(
                "[简单长期记忆] 工具写入闸门异常: error_type=%s", type(e).__name__
            )
        if outcome is not None and outcome.status == "approved":
            return None
        if outcome is not None and outcome.status == "rejected":
            return f"Error: memory rejected by admission gate ({outcome.reason})"
        return "Error: memory gate temporarily unavailable, please retry later"

    # ==================== 用户命令 ====================

    @filter.command_group("memory")
    def memory_group(self):
        """记忆管理指令组"""
        pass

    @memory_group.command("list")
    async def cmd_list(self, event: AstrMessageEvent):
        """列出记忆 /memory list [--all] [页码]"""
        if err := _ensure_initialized(self.memory_mgr):
            yield event.plain_result(err)
            return
        args = _parse_memory_flags(_parse_command_args(event, "memory list"))
        if err := _validate_command(event, args, cmd_name="list", allow_all=True):
            yield event.plain_result(err)
            return
        all_users = args["all"]
        # 解析页码
        page = 1
        positional = args["positional"]
        if positional:
            try:
                page = int(positional)
            except ValueError:
                yield event.plain_result("页码必须是正整数")
                return
            if page < 1:
                yield event.plain_result("页码必须是正整数")
                return
        page_size = 10
        memories, total, truncated = await self.memory_mgr.list_memories(
            event, page=page, page_size=page_size, all_users=all_users
        )
        scope = "全局" if all_users else "个人"
        result = format_memory_for_user(
            memories,
            page=page,
            total=total,
            page_size=page_size,
            all_mode=all_users,
            cmd_prefix=self._get_cmd_prefix(event.unified_msg_origin),
        )
        if truncated:
            result += (
                "\n\n提示: 群聊可见记忆较多，当前总数受扫描上限影响，可能还有更多记录。"
            )
        yield event.plain_result(f"[{scope}记忆]\n{result}")

    @memory_group.command("search")
    async def cmd_search(self, event: AstrMessageEvent):
        """搜索记忆 /memory search [--all] <关键词>"""
        if err := _ensure_initialized(self.memory_mgr):
            yield event.plain_result(err)
            return
        args = _parse_memory_flags(_parse_command_args(event, "memory search"))
        if err := _validate_command(event, args, cmd_name="search", allow_all=True):
            yield event.plain_result(err)
            return
        all_users = args["all"]
        query = args["positional"]
        if not query:
            yield event.plain_result("请提供搜索关键词")
            return
        memories = await self.memory_mgr.recall_memories(
            event, query, all_users=all_users, source="command_search"
        )
        scope = "全局" if all_users else "个人"
        result = format_memory_for_user(
            memories,
            total=len(memories),
            page_size=max(1, len(memories)),
            cmd_prefix=self._get_cmd_prefix(event.unified_msg_origin),
        )
        yield event.plain_result(f"[{scope}搜索]\n{result}")

    @memory_group.command("stats")
    async def cmd_stats(self, event: AstrMessageEvent):
        """查看记忆统计 /memory stats [--all]"""
        if err := _ensure_initialized(self.memory_mgr):
            yield event.plain_result(err)
            return
        args = _parse_memory_flags(_parse_command_args(event, "memory stats"))
        if err := _validate_command(
            event, args, cmd_name="stats", allow_all=True, allow_positional=False
        ):
            yield event.plain_result(err)
            return
        all_users = args["all"]
        stats = await self.memory_mgr.get_memory_stats(event, all_users=all_users)
        scope = "全局" if all_users else "个人"
        result = (
            f"[{scope}记忆统计]\n"
            f"  总数: {stats['total']}\n"
            f"  永久记忆: {stats['permanent']}\n"
            f"  普通记忆: {stats['normal']}\n"
            f"  历史已压缩: {stats['compressed']}"
        )
        yield event.plain_result(result)

    @memory_group.command("test")
    async def cmd_test(self, event: AstrMessageEvent):
        """测试读写或后台整理阶段（管理员）。"""
        if err := _ensure_initialized(self.memory_mgr):
            yield event.plain_result(err)
            return
        if not event.is_admin():
            yield event.plain_result("该操作需要管理员权限")
            return

        args_text = _parse_command_args(event, "memory test")
        try:
            args = shlex.split(args_text)
        except ValueError as e:
            yield event.plain_result(f"参数解析失败: {e}")
            return
        if len(args) > 1:
            yield event.plain_result(
                "用法: /memory test [purge|extract|organizer|analyst|reviewer|cycle]"
            )
            return
        stage = args[0].lower() if args else "storage"
        if stage in {"help", "--help", "-h"}:
            yield event.plain_result(
                "用法: /memory test [purge|extract|organizer|analyst|reviewer|cycle]\n"
                "不带参数测试记忆读写；后台阶段固定 dry-run，不写入记忆或待审队列。"
            )
            return
        if stage in {"storage", "memory"}:
            yield event.plain_result(await self._run_memory_test(event))
            return
        if stage not in {
            "purge",
            "extract",
            "organizer",
            "analyst",
            "reviewer",
            "cycle",
        }:
            yield event.plain_result(
                "未知测试项。合法值: purge, extract, organizer, analyst, reviewer, cycle"
            )
            return
        yield event.plain_result(await self._run_maintenance_test(stage))

    @memory_group.command("forget")
    async def cmd_forget(self, event: AstrMessageEvent):
        """删除记忆 /memory forget <uri> [--user <id>]"""
        if err := _ensure_initialized(self.memory_mgr):
            yield event.plain_result(err)
            return
        args = _parse_memory_flags(_parse_command_args(event, "memory forget"))
        if err := _validate_command(event, args, cmd_name="forget", allow_user=True):
            yield event.plain_result(err)
            return
        target_user_id = args["user"]
        uri = args["positional"]
        if not uri:
            yield event.plain_result("请提供要删除的记忆 URI")
            return

        is_admin = event.is_admin()

        if err := authorize_memory_mutation(event, target_user_id=target_user_id):
            yield event.plain_result(err)
            return

        if target_user_id:
            # 管理员删除指定用户的记忆
            deleted = await self.memory_mgr.forget_memory_by_user(
                event, uri, target_user_id
            )
            if deleted == 0:
                yield event.plain_result(f"未找到用户 {target_user_id} 的记忆: {uri}")
            else:
                yield event.plain_result(
                    f"已删除用户 {target_user_id} 的 {deleted} 条记忆: {uri}"
                )
        elif is_admin:
            # 管理员按 URI 删除所有用户
            deleted = await self.memory_mgr.forget_memory_by_uri(uri)
            if deleted == 0:
                yield event.plain_result(f"未找到匹配的记忆: {uri}")
            else:
                yield event.plain_result(f"已删除 {deleted} 条记忆: {uri}")
        else:
            # 普通用户只能删自己的
            deleted = await self.memory_mgr.forget_memory(event, uri)
            if deleted > 0:
                yield event.plain_result(f"已删除记忆: {uri}")
            else:
                yield event.plain_result(f"未找到或无权限删除记忆: {uri}")

    @memory_group.command("clear")
    async def cmd_clear(self, event: AstrMessageEvent):
        """清空当前用户记忆；管理员可指定用户或全部记忆。"""
        if err := _ensure_initialized(self.memory_mgr):
            yield event.plain_result(err)
            return
        args = _parse_memory_flags(_parse_command_args(event, "memory clear"))
        if err := _validate_command(
            event,
            args,
            cmd_name="clear",
            allow_all=True,
            allow_user=True,
            allow_positional=False,
        ):
            yield event.plain_result(err)
            return
        if args["all"] and args["user"]:
            yield event.plain_result("--all 与 --user 不可同时使用")
            return

        if err := authorize_memory_mutation(
            event,
            target_user_id=args["user"],
            all_users=args["all"],
        ):
            yield event.plain_result(err)
            return
        if args["all"]:
            preview = await self.memory_mgr.count_clear_memories(event, all_users=True)
            scope = "全部记忆"
        elif args["user"]:
            preview = await self.memory_mgr.count_clear_memories_by_user(
                event, args["user"]
            )
            scope = f"用户 {args['user']} 的记忆"
        else:
            preview = await self.memory_mgr.count_clear_memories(event)
            scope = "当前用户记忆"
        yield event.plain_result(
            "[清空确认]\n"
            f"  范围: {scope}\n"
            f"  预计影响: {preview} 条\n"
            "  接触记录: 同范围一并清空\n"
            "  请在本会话发送“确认”执行，发送“取消”放弃"
        )
        confirmation = await _wait_for_confirmation(
            event, _get_confirmation_waiters(self)
        )
        if confirmation.status == "timeout":
            yield event.plain_result("确认超时，操作未执行")
            return
        if confirmation.status == "busy":
            yield event.plain_result("已有待确认操作，请先发送“确认”或“取消”")
            return
        if confirmation.status != "confirmed":
            yield event.plain_result("操作已取消，未删除任何记忆")
            return
        event = confirmation.event or event
        if err := authorize_memory_mutation(
            event,
            target_user_id=args["user"],
            all_users=args["all"],
        ):
            yield event.plain_result(err)
            return
        if args["all"]:
            current_preview = await self.memory_mgr.count_clear_memories(
                event, all_users=True
            )
        elif args["user"]:
            current_preview = await self.memory_mgr.count_clear_memories_by_user(
                event, args["user"]
            )
        else:
            current_preview = await self.memory_mgr.count_clear_memories(event)
        if current_preview != preview:
            yield event.plain_result("确认期间目标数量已变化，请重新预览确认")
            return

        if args["all"]:
            result = await self.memory_mgr.clear_memories(event, all_users=True)
            message = f"已清空全部 {result.memory_count} 条记忆"
        elif args["user"]:
            target_user_id = args["user"]
            result = await self.memory_mgr.clear_memories_by_user(event, target_user_id)
            message = f"已清空用户 {target_user_id} 的 {result.memory_count} 条记忆"
        else:
            result = await self.memory_mgr.clear_memories(event)
            message = f"已清空 {result.memory_count} 条记忆"
        if result.contact_cleanup_failed:
            message += "\n接触记录清理失败；记忆已清空，请查看日志后重试"
        yield event.plain_result(message)

    @memory_group.command("rebuild")
    async def cmd_rebuild(self, event: AstrMessageEvent):
        """重建或迁移记忆（管理员）/memory rebuild [--to <知识库名>] [--clear-cache]"""
        if err := _ensure_initialized(self.memory_mgr):
            yield event.plain_result(err)
            return
        args = _parse_memory_flags(_parse_command_args(event, "memory rebuild"))
        if err := _validate_command(
            event,
            args,
            cmd_name="rebuild",
            require_admin=True,
            allow_to=True,
            allow_clear_cache=True,
            allow_positional=False,
        ):
            yield event.plain_result(err)
            return

        # --clear-cache: 清理重建缓存
        if args["clear_cache"]:
            if args["to"]:
                yield event.plain_result("--clear-cache 不能与 --to 同时使用")
                return
            cache_status = await self.memory_mgr.get_rebuild_cache_status()
            if (
                self.memory_mgr.is_rebuilding
                or cache_status.get("status") == "in_progress"
            ):
                yield event.plain_result("重建/迁移正在进行中，不能清理缓存")
                return
            cache_fingerprint = self.memory_mgr.rebuild_cache_fingerprint(cache_status)
            yield event.plain_result(
                "[清理重建缓存确认]\n"
                f"  缓存记录数: {cache_status.get('memory_records', 0)}\n"
                f"  缓冲写入数: {cache_status.get('pending_writes', 0)}\n"
                f"  上次状态: {cache_status.get('status', '无')}\n"
                f"  源知识库: {cache_status.get('source_kb', '无')}\n"
                f"  目标知识库: {cache_status.get('target_kb', '无')}\n"
                "  请在本会话发送“确认”执行，发送“取消”放弃"
            )
            confirmation = await _wait_for_confirmation(
                event, _get_confirmation_waiters(self)
            )
            if confirmation.status == "timeout":
                yield event.plain_result("确认超时，操作未执行")
                return
            if confirmation.status == "busy":
                yield event.plain_result("已有待确认操作，请先发送“确认”或“取消”")
                return
            if confirmation.status != "confirmed":
                yield event.plain_result("操作已取消，未清理重建缓存")
                return
            event = confirmation.event or event
            if not event.is_admin():
                yield event.plain_result("该操作需要管理员权限")
                return
            current_status = await self.memory_mgr.get_rebuild_cache_status()
            if (
                self.memory_mgr.rebuild_cache_fingerprint(current_status)
                != cache_fingerprint
            ):
                yield event.plain_result("重建缓存状态已变化，请重新预览确认")
                return
            result = await self.memory_mgr.clear_rebuild_cache()
            report = "[清理重建缓存]\n"
            report += f"  缓存记录数: {cache_status.get('memory_records', 0)}\n"
            report += f"  缓冲写入数: {cache_status.get('pending_writes', 0)}\n"
            report += f"  上次状态: {cache_status.get('status', '无')}\n"
            report += f"  源知识库: {cache_status.get('source_kb', '无')}\n"
            report += f"  目标知识库: {cache_status.get('target_kb', '无')}\n"
            report += "  清理结果:\n"
            for key, ok in result.items():
                report += f"    {key}: {'成功' if ok else '失败'}\n"
            yield event.plain_result(report)
            return

        target_kb_name = args["to"] or None
        logger.debug(
            "[简单长期记忆] rebuild 命令开始: target_kb=%s, clear_cache=%s, "
            "current_kb=%s",
            target_kb_name or "(current)",
            args["clear_cache"],
            self.memory_mgr.current_kb_name,
        )

        # --to 与当前 KB 同名时视为原地重建
        if target_kb_name and self.memory_mgr.current_kb_name == target_kb_name:
            target_kb_name = None

        source_kb_id = await self.memory_mgr.get_kb_id_by_name(
            self.memory_mgr.current_kb_name
        )
        target_kb_id = (
            await self.memory_mgr.get_kb_id_by_name(target_kb_name)
            if target_kb_name
            else source_kb_id
        )
        cmd_prefix = _render_cmd_prefix(self._get_cmd_prefix(event.unified_msg_origin))
        source_count = await self.memory_mgr.count_kb_memory_records_by_name(
            self.memory_mgr.current_kb_name
        )
        target_count = (
            await self.memory_mgr.count_kb_memory_records_by_name(target_kb_name)
            if target_kb_name
            else source_count
        )
        target_missing = target_kb_name is not None and not target_kb_id
        mode = "迁移" if target_kb_name else "重建"
        report = (
            f"[{mode}确认]\n"
            f"  当前知识库: {self.memory_mgr.current_kb_name}\n"
            f"  目标知识库: {target_kb_name or self.memory_mgr.current_kb_name}\n"
            f"  当前记忆数: {source_count}\n"
        )
        if target_kb_name:
            if target_missing:
                report += "  目标知识库不存在，请检查名称\n"
                yield event.plain_result(report)
                return
            report += f"  目标已有记忆数: {target_count}\n"
            if target_count > 0:
                report += "  目标已有记忆，执行前需要先清空或更换目标\n"
                yield event.plain_result(report)
                return
        report += "  请在本会话发送“确认”执行，发送“取消”放弃\n"
        yield event.plain_result(report)
        logger.debug(
            "[简单长期记忆] rebuild 确认预览: mode=%s, source_count=%s, "
            "target_count=%s, target_missing=%s, session_confirmation=%s",
            mode,
            source_count,
            target_count,
            target_missing,
            True,
        )
        confirmation = await _wait_for_confirmation(
            event, _get_confirmation_waiters(self)
        )
        if confirmation.status == "timeout":
            yield event.plain_result("确认超时，操作未执行")
            return
        if confirmation.status == "busy":
            yield event.plain_result("已有待确认操作，请先发送“确认”或“取消”")
            return
        if confirmation.status != "confirmed":
            yield event.plain_result("操作已取消，未执行重建或迁移")
            return
        event = confirmation.event or event
        if not event.is_admin():
            yield event.plain_result("该操作需要管理员权限")
            return
        current_source_id = await self.memory_mgr.get_kb_id_by_name(
            self.memory_mgr.current_kb_name
        )
        current_target_id = (
            await self.memory_mgr.get_kb_id_by_name(target_kb_name)
            if target_kb_name
            else current_source_id
        )
        if current_source_id != source_kb_id or current_target_id != target_kb_id:
            yield event.plain_result("知识库状态已变化，请重新预览确认")
            return
        if not source_kb_id:
            yield event.plain_result("当前知识库不存在，请检查配置")
            return
        if target_kb_name and not target_kb_id:
            yield event.plain_result("目标知识库不存在，请重新预览确认")
            return

        if target_kb_name:
            yield event.plain_result(
                f"正在迁移记忆到知识库 '{target_kb_name}'，请稍候..."
            )
        else:
            yield event.plain_result("正在当前知识库重建所有记忆，请稍候...")

        logger.debug(
            "[简单长期记忆] rebuild 执行确认: mode=%s, source_kb=%s, target_kb=%s, "
            "source_id=%s, target_id=%s",
            "migration" if target_kb_name else "rebuild",
            self.memory_mgr.current_kb_name,
            target_kb_name or self.memory_mgr.current_kb_name,
            source_kb_id,
            target_kb_id,
        )
        try:
            result = await self.memory_mgr.rebuild_memories(
                target_kb_name=target_kb_name,
            )
            logger.debug(
                "[简单长期记忆] rebuild 执行返回: status=%s, total=%s, success=%s, "
                "failed=%s, pending_flushed=%s, verification=%s",
                result.get("status"),
                result.get("total"),
                result.get("success"),
                result.get("failed"),
                result.get("pending_flushed"),
                result.get("verification", {}).get("passed"),
            )
            status = result.get("status", "completed")
            is_interrupted = status == "interrupted"

            if is_interrupted:
                # 失败路径：明确告知异常终止
                mode = "迁移" if result["is_migration"] else "重建"
                error_msg = result.get("error", "未知错误")
                if result["is_migration"] and result.get("success", 0) > 0:
                    error_msg += (
                        "；目标知识库可能已有已写入副本，源知识库未确认清理完成"
                    )
                report = (
                    f"[{mode}异常终止]\n"
                    f"  原因: {error_msg}\n"
                    f"  已处理: {result['success']} 条\n"
                    f"  缓冲写入: {result.get('pending_flushed', 0)} 条"
                )
                report += (
                    "\n\n  重建缓存已保留，请排查问题后重试。"
                    "\n  确认无需恢复后可执行:"
                    f"\n  {cmd_prefix}memory rebuild --clear-cache"
                )
                yield event.plain_result(report)
                return

            mode = "迁移" if result["is_migration"] else "重建"
            report = (
                f"[{mode}完成]\n"
                f"  目标知识库: {result['target_kb']}\n"
                f"  总计: {result['total']}\n"
                f"  成功: {result['success']}\n"
                f"  失败: {result['failed']}"
            )
            if result["pending_flushed"]:
                report += f"\n  缓冲写入: {result['pending_flushed']} 条"

            # 完整性校验结果
            v = result.get("verification", {})
            if v:
                report += "\n\n[完整性校验]"
                if v.get("error"):
                    report += f"\n  校验异常: {v['error']}"
                elif v["passed"]:
                    report += (
                        f"\n  状态: 通过"
                        f"\n  预期: {v['expected']} 条, "
                        f"实际: {v['actual']} 条"
                    )
                else:
                    report += (
                        f"\n  状态: 不一致"
                        f"\n  预期: {v['expected']} 条, "
                        f"实际: {v['actual']} 条, "
                        f"差异: {v['diff']:+d} 条"
                    )

            # 迁移提示：仅当实际提交成功时显示切换信息
            if result["is_migration"]:
                committed = result.get("migration_committed", False)
                if committed:
                    report += f"\n\n  插件已切换到知识库: {result['target_kb']}"
                    report += (
                        "\n  注意: 请同步更新插件配置中的知识库选项，"
                        "否则重启后将回退到旧知识库"
                    )
                else:
                    report += "\n\n  迁移未完成（存在失败），插件仍使用原知识库"

            if v and v.get("passed"):
                report += (
                    "\n\n  数据校验通过，请确认记忆无误后执行:"
                    f"\n  {cmd_prefix}memory rebuild --clear-cache"
                )
            elif v and not v.get("passed") and not v.get("error"):
                report += (
                    "\n\n  数据校验未通过，请排查后重试。"
                    "重建缓存已保留，可执行:"
                    f"\n  {cmd_prefix}memory rebuild --clear-cache 清理缓存"
                )

            yield event.plain_result(report)
        except ValueError as e:
            logger.debug(
                "[简单长期记忆] rebuild 命令 ValueError: error=%s",
                e,
            )
            yield event.plain_result(str(e))
        except Exception as e:
            logger.exception("[简单长期记忆] rebuild 命令异常")
            yield event.plain_result(f"重建失败: {e}")

    @memory_group.command("review")
    async def cmd_review(self, event: AstrMessageEvent):
        """争议审查 /memory review [list|approve|reject|clear] [id]"""
        if not event.is_admin():
            yield event.plain_result("仅管理员可用")
            return
        args = _parse_command_args(event, "memory review").strip().split()
        action = args[0] if args else "list"

        if action == "list":
            queue = await self.get_kv_data("maintenance_pending_review", None) or []
            pending = [q for q in queue if q.get("status") == "pending"]
            if not pending:
                yield event.plain_result("✅ 无待审争议项")
                return
            lines = [f"📝 待审争议 ({len(pending)} 条)："]
            for item in pending:
                lines.append(
                    f"#{item['id']} [{item.get('op_type', '?')}] "
                    f"{item.get('verdict_reason', '')} "
                    f"({item.get('created_at', '')})"
                )
            lines.append("\n/memory review approve <id> 批准")
            lines.append("/memory review reject <id> 废案")
            lines.append("/memory review clear 清空")
            text = "\n".join(lines)
            if _chatui_view_supported(event):
                # ChatUI 平台发交互卡片，text 同时作为非 ChatUI 平台的文字回退
                yield event.view_result(
                    name=REVIEW_VIEW_NAME,
                    data=_review_card_data(pending),
                    text=text,
                    title="记忆待审列表",
                )
            else:
                yield event.plain_result(text)

        elif action == "clear":
            async with self._review_lock:
                await self.put_kv_data("maintenance_pending_review", [])
            yield event.plain_result("✅ 待审队列已清空")

        elif action in ("approve", "reject"):
            if len(args) < 2:
                yield event.plain_result(f"用法: /memory review {action} <id>")
                return
            try:
                target_id = int(args[1])
            except ValueError:
                yield event.plain_result("id 必须是数字")
                return
            feedback, _pending = await self._decide_review_item(action, target_id)
            yield event.plain_result(feedback)
        else:
            yield event.plain_result(
                "用法: /memory review [list|approve <id>|reject <id>|clear]"
            )

    async def _decide_review_item(
        self, action: str, target_id: int
    ) -> tuple[str, list[dict[str, Any]]]:
        """裁决一条待审项，返回 (反馈文本, 最新待审列表)。

        approve 复用 runner.execute_approved，与后台周期阶段互斥；
        成功才标记 approved，执行失败保留 pending 可重试。
        """
        async with self._review_lock:
            queue = await self.get_kv_data("maintenance_pending_review", None) or []
            pending = [q for q in queue if q.get("status") == "pending"]
            found = next((q for q in pending if q.get("id") == target_id), None)
            if found is None:
                return f"未找到待审项 #{target_id}（可能已被处理）", pending

            if action == "reject":
                found["status"] = "rejected"
                await self.put_kv_data("maintenance_pending_review", queue)
                return f"🚫 #{target_id} 已废案", [
                    q for q in queue if q.get("status") == "pending"
                ]

            runner = self._scheduler.runner if self._scheduler else None
            if not runner:
                return f"⚠️ #{target_id} 调度器未运行，无法执行", pending
            try:
                ok = await runner.execute_approved(found.get("op", {}))
            except Exception as e:
                return f"⚠️ #{target_id} 执行异常: {e}", pending
            if not ok:
                return f"⚠️ #{target_id} 执行失败，条目保留待审可重试", pending
            found["status"] = "approved"
            await self.put_kv_data("maintenance_pending_review", queue)
            return f"✅ #{target_id} 已批准并执行", [
                q for q in queue if q.get("status") == "pending"
            ]

    async def _handle_review_action(
        self, event: AstrMessageEvent, action: str
    ) -> AsyncGenerator[Any, None]:
        """处理审核卡片的通过/驳回回调，成功后用最新队列刷新卡片。"""
        if not event.is_admin():
            raise PermissionError("仅管理员可操作待审队列")
        interaction = event.get_ui_interaction()
        if interaction is None:
            return
        data = interaction.data if isinstance(interaction.data, dict) else None
        target_id = data.get("id") if isinstance(data, dict) else None
        if not isinstance(target_id, int) or isinstance(target_id, bool):
            raise ValueError("卡片操作缺少有效的待审 id")
        feedback, pending = await self._decide_review_item(action, target_id)
        error = feedback if feedback.startswith("⚠️") else ""
        notice = "" if error else feedback
        yield event.view_result(
            data=_review_card_data(pending, notice=notice, error=error)
        )

    @_ui_action("approve")
    async def review_approve_action(self, event: AstrMessageEvent):
        """审核卡片回调：通过一条待审操作。"""
        async for result in self._handle_review_action(event, "approve"):
            yield result

    @_ui_action("reject")
    async def review_reject_action(self, event: AstrMessageEvent):
        """审核卡片回调：驳回一条待审操作。"""
        async for result in self._handle_review_action(event, "reject"):
            yield result

    async def _run_memory_test(self, event: AstrMessageEvent) -> str:
        """执行一次记忆写入-读取测试并返回报告"""
        test_content = "memory_test_probe_这是一条测试记忆"
        test_domain = "facts"
        uri = str(MemoryURI.generate(test_domain))
        report = ["[记忆读写测试]"]

        # 写入测试
        try:
            await self.memory_mgr.store_memory(
                event=event,
                content=test_content,
                domain=test_domain,
                uri=uri,
                memory_type=MemoryType.NORMAL,
                disclosure="测试",
                importance=1,
            )
            report.append(f"  写入: 成功 (URI: {uri})")
        except Exception as e:
            report.append(f"  写入: 失败 ({e})")
            return "\n".join(report)

        # 读取测试
        hit = False
        try:
            results = await self.memory_mgr.recall_memories(
                event=event,
                query=test_content,
                top_k=3,
                source="command_test",
            )
            hit = any("memory_test_probe" in r.get("text", "") for r in results)
            report.append(
                f"  召回: {'命中' if hit else '未命中'} (返回 {len(results)} 条)"
            )
        except Exception as e:
            report.append(f"  召回: 失败 ({e})")

        # 清理测试数据
        try:
            deleted = await self.memory_mgr.forget_memory(event=event, uri=uri)
            report.append(
                "  清理: 已删除测试记忆" if deleted > 0 else "  清理: 未找到测试记忆"
            )
        except Exception as e:
            report.append(f"  清理: 失败 ({e})")

        report.append(
            "  结论: " + ("全部通过" if hit else "召回异常，请检查 embedding 配置")
        )
        return "\n".join(report)

    async def _run_maintenance_test(self, stage: str) -> str:
        """运行后台阶段 dry-run，并只返回计数摘要。"""
        if not maintenance_supported():
            # 后台整理管线整体要求 4.30+，不通过临时调度器绕过版本门控
            return (
                f"[后台整理测试: {stage}]\n"
                f"  结果: 失败（{unsupported_maintenance_reason()}）\n"
                "  模式: dry-run，无持久化写入"
            )
        scheduler = self._scheduler
        if scheduler is None:
            scheduler = MaintenanceScheduler(
                context=self.context,
                memory_mgr=self.memory_mgr,
                config=self.config,
                kv_put=self.put_kv_data,
                kv_get=self.get_kv_data,
            )
        try:
            result = await scheduler.runner.run_test_stage(stage)
        except Exception as e:
            logger.exception("[简单长期记忆] 后台测试失败: stage=%s", stage)
            return f"[后台整理测试: {stage}]\n  结果: 失败 ({e})\n  模式: dry-run，无持久化写入"

        report = [f"[后台整理测试: {stage}]", "  模式: dry-run（不执行持久化写入）"]
        purge = result.get("purge")
        if isinstance(purge, dict):
            report.append(
                "  purge: 候选 {candidates}，实际清理 {purged}，失败 {failed}".format(
                    candidates=purge.get("candidates", 0),
                    purged=purge.get("purged", 0),
                    failed=purge.get("failed", 0),
                )
            )
        extract = result.get("extract")
        if isinstance(extract, dict):
            metrics = extract.get("metrics") or {}
            inbox_stats = metrics.get("inbox") or {}
            report.append(
                "  extract: 解析 {parsed}，缓冲行 {rows}，产出块 {blocks}，"
                "闸门通过 {approved}/拒绝 {rejected}/失败 {unavailable}，"
                "新增 {created}，更新 {updated}".format(
                    parsed="成功" if extract.get("parsed") else "失败",
                    rows=inbox_stats.get("rows_fetched", 0),
                    blocks=metrics.get("blocks", 0),
                    approved=metrics.get("gate_approved", 0),
                    rejected=metrics.get("gate_rejected", 0),
                    unavailable=metrics.get("gate_unavailable", 0),
                    created=metrics.get("created", 0),
                    updated=metrics.get("updated", 0),
                )
            )
        for name in ("organizer", "analyst"):
            manifest = result.get(name)
            if isinstance(manifest, dict):
                metrics = manifest.get("metrics") or {}
                parts = [
                    "  {name}: 解析 {parsed}，操作 {operations}，记忆 {memories}，"
                    "向量 {vectors}，缺失 {missing}，候选 {candidates}".format(
                        name=name,
                        parsed="成功" if manifest.get("parsed") else "失败",
                        operations=manifest.get(
                            "operations", manifest.get("ops_count", 0)
                        ),
                        memories=metrics.get("memory_count", 0),
                        vectors=metrics.get("vector_count", 0),
                        missing=metrics.get("vector_missing", 0),
                        candidates=metrics.get("candidates_screened", 0),
                    )
                ]
                if name == "organizer":
                    parts.append(
                        "巡检 {inspected}，merge {merge}，archive {archive}，"
                        "update {update}".format(
                            inspected=metrics.get("inspected", 0),
                            merge=metrics.get("op_merge", 0),
                            archive=metrics.get("op_archive", 0),
                            update=metrics.get("op_update", 0),
                        )
                    )
                else:
                    parts.append(
                        "link {link}，conflict {conflict}，merge {merge}".format(
                            link=metrics.get("op_new_link", 0),
                            conflict=metrics.get("op_contradiction", 0),
                            merge=metrics.get("op_merge", 0),
                        )
                    )
                parts.append(
                    "需核验 {needed}，实际核验 {executed}，预算跳过 {skipped}".format(
                        needed=metrics.get("verification_needed", 0),
                        executed=metrics.get("verification_executed", 0),
                        skipped=metrics.get("verification_budget_skipped", 0),
                    )
                )
                report.append("，".join(parts))
        reviewer = result.get("reviewer")
        if isinstance(reviewer, dict):
            report.append(
                "  reviewer: 裁决 {verdicts}，通过 {approved}，拒绝 {rejected}".format(
                    verdicts=reviewer.get(
                        "verdicts", result.get("reviewer_verdicts_count", 0)
                    ),
                    approved=reviewer.get("approved", 0),
                    rejected=reviewer.get("rejected", 0),
                )
            )
        elif "reviewer_verdicts_count" in result:
            report.append(f"  reviewer: 裁决 {result['reviewer_verdicts_count']}")
        llm_stats = result.get("llm_stats") or {}
        report.append(
            "  LLM: 调用 {calls}，缓存命中 {cache_hits}，错误 {errors}".format(
                calls=llm_stats.get("calls", 0),
                cache_hits=llm_stats.get("cache_hits", 0),
                errors=llm_stats.get("errors", 0),
            )
        )
        errors = result.get("errors") or []
        conclusion = (
            "完成（无阶段错误）" if not errors else f"完成（{len(errors)} 项阶段错误）"
        )
        report.append(f"  结论: {conclusion}")
        return "\n".join(report)

    # ==================== LLM 工具 ====================

    def _format_contact_records(self, records: list[dict[str, Any]], scope: str) -> str:
        """格式化已知接触记录，不包含消息正文。"""
        titles = {
            "self": "当前用户的已知会话",
            "current_group": "当前群的已知联系人",
            "user": "指定用户的已知会话",
        }
        lines = [f"[{titles[scope]}；仅为历史接触，不是完整平台通讯录]"]
        show_bot = bool(self.config.get("share_contacts_across_bots", False))
        for record in records:
            user_id = html.escape(str(record["user_id"]), quote=False)
            sender_name = html.escape(str(record["sender_name"] or ""), quote=False)
            date = str(record["last_seen_at"] or "")[:10] or "未知"
            bot_suffix = (
                f"；bot_id={html.escape(str(record['bot_id']), quote=False)}"
                if show_bot
                else ""
            )
            if scope == "current_group":
                display_name = sender_name or user_id
                lines.append(
                    f"- {display_name}（{user_id}）；最近接触 {date}{bot_suffix}"
                )
                continue
            if record["session_type"] == "group":
                label = record["group_name"] or record["group_id"] or "群聊"
            elif record["session_type"] == "private":
                label = "私聊"
            else:
                label = record["session_type"] or "其他会话"
            label = html.escape(str(label), quote=False)
            lines.append(f"- {label}；最近接触 {date}{bot_suffix}")
        return "<user_history_info>\n" + "\n".join(lines) + "\n</user_history_info>"

    @filter.llm_tool(name="memory_contacts")
    async def tool_contacts(
        self,
        event: AstrMessageEvent,
        user_id: str = "",
        scope: str = "self",
        limit: int = 10,
    ) -> str:
        """Query known contact/session observations, not complete platform membership

        Args:
            user_id(string): canonical user ID, required only when scope is user
            scope(string): self, current_group, or user
            limit(int): maximum records to return, from 1 to 20
        """
        if not self.memory_mgr:
            return "Memory plugin not initialized"
        scope = (scope or "self").lower().strip()
        valid_scopes = ("current_group", "self", "user")
        if scope not in valid_scopes:
            valid = ", ".join(valid_scopes)
            return f"Error: invalid scope '{scope}'. Valid scopes: {valid}"
        try:
            limit_value = int(limit)
        except (TypeError, ValueError):
            return f"Error: limit must be an integer between 1 and 20, got '{limit}'"
        if not 1 <= limit_value <= 20:
            return f"Error: limit must be between 1 and 20, got {limit_value}"
        user_id = (user_id or "").strip()
        if scope != "user" and user_id:
            return "Error: user_id is only allowed when scope='user'"
        if scope == "user":
            if not self.config.get("enable_admin_global_memory_tool", False):
                return (
                    "Administrator memory management tool is disabled in plugin config"
                )
            if not event.is_admin():
                return "Only administrators can query other users' contact records"
            if not user_id:
                return "Error: user_id is required when scope='user'"
        if (
            scope == "current_group"
            and not UMOInfo.parse(event.unified_msg_origin).is_group
        ):
            return "Error: scope='current_group' requires a group conversation"
        records = await self.memory_mgr.list_contact_records(
            event, scope=scope, user_id=user_id, limit=limit_value
        )
        if not records:
            return "No known contact records"
        return self._format_contact_records(records, scope)

    @filter.llm_tool(name="memory_recall")
    async def tool_recall(
        self,
        event: AstrMessageEvent,
        query: str,
        domain: str = "",
        scope: str = "",
    ) -> str:
        """Search long-term memory for relevant information

        Args:
            query(string): search keywords or question
            domain(string): optional filter by memory domain (user_profile/preferences/facts/events/context)
            scope(string): optional filter by memory scope (personal/group/conversation/global)
        """
        if not self.memory_mgr:
            return "Memory plugin not initialized"

        # 参数校验：错误即报错，不静默降级
        domain_filter: str | None = None
        if domain:
            domain_lower = domain.lower().strip()
            if domain_lower not in VALID_TOOL_DOMAINS:
                valid = ", ".join(sorted(VALID_TOOL_DOMAINS))
                return f"Error: invalid domain '{domain}'. Valid domains: {valid}"
            domain_filter = domain_lower

        scope_filter: str | None = None
        if scope:
            scope_lower = scope.lower().strip()
            if scope_lower not in (*VALID_TOOL_SCOPES, MemoryScope.GLOBAL):
                valid = ", ".join(sorted((*VALID_TOOL_SCOPES, MemoryScope.GLOBAL)))
                return f"Error: invalid scope '{scope}'. Valid scopes: {valid}"
            scope_filter = scope_lower

        memories = await self.memory_mgr.recall_memories(
            event,
            query,
            domain=domain_filter,
            memory_scope=scope_filter,
            source="tool_recall",
        )
        if not memories:
            return "No relevant memories found"

        result = format_memory_for_injection(
            memories,
            timezone_name=self._get_display_timezone(event.unified_msg_origin),
            viewer_user_id=self._get_viewer_user_id(event),
        )
        return f"<user_history_info>\n{result}\n</user_history_info>"

    @filter.llm_tool(name="memory_store")
    async def tool_store(
        self,
        event: AstrMessageEvent,
        content: str,
        memory_type: str = "fact",
        disclosure: str = "",
        importance: int = 3,
        scope: str = "personal",
    ) -> str:
        """Store information to long-term memory

        Args:
            content(string): content to remember
            memory_type(string): memory type (fact/preference/event/context)
            disclosure(string): condition description for triggering recall
            importance(int): importance level 1-5, where 5 is most important
            scope(string): memory scope (personal/group/conversation)
        """
        if not self.memory_mgr:
            return "Memory plugin not initialized"

        # 参数校验：错误即报错，不静默降级
        mem_type_lower = memory_type.lower().strip()
        if mem_type_lower not in ALLOWED_MEMORY_TYPES:
            valid = ", ".join(sorted(ALLOWED_MEMORY_TYPES))
            return f"Error: invalid memory_type '{memory_type}'. Valid types: {valid}"

        scope_lower = scope.lower().strip()
        if scope_lower == MemoryScope.GLOBAL:
            return (
                "Error: scope 'global' is not allowed here. "
                "Use the memory_store_global tool instead."
            )
        if scope_lower not in VALID_TOOL_SCOPES:
            valid = ", ".join(sorted(VALID_TOOL_SCOPES))
            return f"Error: invalid scope '{scope}'. Valid scopes: {valid}"

        try:
            importance_val = int(importance)
        except (TypeError, ValueError):
            return f"Error: importance must be an integer between 1 and 5, got '{importance}'"
        if importance_val < 1 or importance_val > 5:
            return f"Error: importance must be between 1 and 5, got {importance_val}"

        # group scope 必须在群聊中使用
        if scope_lower == MemoryScope.GROUP:
            parsed_umo = UMOInfo.parse(event.unified_msg_origin)
            if parsed_umo.session_type != "group":
                return "Error: scope 'group' can only be used in group chats"

        content = _sanitize_memory_content(content)
        if not content:
            return "Invalid memory content"

        # 准入闸门：sanitize 后、入库前统一判定；失败 fail closed 不静默降级
        gate_error = await self._gate_tool_content(event, content)
        if gate_error:
            return gate_error

        domain = normalize_domain(mem_type_lower)
        uri = str(MemoryURI.generate(domain))

        await self.memory_mgr.store_memory(
            event=event,
            content=content,
            domain=domain,
            uri=uri,
            memory_type=MemoryType.NORMAL,
            disclosure=_sanitize_memory_content(disclosure)[:200] if disclosure else "",
            importance=importance_val,
            memory_scope=scope_lower,
        )
        return f"Memory stored: {uri}"

    @filter.llm_tool(name="memory_update")
    async def tool_update(
        self,
        event: AstrMessageEvent,
        uri: str,
        content: str,
        disclosure: str = "",
    ) -> str:
        """Update an existing memory by URI, preserving its domain, importance and scope

        Args:
            uri(string): URI of the memory to update
            content(string): new content to replace the old memory
            disclosure(string): optional new condition description for triggering recall
        """
        if not self.memory_mgr:
            return "Memory plugin not initialized"

        if not uri or not uri.strip():
            return "Error: uri is required"

        # 查找旧记忆（按 user_id 隔离，找不到或不属于自己都报错）
        old_memory = await self.memory_mgr.get_memory_by_uri(
            event, uri.strip(), all_users=event.is_admin()
        )
        if not old_memory:
            return f"Error: memory not found or not yours: {uri}"

        old_meta = old_memory.get("metadata", {})

        target_user_id = old_meta.get("owner_user_id") or old_meta.get("user_id")
        if err := authorize_memory_mutation(event, target_user_id=target_user_id):
            return f"Error: {err}"

        # global 记忆仅允许管理员通过此工具修改
        if old_meta.get("memory_scope") == MemoryScope.GLOBAL and not event.is_admin():
            return "Error: global memories can only be updated by administrators."

        content = _sanitize_memory_content(content)
        if not content:
            return "Invalid memory content"

        # 使用 replace_memory 保留完整原始 metadata（租户/作用域/归属/关联）
        new_disclosure = (
            _sanitize_memory_content(disclosure)[:200] if disclosure else None
        )
        try:
            new_uri = await self.memory_mgr.replace_memory(
                old_metadata=old_meta,
                new_content=content,
                new_disclosure=new_disclosure,
                updated_by="admin" if event.is_admin() else "user",
            )
        except Exception as e:
            return f"Error: failed to update memory: {e}"

        return f"Memory updated: {uri} -> {new_uri}"

    @filter.llm_tool(name="memory_store_global")
    async def tool_store_global(
        self,
        event: AstrMessageEvent,
        content: str,
        memory_type: str = "fact",
        disclosure: str = "",
    ) -> str:
        """Store a global long-term memory visible to all chats.

        Only use this when the current user is an admin and explicitly asks to
        add a global memory, global rule, shared fact, or bot-wide preference.

        Args:
            content(string): global content to remember for all users/chats
            memory_type(string): memory type (fact/preference/event/context)
            disclosure(string): condition description for triggering recall
        """
        if not self.config.get("enable_admin_global_memory_tool", False):
            return "Global memory tool is disabled in plugin config"

        if not event.is_admin():
            return "Only administrators can store global memories"

        if not self.memory_mgr:
            return "Memory plugin not initialized"

        # 参数校验
        mem_type_lower = memory_type.lower().strip()
        if mem_type_lower not in ALLOWED_MEMORY_TYPES:
            valid = ", ".join(sorted(ALLOWED_MEMORY_TYPES))
            return f"Error: invalid memory_type '{memory_type}'. Valid types: {valid}"

        content = _sanitize_memory_content(content)
        if not content:
            return "Invalid memory content"

        domain = normalize_domain(mem_type_lower)
        uri = str(MemoryURI.generate(domain))

        await self.memory_mgr.store_memory(
            event=event,
            content=content,
            domain=domain,
            uri=uri,
            memory_type=MemoryType.PERMANENT,
            disclosure=_sanitize_memory_content(disclosure)[:200] if disclosure else "",
            importance=5,
            memory_scope=MemoryScope.GLOBAL,
            visibility="group",
            subject="global",
        )
        return f"Global memory stored: {uri}"

    @filter.llm_tool(name="memory_search_admin")
    async def tool_search_admin(
        self,
        event: AstrMessageEvent,
        query: str,
        domain: str = "",
        scope: str = "",
        top_k: int = 5,
    ) -> str:
        """Search active memories across scopes for an administrator.

        Use only for an administrator's explicit memory-management request.

        Args:
            query(string): search keywords or question
            domain(string): optional filter (user_profile/preferences/facts/events/context)
            scope(string): optional exact scope (global/personal/group/conversation)
            top_k(int): number of results, from 1 to 20
        """
        if not self.config.get("enable_admin_global_memory_tool", False):
            return "Administrator memory management tool is disabled in plugin config"
        if not event.is_admin():
            return "Only administrators can search memories across scopes"
        if not self.memory_mgr:
            return "Memory plugin not initialized"
        query = _sanitize_memory_content(query)
        if not query:
            return "Error: query is required"
        domain_filter: str | None = None
        if domain:
            domain_filter = domain.lower().strip()
            if domain_filter not in VALID_TOOL_DOMAINS:
                valid = ", ".join(sorted(VALID_TOOL_DOMAINS))
                return f"Error: invalid domain '{domain}'. Valid domains: {valid}"
        scope_filter: str | None = None
        if scope:
            scope_filter = scope.lower().strip()
            if scope_filter not in ADMIN_MANAGED_SCOPES:
                valid = ", ".join(sorted(ADMIN_MANAGED_SCOPES))
                return f"Error: invalid scope '{scope}'. Valid scopes: {valid}"
        if isinstance(top_k, bool):
            return f"Error: top_k must be an integer between 1 and 20, got '{top_k}'"
        try:
            top_k_value = int(top_k)
        except (TypeError, ValueError):
            return f"Error: top_k must be an integer between 1 and 20, got '{top_k}'"
        if not 1 <= top_k_value <= 20:
            return f"Error: top_k must be between 1 and 20, got {top_k_value}"
        memories = await self.memory_mgr.recall_memories(
            event,
            query,
            domain=domain_filter,
            top_k=top_k_value,
            all_users=True,
            memory_scope=scope_filter,
            bump=False,
            source="tool_search_admin",
        )
        if not memories:
            return "No active memories found"
        result = format_memory_for_injection(
            memories,
            timezone_name=self._get_display_timezone(event.unified_msg_origin),
            viewer_user_id=self._get_viewer_user_id(event),
        )
        targets = []
        for memory in memories:
            metadata = memory.get("metadata")
            if not isinstance(metadata, dict) or not metadata.get("uri"):
                continue
            memory_scope = str(metadata.get("memory_scope", ""))
            owner_field = ADMIN_SCOPE_OWNER_FIELDS.get(memory_scope, "")
            owner_id = str(metadata.get(owner_field, "")) if owner_field else ""
            targets.append(
                f"- uri={metadata['uri']}; scope={memory_scope}; "
                f"owner_id={owner_id or 'global'}"
            )
        targets_section = "\n".join(dict.fromkeys(targets))
        return (
            "<administrator_memory_management_results>\n"
            f"Exact management targets:\n{targets_section}\n"
            f"{result}\n</administrator_memory_management_results>"
        )

    @filter.llm_tool(name="memory_remove_admin")
    async def tool_remove_admin(
        self,
        event: AstrMessageEvent,
        uri: str,
        scope: str,
        owner_id: str = "",
        confirm: str = "",
    ) -> str:
        """Preview then remove one exact-scope memory target for an administrator.

        Args:
            uri(string): exact memory URI returned by memory_search_admin
            scope(string): exact scope (global/personal/group/conversation)
            owner_id(string): required owner_user_id, owner_session_id, or umo for non-global scopes
            confirm(string): confirmation code from the removal preview
        """
        if not self.config.get("enable_admin_global_memory_tool", False):
            return "Administrator memory management tool is disabled in plugin config"
        if not event.is_admin():
            return "Only administrators can remove memories across scopes"
        if not self.memory_mgr:
            return "Memory plugin not initialized"
        uri = (uri or "").strip()
        if not uri:
            return "Error: uri is required"
        try:
            MemoryURI.parse(uri)
        except ValueError as exc:
            return f"Error: invalid memory URI '{uri}': {exc}"
        scope_value = (scope or "").lower().strip()
        if scope_value not in ADMIN_MANAGED_SCOPES:
            valid = ", ".join(sorted(ADMIN_MANAGED_SCOPES))
            return f"Error: invalid scope '{scope}'. Valid scopes: {valid}"
        owner_field = ADMIN_SCOPE_OWNER_FIELDS.get(scope_value, "")
        owner_value = (owner_id or "").strip()
        if owner_field and not owner_value:
            return (
                f"Error: owner_id is required for scope '{scope_value}' ({owner_field})"
            )
        if not owner_field and owner_value:
            return "Error: owner_id must be empty for scope 'global'"
        preview = await self.memory_mgr.preview_admin_memory_removal(
            uri, scope_value, owner_value
        )
        if preview["count"] == 0:
            return "Error: active memory not found for the exact URI, scope, and owner"
        confirmation_code = _confirmation_code("remove-admin", preview["fingerprint"])
        if str(confirm or "").strip() != confirmation_code:
            return _format_admin_removal_preview(
                uri,
                scope_value,
                owner_value,
                preview,
                confirmation_code,
                owner_field,
            )
        outcome = await self.memory_mgr.forget_admin_memory_by_target(
            uri, scope_value, owner_value, preview["fingerprint"]
        )
        if outcome["status"] == "changed":
            current = outcome["preview"]
            current_code = _confirmation_code("remove-admin", current["fingerprint"])
            return (
                "Error: memory changed since preview; no deletion occurred.\n"
                + _format_admin_removal_preview(
                    uri,
                    scope_value,
                    owner_value,
                    current,
                    current_code,
                    owner_field,
                )
            )
        if outcome["status"] == "not_found":
            return "Error: active memory not found for the exact URI, scope, and owner"
        return f"Memory deleted: {uri} ({outcome['deleted']} record(s))"

    @filter.llm_tool(name="memory_forget")
    async def tool_forget(self, event: AstrMessageEvent, uri: str) -> str:
        """Delete a specific memory by URI

        Args:
            uri(string): memory URI identifier
        """
        if not self.memory_mgr:
            return "Memory plugin not initialized"

        deleted = await self.memory_mgr.forget_memory(
            event, uri, all_users=event.is_admin()
        )
        if deleted == 0:
            return f"Memory not found or not permitted: {uri}"
        return f"Memory deleted: {uri} ({deleted} record(s))"
