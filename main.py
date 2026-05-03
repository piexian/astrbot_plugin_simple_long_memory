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
import json
import re
import time
from typing import TYPE_CHECKING, Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star

from .memory_manager import MemoryManager, normalize_domain
from .memory_protocol import (
    MemoryScope,
    MemoryURI,
    UMOInfo,
    format_memory_for_injection,
    format_memory_for_user,
    normalize_memory_scope,
)

if TYPE_CHECKING:
    from .memory_manager import MemoryManager

from .prompts import (
    ALLOWED_MEMORY_TYPES,
    MAX_EXTRACTED_MEMORIES,
    MEMORY_EXTRACTION_PROMPT,
    RECALL_QUERY_PROMPT,
)
from .prompts import (
    sanitize_memory_content as _sanitize_memory_content,
)

DEFAULT_RECALL_QUERY_OPTIMIZATION_TIMEOUT = 10


def _sanitize_string_list(value: Any, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []

    result = []
    for item in value[:limit]:
        text = _sanitize_memory_content(str(item))[:80]
        if text:
            result.append(text)
    return result


def _normalize_extracted_scope(scope: str, session_type: str) -> str:
    scope = normalize_memory_scope(scope)
    if session_type != "group" and scope == MemoryScope.GROUP:
        return MemoryScope.PERSONAL
    return scope


def _normalize_subject_id(subject: str) -> str:
    subject = subject.strip()
    for prefix in ("用户:", "user:", "sender:"):
        if subject.lower().startswith(prefix):
            return subject[len(prefix) :].strip()
    return subject


def _normalize_subject_ids(value: Any) -> list[str]:
    raw_values = value if isinstance(value, list) else str(value).split(",")
    subjects = []
    for item in raw_values:
        subject = _normalize_subject_id(_sanitize_memory_content(str(item))[:120])
        if subject and subject not in {"current_sender", "group", "conversation"}:
            subjects.append(subject)
    return list(dict.fromkeys(subjects))


def _current_speaker_subject(event: AstrMessageEvent, scope: str) -> str:
    parsed = UMOInfo.parse(event.unified_msg_origin)
    if scope == MemoryScope.GROUP:
        return parsed.session_id
    if scope == MemoryScope.CONVERSATION:
        return event.unified_msg_origin
    return event.get_sender_id()


def _flatten_content(content: Any) -> str:
    """将内容转换为字符串"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(_flatten_content(item) for item in content)
    if isinstance(content, dict):
        return content.get("text", "") or content.get("content", "")
    return str(content)


def _normalize_contexts(contexts: Any) -> list[dict[str, Any]]:
    """标准化 contexts 为列表"""
    return list(contexts) if isinstance(contexts, list) else []


def _build_recall_query(prompt: str, contexts: list[dict[str, Any]]) -> str:
    """构建召回查询，包含 prompt 和最近的上下文"""
    parts = [prompt] if prompt else []
    for ctx in contexts[-3:]:  # 最近 3 条上下文
        role = ctx.get("role", "")
        content = _flatten_content(ctx.get("content", ""))
        if content:
            parts.append(f"[{role}]: {content}")
    return "\n".join(parts)


def _clamp_timeout(
    value: Any, default: int = DEFAULT_RECALL_QUERY_OPTIMIZATION_TIMEOUT
) -> int:
    try:
        timeout = int(value)
    except (TypeError, ValueError):
        timeout = default
    return max(1, min(20, timeout))


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
         "positional": str,
         "user_missing_value": bool, "to_missing_value": bool,
         "unknown_flags": list[str]}
    """
    result: dict[str, Any] = {
        "all": False,
        "user": "",
        "to": "",
        "clear_cache": False,
        "positional": "",
        "user_missing_value": False,
        "to_missing_value": False,
        "unknown_flags": [],
    }
    tokens = args_text.split()
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


class MemoryPlugin(Star):
    """长期记忆插件"""

    # 请求快照过期时间（秒）
    SNAPSHOT_TTL = 300  # 5 分钟

    def __init__(self, context: Context, config=None):
        super().__init__(context, config)
        self.config = config or {}
        self.memory_mgr: MemoryManager | None = None
        # 实例级请求快照字典，按 session_key 键控，避免跨用户污染
        # 结构: {session_key: {"snapshots": [...], "timestamp": float}}
        # snapshots 是一个列表，累积多轮对话
        self._request_snapshots: dict[str, dict[str, Any]] = {}
        # 每个会话的对话计数器
        self._session_counters: dict[str, int] = {}

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

        # 尝试立即连接 KB（热重载时 KB 已就绪）
        try:
            await self.memory_mgr.connect_kb()
            logger.info("[简单长期记忆] 插件初始化成功")
            if self.config.get("install_skill", False):
                self._install_skill()
        except Exception:
            # 首次启动时 KB 尚未就绪，由 on_astrbot_loaded 钩子处理
            logger.info("[简单长期记忆] 配置校验通过，等待知识库就绪")

    @filter.on_astrbot_loaded()
    async def on_loaded(self):
        if not self.memory_mgr or self.memory_mgr.is_kb_connected:
            # KB 已连接（热重载场景），但仍需检查中断恢复
            if self.memory_mgr:
                await self._recover_interrupted_rebuild()
            return
        try:
            await self.memory_mgr.connect_kb()
            logger.info("[简单长期记忆] 插件初始化成功")
        except Exception as e:
            logger.error(f"[简单长期记忆] 连接知识库失败: {e}")
            self.memory_mgr = None
            return

        if self.config.get("install_skill", False):
            self._install_skill()

        # 检测上次中断的重建：恢复缓冲写入
        await self._recover_interrupted_rebuild()

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
                    memory_records
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
                else:
                    await self.delete_kv_data("rebuild_memory_records")
        except Exception as e:
            logger.warning(f"[简单长期记忆] 恢复主数据快照失败: {e}")

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

    def _install_skill(self) -> None:
        """安装记忆 Skill 到 AstrBot skills 目录"""
        import shutil
        from pathlib import Path

        try:
            from astrbot.core.skills.skill_manager import SkillManager
        except ImportError:
            logger.warning("[简单长期记忆] 无法导入 SkillManager，跳过 Skill 安装")
            return

        source = Path(__file__).parent / "skills" / "long-term-memory" / "SKILL.md"
        if not source.exists():
            logger.warning("[简单长期记忆] SKILL.md 文件不存在，跳过安装")
            return

        try:
            sm = SkillManager()
            target_dir = Path(sm.skills_root) / "long-term-memory"
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source), str(target_dir / "SKILL.md"))
            sm.set_skill_active("long-term-memory", True)
            logger.info("[简单长期记忆] 已安装并激活记忆 Skill")
        except Exception as e:
            logger.warning(f"[简单长期记忆] Skill 安装失败: {e}")

    async def terminate(self):
        """插件销毁"""
        logger.info("[简单长期记忆] 插件已卸载")

    def _get_cmd_prefix(self) -> str:
        """从 AstrBot 配置读取命令前缀，默认 /"""
        try:
            prefixes = self.context.astrbot_config.get("wake_prefix", [])
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
        session_key = self._get_session_key(event)
        current_time = time.time()

        if session_key not in self._request_snapshots:
            self._request_snapshots[session_key] = {
                "snapshots": [],
                "pending_prompt": None,
                "timestamp": current_time,
            }

        # 存储待匹配的请求
        self._request_snapshots[session_key]["pending_prompt"] = request.prompt or ""
        self._request_snapshots[session_key]["pending_contexts"] = (
            list(request.contexts) if request.contexts else []
        )
        self._request_snapshots[session_key]["pending_sender_id"] = (
            event.get_sender_id()
        )
        self._request_snapshots[session_key]["timestamp"] = current_time

        self._cleanup_expired_snapshots()

    def _complete_snapshot_with_response(
        self, event: AstrMessageEvent, response_text: str
    ) -> None:
        """用响应完成快照（在响应阶段调用）"""
        session_key = self._get_session_key(event)
        entry = self._request_snapshots.get(session_key)

        if not entry or not entry.get("pending_prompt"):
            return

        # 创建完整的快照
        snapshot = {
            "prompt": entry["pending_prompt"],
            "contexts": entry.get("pending_contexts", []),
            "response": response_text,
            "sender_id": entry.get("pending_sender_id", event.get_sender_id()),
        }
        entry["snapshots"].append(snapshot)

        # 清除待匹配状态
        entry["pending_prompt"] = None
        entry["pending_contexts"] = []
        entry["pending_sender_id"] = ""

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

    def _increment_session_counter(self, event: AstrMessageEvent) -> int:
        """递增会话对话计数器并返回当前值"""
        key = self._get_session_key(event)
        self._session_counters[key] = self._session_counters.get(key, 0) + 1
        return self._session_counters[key]

    def _cleanup_expired_snapshots(self) -> None:
        """清理过期的请求快照"""
        current_time = time.time()
        expired_keys = [
            key
            for key, entry in self._request_snapshots.items()
            if current_time - entry.get("timestamp", 0) > self.SNAPSHOT_TTL
        ]
        for key in expired_keys:
            del self._request_snapshots[key]

    # ==================== JSON 解析辅助 ====================

    def _strip_json_fence(self, text: str) -> str:
        """移除 markdown JSON 围栏"""
        text = text.strip()
        if not text.startswith("```"):
            return text
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        return text.strip()

    def _parse_extracted_memories(
        self, text: str, session_type: str = "private"
    ) -> list[dict[str, Any]]:
        """解析 LLM 返回的记忆 JSON，带校验和上限"""
        text = self._strip_json_fence(text)
        try:
            data = json.loads(text)
            if not isinstance(data, list):
                return []

            # 校验并限制结果
            validated = []
            for item in data[:MAX_EXTRACTED_MEMORIES]:  # 限制数量
                if not isinstance(item, dict):
                    continue

                # 校验必需字段
                content = item.get("content", "")
                if not content or not isinstance(content, str):
                    continue

                # 清理内容，防止 Prompt Injection
                content = _sanitize_memory_content(content)
                if not content:
                    continue

                # 校验并规范化字段
                mem_type = str(item.get("type", "fact")).lower()
                if mem_type not in ALLOWED_MEMORY_TYPES:
                    mem_type = "fact"

                scope = _normalize_extracted_scope(
                    str(item.get("scope", "personal")), session_type
                )
                subjects = _normalize_subject_ids(
                    item.get("subjects", item.get("subject", ""))
                )
                subject = subjects[0] if subjects else ""
                if session_type == "group" and scope == MemoryScope.PERSONAL:
                    if not subjects:
                        continue
                entities = _sanitize_string_list(item.get("entities", []))
                topics = _sanitize_string_list(item.get("topics", []))
                disclosure = str(item.get("disclosure", ""))[:200]  # 限制长度

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
        except json.JSONDecodeError:
            return []

    def _build_conversation_from_snapshots(
        self, snapshots: list[dict[str, Any]]
    ) -> str:
        """从快照列表构建对话文本"""
        lines = []
        for snapshot in snapshots:
            prompt = snapshot.get("prompt", "")
            response = snapshot.get("response", "")
            sender_id = snapshot.get("sender_id", "")
            if prompt:
                sender_label = f"用户:{sender_id}" if sender_id else "用户"
                lines.append(f"[{sender_label}]: {prompt}")
            if response:
                lines.append(f"[助手]: {response}")
        return "\n".join(lines)

    # ==================== 检索优化 ====================

    async def _optimize_recall_query(
        self, event: AstrMessageEvent, raw_query: str
    ) -> str:
        """调用 LLM 从对话上下文中提炼检索关键词"""
        provider_id = await self._get_llm_provider_id(event, "extraction")
        if not provider_id:
            return raw_query

        prompt = RECALL_QUERY_PROMPT.format(context=raw_query[:1000])
        try:
            timeout = _clamp_timeout(
                self.config.get(
                    "optimize_recall_query_timeout",
                    DEFAULT_RECALL_QUERY_OPTIMIZATION_TIMEOUT,
                )
            )
            llm_response = await asyncio.wait_for(
                self.context.llm_generate(
                    provider_id=provider_id,
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

    @filter.on_llm_request()
    async def inject_memories(self, event: AstrMessageEvent, request: ProviderRequest):
        if not self.memory_mgr:
            return

        if not self.config.get("auto_memorize", True):
            return

        try:
            # 累积请求快照（等待响应后完成）
            self._accumulate_request_snapshot(event, request)

            # 构建召回查询
            contexts = _normalize_contexts(request.contexts)
            query = _build_recall_query(request.prompt or "", contexts)

            # 检索优化：调用 LLM 提炼关键词
            if self.config.get("optimize_recall_query", False):
                query = await self._optimize_recall_query(event, query)

            # 召回相关记忆
            memories = await self.memory_mgr.recall_memories(
                event=event,
                query=query,
                top_k=self.config.get("max_memories_per_inject", 5),
            )

            if memories:
                # format_memory_for_injection 已返回含 <user_context_reference> 包装的完整字符串
                safe_memory_context = format_memory_for_injection(memories)
                if safe_memory_context:
                    # 优先注入到 contexts 顶部（如果存在）
                    # 使用 user 角色而非 system，降低优先级
                    if contexts:
                        memory_msg = {"role": "user", "content": safe_memory_context}
                        request.contexts = [memory_msg] + contexts
                        logger.debug(
                            f"[简单长期记忆] 注入 {len(memories)} 条记忆到 contexts 顶部"
                        )
                    else:
                        # 回退：注入到 prompt 前面
                        request.prompt = (
                            f"{safe_memory_context}\n\n{request.prompt or ''}"
                        )
                        logger.debug(
                            f"[简单长期记忆] 注入 {len(memories)} 条记忆到 prompt 前"
                        )

        except Exception as e:
            logger.warning(f"[简单长期记忆] 注入记忆失败: {e}")

    @filter.on_llm_response()
    async def extract_memories(self, event: AstrMessageEvent, response: LLMResponse):
        if not self.memory_mgr:
            return

        if not self.config.get("auto_memorize", True):
            return

        try:
            # 获取响应文本
            assistant_output = (
                getattr(response, "completion_text", "")
                or getattr(response, "result", "")
                or ""
            )

            # 用响应完成快照
            self._complete_snapshot_with_response(event, assistant_output)

            # 递增会话对话计数器
            current_count = self._increment_session_counter(event)

            # 检查是否达到提取间隔
            extraction_interval = self.config.get("extraction_interval", 20)
            if extraction_interval <= 0:
                return
            if current_count % extraction_interval != 0:
                return

            # 获取累积的快照列表
            snapshots = self._get_and_clear_session_snapshots(event)
            if not snapshots:
                return

            # 构建对话文本（包含所有累积的对话）
            conversation = self._build_conversation_from_snapshots(snapshots)

            # 检查最小内容长度
            min_length = self.config.get("extraction_min_content_length", 150)
            if len(conversation) < min_length:
                logger.debug(
                    f"[简单长期记忆] 对话总长度 {len(conversation)} < {min_length}，跳过提取"
                )
                return

            if not conversation:
                return

            # 获取提取模型
            provider_id = await self._get_llm_provider_id(event, "extraction")
            if not provider_id:
                logger.debug("[简单长期记忆] 未配置提取模型，跳过记忆提取")
                return

            parsed_umo = UMOInfo.parse(event.unified_msg_origin)

            # 调用 LLM 提取记忆
            prompt = MEMORY_EXTRACTION_PROMPT.format(
                platform_id=parsed_umo.platform_id,
                session_type=parsed_umo.session_type,
                session_id=parsed_umo.session_id,
                sender_id=event.get_sender_id(),
                conversation=conversation,
            )
            try:
                llm_response = await self.context.llm_generate(
                    provider_id=provider_id,
                    prompt=prompt,
                )
                extraction_result = getattr(llm_response, "completion_text", "") or ""
            except Exception as e:
                logger.warning(f"[简单长期记忆] LLM 提取调用失败: {e}")
                return

            # 解析提取结果
            memories = self._parse_extracted_memories(
                extraction_result, parsed_umo.session_type
            )
            if not memories:
                return

            # 存储提取的记忆
            for mem in memories:
                mem_type = mem.get("type", "fact")
                scope = mem.get("scope", MemoryScope.PERSONAL)
                content = mem.get("content", "")
                subject = mem.get("subject", "") or _current_speaker_subject(
                    event, scope
                )
                subjects = mem.get("subjects", [])
                if not subjects and subject:
                    subjects = [subject]
                entities = mem.get("entities", [])
                topics = mem.get("topics", [])
                disclosure = mem.get("disclosure", "")
                importance = mem.get("importance", 3)
                owner_sender_ids = subjects if scope == MemoryScope.PERSONAL else []

                if not content:
                    continue

                domain = normalize_domain(mem_type)

                uri = await self.memory_mgr.store_memory(
                    event=event,
                    content=content,
                    domain=domain,
                    memory_type=mem_type,
                    disclosure=disclosure,
                    importance=importance,
                    memory_scope=scope,
                    subject=subject,
                    entities=entities,
                    topics=topics,
                    owner_sender_ids=owner_sender_ids,
                )
                logger.debug(f"[简单长期记忆] 提取并存储记忆: {uri}")

            logger.info(
                f"[简单长期记忆] 已从 {len(snapshots)} 轮对话中提取 {len(memories)} 条记忆"
            )

        except Exception as e:
            logger.warning(f"[简单长期记忆] 提取记忆失败: {e}")

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
                page = max(1, int(positional))
            except ValueError:
                pass
        page_size = 10
        memories, total = await self.memory_mgr.list_memories(
            event, page=page, page_size=page_size, all_users=all_users
        )
        scope = "全局" if all_users else "个人"
        result = format_memory_for_user(
            memories,
            page=page,
            total=total,
            page_size=page_size,
            all_mode=all_users,
            cmd_prefix=self._get_cmd_prefix(),
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
            event, query, all_users=all_users
        )
        scope = "全局" if all_users else "个人"
        result = format_memory_for_user(
            memories, total=len(memories), cmd_prefix=self._get_cmd_prefix()
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
            f"  已压缩: {stats['compressed']}"
        )
        yield event.plain_result(result)

    @memory_group.command("test")
    async def cmd_test(self, event: AstrMessageEvent):
        """测试记忆读写（管理员）/memory test"""
        if err := _ensure_initialized(self.memory_mgr):
            yield event.plain_result(err)
            return
        args_text = _parse_command_args(event, "memory test")
        if args_text:
            yield event.plain_result(f"未知参数: {args_text}")
            return
        if not event.is_admin():
            yield event.plain_result("该操作需要管理员权限")
            return
        yield event.plain_result(await self._run_memory_test(event))

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

        if target_user_id and not is_admin:
            yield event.plain_result("--user 参数仅管理员可用")
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
            deleted, owned_by_other = await self.memory_mgr.forget_memory(event, uri)
            if deleted > 0:
                yield event.plain_result(f"已删除记忆: {uri}")
            elif owned_by_other:
                yield event.plain_result("该记忆不属于你，无法删除")
            else:
                yield event.plain_result(f"未找到记忆: {uri}")

    @memory_group.command("clear")
    async def cmd_clear(self, event: AstrMessageEvent):
        """清空记忆（管理员）/memory clear [--all] [--user <id>]"""
        if err := _ensure_initialized(self.memory_mgr):
            yield event.plain_result(err)
            return
        args = _parse_memory_flags(_parse_command_args(event, "memory clear"))
        if err := _validate_command(
            event,
            args,
            cmd_name="clear",
            require_admin=True,
            allow_all=True,
            allow_user=True,
            allow_positional=False,
        ):
            yield event.plain_result(err)
            return
        if args["all"] and args["user"]:
            yield event.plain_result("--all 与 --user 不可同时使用")
            return
        if args["all"]:
            count = await self.memory_mgr.clear_memories(event, all_users=True)
            yield event.plain_result(f"已清空全部 {count} 条记忆")
        elif args["user"]:
            target_user_id = args["user"]
            count = await self.memory_mgr.clear_memories_by_user(event, target_user_id)
            yield event.plain_result(f"已清空用户 {target_user_id} 的 {count} 条记忆")
        else:
            count = await self.memory_mgr.clear_memories(event)
            yield event.plain_result(f"已清空 {count} 条记忆")

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
            result = await self.memory_mgr.clear_rebuild_cache()
            report = "[清理重建缓存]\n"
            report += f"  缓存记录数: {cache_status.get('memory_records', 0)}\n"
            report += f"  缓冲写入数: {cache_status.get('pending_writes', 0)}\n"
            report += f"  上次状态: {cache_status.get('status', '无')}\n"
            report += "  清理结果:\n"
            for key, ok in result.items():
                report += f"    {key}: {'成功' if ok else '失败'}\n"
            yield event.plain_result(report)
            return

        target_kb_name = args["to"] or None

        # --to 与当前 KB 同名时视为原地重建
        if target_kb_name and self.memory_mgr.current_kb_name == target_kb_name:
            target_kb_name = None

        if target_kb_name:
            yield event.plain_result(
                f"正在迁移记忆到知识库 '{target_kb_name}'，请稍候..."
            )
        else:
            yield event.plain_result("正在当前知识库重建所有记忆，请稍候...")

        try:
            result = await self.memory_mgr.rebuild_memories(
                target_kb_name=target_kb_name,
            )
            status = result.get("status", "completed")
            is_interrupted = status == "interrupted"

            if is_interrupted:
                # 失败路径：明确告知异常终止
                mode = "迁移" if result["is_migration"] else "重建"
                error_msg = result.get("error", "未知错误")
                report = (
                    f"[{mode}异常终止]\n"
                    f"  原因: {error_msg}\n"
                    f"  已处理: {result['success']} 条\n"
                    f"  缓冲写入: {result.get('pending_flushed', 0)} 条"
                )
                report += (
                    "\n\n  重建缓存已保留，请排查问题后重试。"
                    "\n  确认无需恢复后可执行:"
                    "\n  /memory rebuild --clear-cache"
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
                    "\n  /memory rebuild --clear-cache"
                )
            elif v and not v.get("passed") and not v.get("error"):
                report += (
                    "\n\n  数据校验未通过，请排查后重试。"
                    "重建缓存已保留，可执行:"
                    "\n  /memory rebuild --clear-cache 清理缓存"
                )

            yield event.plain_result(report)
        except ValueError as e:
            yield event.plain_result(str(e))
        except Exception as e:
            yield event.plain_result(f"重建失败: {e}")

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
                memory_type="fact",
                disclosure="测试",
                importance=1,
            )
            report.append(f"  写入: 成功 (URI: {uri})")
        except Exception as e:
            report.append(f"  写入: 失败 ({e})")
            return "\n".join(report)

        # 读取测试
        try:
            results = await self.memory_mgr.recall_memories(
                event=event, query=test_content, top_k=3
            )
            hit = any("memory_test_probe" in r.get("text", "") for r in results)
            report.append(
                f"  召回: {'命中' if hit else '未命中'} (返回 {len(results)} 条)"
            )
        except Exception as e:
            report.append(f"  召回: 失败 ({e})")

        # 清理测试数据
        try:
            await self.memory_mgr.forget_memory(event=event, uri=uri)
            report.append("  清理: 已删除测试记忆")
        except Exception as e:
            report.append(f"  清理: 失败 ({e})")

        report.append(
            "  结论: " + ("全部通过" if hit else "召回异常，请检查 embedding 配置")
        )
        return "\n".join(report)

    # ==================== LLM 工具 ====================

    @filter.llm_tool(name="memory_recall")
    async def tool_recall(self, event: AstrMessageEvent, query: str) -> str:
        """Search long-term memory for relevant information

        Args:
            query(string): search keywords or question
        """
        if not self.memory_mgr:
            return "Memory plugin not initialized"

        memories = await self.memory_mgr.recall_memories(event, query)
        if not memories:
            return "No relevant memories found"

        result = format_memory_for_injection(memories)
        return f"<user_history_info>\n{result}\n</user_history_info>"

    @filter.llm_tool(name="memory_store")
    async def tool_store(
        self,
        event: AstrMessageEvent,
        content: str,
        memory_type: str = "fact",
        disclosure: str = "",
    ) -> str:
        """Store information to long-term memory

        Args:
            content(string): content to remember
            memory_type(string): memory type (fact/preference/event/context)
            disclosure(string): condition description for triggering recall
        """
        if not self.memory_mgr:
            return "Memory plugin not initialized"

        content = _sanitize_memory_content(content)
        if not content:
            return "Invalid memory content"

        domain = normalize_domain(memory_type)
        uri = str(MemoryURI.generate(domain))

        await self.memory_mgr.store_memory(
            event=event,
            content=content,
            domain=domain,
            uri=uri,
            memory_type=memory_type,
            disclosure=disclosure[:200] if disclosure else "",
        )
        return f"Memory stored: {uri}"

    @filter.llm_tool(name="memory_forget")
    async def tool_forget(self, event: AstrMessageEvent, uri: str) -> str:
        """Delete a specific memory by URI

        Args:
            uri(string): memory URI identifier
        """
        if not self.memory_mgr:
            return "Memory plugin not initialized"

        deleted, owned_by_other = await self.memory_mgr.forget_memory(event, uri)
        if deleted == 0:
            if owned_by_other:
                return f"Cannot delete memory {uri}: it belongs to another user. Ask an admin to delete it."
            return f"Memory not found: {uri}"
        return f"Memory deleted: {uri} ({deleted} record(s))"
