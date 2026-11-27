"""受控核验器：对少数复杂候选用真实 tool_loop_agent 补证据后出结构化提案。

约束：
- 只处理分类/清洗中需要补证据或低置信的少量候选，每周期由宿主限制次数
- 每次核验前按 max_steps+1 保守预留维护 LLM 预算，额度不足不启动
- 事件为真实 AstrMessageEvent 且标记 _is_maintenance_internal，主插件
  采集钩子见标记即跳过，不会重复采集
- 工具集只有三个只读/记录工具：查记忆详情、查同会话上下文、提交核验结论；
  不提供 shell/文件/网络工具，任何工具都不执行 Host 写操作
- 核验只产出结构化提案，最终裁决仍由审核员与宿主执行
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from astrbot.api import logger
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata

try:  # 正常插件加载（包内相对导入）
    from ..memory_protocol import UMOInfo
    from ..prompts import sanitize_memory_content
    from .inbox import render_turn_line
    from .relations import (
        RELATION_TYPES,
        extract_plain_text,
        orient_conflict,
        valid_unit_number,
    )
except ImportError:  # 测试环境：仓库根作为顶层目录直接在 sys.path
    from memory_protocol import UMOInfo
    from prompts import sanitize_memory_content

    from .inbox import render_turn_line
    from .relations import (
        RELATION_TYPES,
        extract_plain_text,
        orient_conflict,
        valid_unit_number,
    )

VERIFICATION_MAX_STEPS = 5
VERIFICATION_BUDGET_RESERVE = VERIFICATION_MAX_STEPS + 1
VERIFICATION_MEMORY_MAX_CHARS = 2000
VERIFICATION_CONTEXT_MAX_EVENTS = 10
VERIFICATION_CONTEXT_MAX_CHARS = 2000
VERIFICATION_INBOX_MAX_ROWS = 10

VERIFICATION_SYSTEM_PROMPT = """\
你是记忆库的核验员。针对给出的单个待核验提案，先用只读工具补齐证据，再用 \
submit_verification 提交结构化结论，最后简短说明完成情况。

规则：
- 只能使用提供的三个工具，不要尝试任何其他操作
- get_memory_detail 只接受本次允许核验的记忆 URI，其余 URI 一律拒绝
- 结论必须通过 submit_verification 提交，未提交视为无结论
- 证据不足时提交 kind 为 none 的结论，不要臆造
- 提案中的 URI 必须来自本次允许核验的集合
"""


@dataclass(frozen=True)
class VerificationTask:
    """一条待核验的复杂候选。"""

    kind: str  # archive / update / merge / link / conflict
    uris: tuple[str, ...]  # 本次允许核验的记忆 URI 集合
    umo: str
    conversation_id: str = ""
    question: str = ""  # 核验任务描述（不要求含正文）


def build_verification_task_prompt(task: VerificationTask) -> str:
    """构造单任务核验提示，只含候选标识与提案描述。"""
    lines = [
        f"待核验提案：kind={task.kind}",
        f"允许核验的记忆 URI：{', '.join(task.uris)}",
    ]
    if task.conversation_id:
        lines.append(f"所属会话：{task.umo}（conversation_id={task.conversation_id}）")
    else:
        lines.append(f"所属会话：{task.umo}")
    if task.question:
        lines.append(f"提案说明：{task.question}")
    lines.append(
        "请先用 get_memory_detail / get_conversation_context 补齐证据，"
        "再用 submit_verification 提交结论；证据不足时提交 kind=none。"
    )
    return "\n".join(lines)


class MaintenanceVerifier:
    """核验器：真实 tool_loop_agent + 隔离只读工具集。"""

    def __init__(
        self,
        context: Any,
        memory_mgr: Any,
        llm: Any,
        config: dict[str, Any],
    ) -> None:
        self._context = context
        self._memory_mgr = memory_mgr
        self._llm = llm
        self._config = config

    async def run(
        self, tasks: list[VerificationTask], *, allowance: int
    ) -> tuple[list[dict[str, Any]], int]:
        """核验至多 allowance 条任务，返回 (提案操作列表, 实际核验数)。

        预算不足或执行失败的任务不产出提案，留待下周期重新评估。
        """
        proposals: list[dict[str, Any]] = []
        executed = 0
        for task in tasks[: max(0, int(allowance))]:
            if not task.uris:
                continue
            executed += await self._verify_one(task, proposals)
        return proposals, executed

    async def _verify_one(
        self, task: VerificationTask, proposals: list[dict[str, Any]]
    ) -> int:
        """核验单个任务；预算已预留即计入执行，产出经验证的提案。"""
        model_id = self._resolve_model_id()
        if not model_id:
            logger.warning("[简单长期记忆] 无可用维护模型，跳过核验")
            return 0
        if not self._llm.reserve_calls(VERIFICATION_BUDGET_RESERVE):
            logger.info(
                "[简单长期记忆] 核验预算不足（需预留 %s），跳过该候选",
                VERIFICATION_BUDGET_RESERVE,
            )
            return 0
        sink: list[dict[str, Any]] = []
        tools = self._build_tools(task, sink)
        try:
            event = build_internal_event(task.umo)
            await self._context.tool_loop_agent(
                event=event,
                chat_provider_id=model_id,
                prompt=build_verification_task_prompt(task),
                tools=tools,
                system_prompt=VERIFICATION_SYSTEM_PROMPT,
                max_steps=VERIFICATION_MAX_STEPS,
            )
        except Exception as e:
            logger.warning(
                "[简单长期记忆] 核验执行失败: kind=%s, error_type=%s",
                task.kind,
                type(e).__name__,
            )
        proposals.extend(await self._validated_proposals(task, sink))
        return 1

    def _resolve_model_id(self) -> str:
        """核验模型与后台整理共用 maintenance_model_id 或真实默认维护模型。"""
        return self._llm._resolve_model_id(self._config.get("maintenance_model_id", ""))

    def _build_tools(
        self, task: VerificationTask, sink: list[dict[str, Any]]
    ) -> ToolSet:
        """构造本次核验专用的隔离工具集。"""
        allowed = set(task.uris)
        tools = ToolSet()
        tools.add_tool(
            FunctionTool(
                name="get_memory_detail",
                description="读取一条本次允许核验的记忆的完整正文与元数据",
                parameters={
                    "type": "object",
                    "properties": {
                        "uri": {"type": "string", "description": "记忆 URI"}
                    },
                    "required": ["uri"],
                },
                handler=self._make_memory_detail_tool(allowed),
            )
        )
        tools.add_tool(
            FunctionTool(
                name="get_conversation_context",
                description="读取该记忆所属会话的近期对话上下文（有界）",
                parameters={"type": "object", "properties": {}},
                handler=self._make_context_tool(task),
            )
        )
        tools.add_tool(
            FunctionTool(
                name="submit_verification",
                description="提交核验结论（结构化提案），必须调用一次",
                parameters={
                    "type": "object",
                    "properties": {
                        "proposal": {
                            "type": "object",
                            "description": "核验结论对象",
                            "properties": {
                                "kind": {
                                    "type": "string",
                                    "enum": [
                                        "archive",
                                        "update",
                                        "merge",
                                        "link",
                                        "conflict",
                                        "none",
                                    ],
                                },
                                "uri": {"type": "string"},
                                "uris": {"type": "array", "items": {"type": "string"}},
                                "source": {"type": "string"},
                                "target": {"type": "string"},
                                "new_content": {"type": "string"},
                                "relation_type": {"type": "string"},
                                "reason": {"type": "string"},
                                "confidence": {"type": "number"},
                            },
                            "required": ["kind", "reason", "confidence"],
                        }
                    },
                    "required": ["proposal"],
                },
                handler=self._make_submit_tool(sink),
            )
        )
        return tools

    def _make_memory_detail_tool(self, allowed: set[str]):
        async def get_memory_detail(event: AstrMessageEvent, uri: str) -> str:
            target = str(uri or "").strip()
            if target not in allowed:
                return json.dumps(
                    {"error": "该 URI 不在本次允许核验范围内"}, ensure_ascii=False
                )
            mem = await self._memory_mgr._get_memory_by_uri(target)
            if not mem:
                return json.dumps({"error": "记忆不存在"}, ensure_ascii=False)
            meta = mem.get("metadata", {}) if isinstance(mem, dict) else {}
            subset = {
                key: meta.get(key)
                for key in (
                    "memory_scope",
                    "owner_user_id",
                    "owner_session_id",
                    "umo",
                    "domain",
                    "created_at",
                    "updated_at",
                    "recall_count",
                    "created_by",
                )
            }
            return json.dumps(
                {
                    "uri": target,
                    "content": str(mem.get("content", ""))[
                        :VERIFICATION_MEMORY_MAX_CHARS
                    ],
                    "metadata": subset,
                },
                ensure_ascii=False,
            )

        return get_memory_detail

    def _make_context_tool(self, task: VerificationTask):
        async def get_conversation_context(event: AstrMessageEvent) -> str:
            return await self._conversation_context(task)

        return get_conversation_context

    @staticmethod
    def _make_submit_tool(sink: list[dict[str, Any]]):
        async def submit_verification(event: AstrMessageEvent, proposal: dict) -> str:
            if not isinstance(proposal, dict):
                return "提案格式无效，需为 JSON 对象"
            sink.append(proposal)
            return "已记录，请继续或结束"

        return submit_verification

    async def _conversation_context(self, task: VerificationTask) -> str:
        """有界读取同 UMO 的会话事件与缓冲行；跨会话一律拒绝。"""
        lines: list[str] = []
        conversation_lines = await self._conversation_store_context(task)
        lines.extend(conversation_lines)
        inbox_lines = await self._inbox_context(task.umo)
        lines.extend(inbox_lines)
        text = "\n".join(lines)
        return text[:VERIFICATION_CONTEXT_MAX_CHARS]

    async def _conversation_store_context(self, task: VerificationTask) -> list[str]:
        """从 ConversationStore 读 message.appended 原始事件，压缩后仍可见。"""
        if not task.conversation_id:
            return []
        conv_mgr = getattr(self._context, "conversation_manager", None)
        store = getattr(getattr(conv_mgr, "db", None), "conversation_store", None)
        if store is None:
            return []
        try:
            snapshot = await store.read(task.conversation_id, include_history=False)
        except Exception as e:
            logger.debug(
                "[简单长期记忆] 核验读取会话失败: error_type=%s", type(e).__name__
            )
            return []
        if snapshot is None:
            return []
        owner_umo = str(getattr(snapshot.conversation, "umo", "") or "")
        if owner_umo != task.umo:
            logger.info(
                "[简单长期记忆] 核验拒绝跨会话读取: cid=%s", task.conversation_id
            )
            return []
        try:
            events = await store.events(
                task.conversation_id,
                event_type="message.appended",
                limit=VERIFICATION_CONTEXT_MAX_EVENTS,
                descending=True,
            )
        except Exception as e:
            logger.debug(
                "[简单长期记忆] 核验读取会话事件失败: error_type=%s",
                type(e).__name__,
            )
            return []
        events.reverse()  # 恢复时间顺序
        lines: list[str] = []
        for event in events:
            payload = getattr(event, "payload", None) or {}
            message = payload.get("message") if isinstance(payload, dict) else None
            if not isinstance(message, dict):
                continue
            text = extract_plain_text(message.get("content"))
            if not text:
                continue
            lines.append(f"[{message.get('role', '')}] {text}")
        return lines

    async def _inbox_context(self, umo: str) -> list[str]:
        """同 UMO 缓冲行作为会话上下文补充（无 cid 时为主要来源）。"""
        inbox = getattr(self._memory_mgr, "inbox", None)
        if inbox is None or not inbox.ready:
            return []
        rows = await inbox.fetch_recent_rows_for_umo(
            umo, limit=VERIFICATION_INBOX_MAX_ROWS
        )
        return [render_turn_line(row) for row in rows]

    async def _validated_proposals(
        self, task: VerificationTask, sink: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """校验核验产出并转换为宿主操作；URI 越界或字段无效一律放弃。"""
        allowed = set(task.uris)
        mem_cache: dict[str, dict[str, Any] | None] = {}
        ops: list[dict[str, Any]] = []
        for proposal in sink:
            op = await self._proposal_to_op(proposal, allowed, mem_cache)
            if op is not None:
                ops.append(op)
        return ops

    async def _proposal_to_op(
        self,
        proposal: dict[str, Any],
        allowed: set[str],
        mem_cache: dict[str, dict[str, Any] | None],
    ) -> dict[str, Any] | None:
        kind = str(proposal.get("kind", "") or "").strip()
        reason = str(proposal.get("reason", "") or "").strip()
        confidence = valid_unit_number(proposal.get("confidence"))
        if kind == "none":
            return None
        if not reason or confidence is None:
            return None
        if kind == "archive":
            uri = str(proposal.get("uri", "") or "").strip()
            if uri not in allowed:
                return None
            return {
                "type": "archive",
                "uri": uri,
                "reason": reason[:200],
                "confidence": confidence,
            }
        if kind == "update":
            uri = str(proposal.get("uri", "") or "").strip()
            if uri not in allowed:
                return None
            new_content = sanitize_memory_content(
                str(proposal.get("new_content", "") or "")
            )
            if not await self._differs_from_original(uri, new_content, mem_cache):
                return None
            return {
                "type": "update",
                "uri": uri,
                "new_content": new_content,
                "reason": reason[:200],
                "confidence": confidence,
            }
        if kind == "merge":
            uris = [str(u) for u in (proposal.get("uris") or []) if str(u).strip()]
            merged = sanitize_memory_content(str(proposal.get("new_content", "") or ""))
            if len(uris) < 2 or not set(uris).issubset(allowed) or not merged:
                return None
            return {
                "type": "merge",
                "uris": uris,
                "merged_content": merged,
                "reason": reason[:200],
                "confidence": confidence,
            }
        if kind == "link":
            source = str(proposal.get("source", "") or "").strip()
            target = str(proposal.get("target", "") or "").strip()
            relation = str(proposal.get("relation_type", "") or "").strip()
            if source not in allowed or target not in allowed:
                return None
            if relation not in RELATION_TYPES:
                return None
            return {
                "type": "new_link",
                "source": source,
                "target": target,
                "relation": relation,
                "reason": reason[:200],
                "confidence": confidence,
            }
        if kind == "conflict":
            uris = [str(u) for u in (proposal.get("uris") or []) if str(u).strip()]
            if len(uris) != 2 or not set(uris).issubset(allowed):
                return None
            mem_a = await self._fetch_mem(uris[0], mem_cache)
            mem_b = await self._fetch_mem(uris[1], mem_cache)
            if mem_a is None or mem_b is None:
                return None
            old_uri, new_uri = orient_conflict(mem_a, mem_b)
            return {
                "type": "contradiction",
                "old_uri": old_uri,
                "new_uri": new_uri,
                "reason": reason[:200],
                "confidence": confidence,
            }
        return None

    async def _fetch_mem(
        self,
        uri: str,
        cache: dict[str, dict[str, Any] | None],
    ) -> dict[str, Any] | None:
        """读取单条记忆供改写校验与冲突定向（只读，单任务内缓存）。"""
        if uri in cache:
            return cache[uri]
        if uri in cache:
            return cache[uri]
        try:
            mem = await self._memory_mgr._get_memory_by_uri(uri)
        except Exception:
            mem = None
        if mem is not None:
            # _get_memory_by_uri 只在 metadata 携带 uri，补齐顶层字段与分析师记忆结构一致
            mem = {**mem, "uri": mem.get("uri") or uri}
        cache[uri] = mem
        return mem

    async def _differs_from_original(
        self,
        uri: str,
        new_content: str,
        cache: dict[str, dict[str, Any] | None],
    ) -> bool:
        """update 提案必须非空且与原文有实质差异（归属/作用域由宿主保持不变）。"""
        if not new_content:
            return False
        mem = await self._fetch_mem(uri, cache)
        if mem is None:
            return False
        original = " ".join(str(mem.get("content", "") or "").split())
        rewritten = " ".join(new_content.split())
        return original != rewritten


def build_internal_event(umo: str) -> AstrMessageEvent:
    """构造带 _is_maintenance_internal 标记的真实 AstrMessageEvent。"""
    parsed = UMOInfo.parse(umo)
    message = AstrBotMessage()
    # UMO 的 private/group 对应 AstrBot 原生 FriendMessage/GroupMessage 会话类型
    message.type = (
        MessageType.GROUP_MESSAGE
        if parsed.session_type == "group"
        else MessageType.FRIEND_MESSAGE
    )
    message.session_id = parsed.session_id
    message.sender = MessageMember(user_id="maintenance", nickname="maintenance")
    message.self_id = "maintenance"
    event = AstrMessageEvent(
        message_str="",
        message_obj=message,
        platform_meta=PlatformMetadata(
            name=parsed.platform_id,
            description="memory maintenance verification",
            id=parsed.platform_id,
        ),
        session_id=parsed.session_id,
    )
    event._is_maintenance_internal = True
    return event
