"""审核员：逐条复核整理师和分析师的操作建议。

核心职责：
1. 每次只审一条操作（单操作单调用，带磁盘缓存，同操作同模型永不重判）
2. approve / reject 每个操作
3. 对争议操作标记 controversial，触发人工升级

审核原则：
- 破坏性操作（merge/archive/update）：默认 reject，只有证据充分时才 approve
- 非破坏性操作（new_link）：默认 approve，只在明显不合理时 reject
- 不确定时选择 reject，保持现状永远比误删安全

关联操作只作为提醒列出（编号+类型+理由），不直接注入全文；
模型判断需要上下文时置 needs_context=true，宿主补入关联操作全文后重审一次。
"""

from __future__ import annotations

import json
from typing import Any

from astrbot.api import logger

from ..llm import MaintenanceLLM
from ..prompts import DEFAULT_REVIEWER_PROMPT, build_prompt


class ReviewerAgent:
    """审核员 Agent（逐条审核）。"""

    # 操作中可能引用记忆 URI 的字段
    _URI_KEYS = ("uris", "uri", "source", "target", "old_uri", "new_uri")

    def __init__(
        self,
        context: Any,
        memory_mgr: Any,
        llm: MaintenanceLLM,
        config: dict[str, Any],
    ) -> None:
        self._context = context
        self._memory_mgr = memory_mgr
        self._llm = llm
        self._config = config
        self._controversial_threshold = 0.5  # 置信度低于此值标记争议
        # 最近一次 review() 中未产出裁决的原因分布（供宿主区分有意部分评审与异常）
        self.last_unresolved: dict[str, int] = {
            "budget_exhausted": 0,
            "invalid_output": 0,
        }

    async def review(
        self,
        proposed_changes: list[dict[str, Any]],
        original_data: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """逐条复核操作建议，返回 verdicts。

        LLM 预算耗尽时提前返回；未产出裁决的操作由宿主 fail closed 转待审。

        Args:
            proposed_changes: 待审核的操作列表
            original_data: 兼容旧调用的占位参数，已不使用（源数据按操作逐条拉取）

        Returns:
            [
                {
                    "index": 0,
                    "verdict": "approve|reject",
                    "reason": "...",
                    "confidence": 0.0-1.0,
                    "controversial": bool,
                }
            ]
        """
        if not proposed_changes:
            logger.debug("[简单长期记忆] 审核员跳过: proposed_changes=0")
            return []

        logger.debug(
            "[简单长期记忆] 审核员开始: proposed_changes=%s, remaining_llm=%s",
            len(proposed_changes),
            self._llm.remaining_calls,
        )
        related_map = self._build_related_map(proposed_changes)
        persona_summary = await self._get_persona_summary()
        admin_guides = ""  # Phase 5 完善：从 KV 拉取管理员历史指引
        model_id = self._config.get(
            "maintenance_reviewer_model_id", ""
        ) or self._config.get("maintenance_model_id", "")
        resolved_model = self._llm._resolve_model_id(model_id)

        # 单次运行内的源记忆缓存：同一 URI 只拉取一次
        source_cache: dict[str, dict[str, Any] | None] = {}
        self.last_unresolved = {"budget_exhausted": 0, "invalid_output": 0}

        verdicts: list[dict[str, Any]] = []
        for index, op in enumerate(proposed_changes):
            if self._llm.remaining_calls <= 0:
                self.last_unresolved["budget_exhausted"] = len(proposed_changes) - index
                logger.warning(
                    f"[简单长期记忆] 审核员 LLM 预算耗尽，剩余 "
                    f"{len(proposed_changes) - index} 条操作未审，将 fail closed 转待审"
                )
                break
            verdict = await self._review_one(
                index=index,
                op=op,
                proposed_changes=proposed_changes,
                related_map=related_map,
                source_cache=source_cache,
                persona_summary=persona_summary,
                admin_guides=admin_guides,
                resolved_model=resolved_model,
            )
            if verdict is not None:
                verdicts.append(verdict)
            else:
                self.last_unresolved["invalid_output"] += 1
        logger.debug(
            "[简单长期记忆] 审核员完成: verdicts=%s, unresolved=%s, remaining_llm=%s",
            len(verdicts),
            self.last_unresolved,
            self._llm.remaining_calls,
        )
        return verdicts

    async def _review_one(
        self,
        *,
        index: int,
        op: dict[str, Any],
        proposed_changes: list[dict[str, Any]],
        related_map: dict[int, list[int]],
        source_cache: dict[str, dict[str, Any] | None],
        persona_summary: str,
        admin_guides: str,
        resolved_model: str,
    ) -> dict[str, Any] | None:
        """审核单条操作。返回 None 表示未产出有效裁决（宿主 fail closed）。"""
        # 磁盘缓存：同操作同模型不重判
        cache_key = self._llm.op_review_cache_key(op, resolved_model)
        cached = self._llm._cache_get(cache_key)
        if cached is not None:
            return self._normalize_verdict(index, cached)

        source_data = await self._get_op_source_data(op, source_cache)
        related_indices = related_map.get(index, [])
        parsed = await self._call_llm(
            op=op,
            source_data=source_data,
            related_indices=related_indices,
            proposed_changes=proposed_changes,
            with_related=False,
            persona_summary=persona_summary,
            admin_guides=admin_guides,
            model_id=resolved_model,
        )

        # 模型要求补充关联上下文 → 注入关联操作完整内容重审一次
        if (
            parsed
            and parsed.get("needs_context")
            and related_indices
            and self._llm.remaining_calls > 0
        ):
            parsed = await self._call_llm(
                op=op,
                source_data=source_data,
                related_indices=related_indices,
                proposed_changes=proposed_changes,
                with_related=True,
                persona_summary=persona_summary,
                admin_guides=admin_guides,
                model_id=resolved_model,
            )

        if not parsed or parsed.get("verdict") not in ("approve", "reject"):
            return None

        result = {
            "verdict": parsed["verdict"],
            "reason": str(parsed.get("reason", "")),
            "confidence": float(parsed.get("confidence", 1.0) or 0.0),
        }
        self._llm._cache_put(cache_key, result)
        return self._normalize_verdict(index, result)

    async def _call_llm(
        self,
        *,
        op: dict[str, Any],
        source_data: dict[str, Any],
        related_indices: list[int],
        proposed_changes: list[dict[str, Any]],
        with_related: bool,
        persona_summary: str,
        admin_guides: str,
        model_id: str,
    ) -> dict[str, Any] | None:
        """组装单操作 prompt 并调用 LLM，返回解析后的 JSON（失败返回 None）。"""
        variables = {
            "proposed_changes": json.dumps(op, ensure_ascii=False, indent=2),
            "original_data": json.dumps(source_data, ensure_ascii=False, indent=2),
            "related_changes": self._format_related(
                related_indices, proposed_changes, with_related
            ),
            "persona_summary": persona_summary,
            "admin_guides": admin_guides,
        }
        prompt = build_prompt(
            default_template=DEFAULT_REVIEWER_PROMPT,
            variables=variables,
            prompt_override=self._config.get(
                "maintenance_reviewer_prompt_override", ""
            ),
            prompt_extra=self._config.get("maintenance_reviewer_prompt_extra", ""),
        )
        raw = await self._llm._chat(
            system_prompt="你是记忆操作审核员，只输出结构化 JSON。",
            user_prompt=prompt,
            model_id=model_id,
        )
        if not raw:
            return None
        return self._llm._parse_json(raw)

    def _normalize_verdict(self, index: int, data: dict[str, Any]) -> dict[str, Any]:
        """补全 index 与 controversial 标记。"""
        confidence = float(data.get("confidence", 1.0) or 0.0)
        return {
            "index": index,
            "verdict": data.get("verdict"),
            "reason": data.get("reason", ""),
            "confidence": confidence,
            "controversial": confidence < self._controversial_threshold,
        }

    @classmethod
    def _op_uris(cls, op: dict[str, Any]) -> list[str]:
        """提取操作引用的全部记忆 URI。"""
        uris: list[str] = []
        for key in cls._URI_KEYS:
            val = op.get(key)
            if isinstance(val, list):
                uris.extend(u for u in val if isinstance(u, str) and u)
            elif isinstance(val, str) and val:
                uris.append(val)
        return uris

    @classmethod
    def _build_related_map(
        cls, proposed_changes: list[dict[str, Any]]
    ) -> dict[int, list[int]]:
        """按共享 URI 找出互相关联的操作：{操作下标: [关联操作下标]}。

        跨 URI 分组的关联取并集（同一条操作触及多个记忆时不能互相覆盖）；
        单条操作内重复 URI 先去重，避免自关联产生空列表。
        """
        uri_to_indices: dict[str, set[int]] = {}
        for i, op in enumerate(proposed_changes):
            for uri in set(cls._op_uris(op)):
                uri_to_indices.setdefault(uri, set()).add(i)
        related_sets: dict[int, set[int]] = {}
        for indices in uri_to_indices.values():
            if len(indices) < 2:
                continue
            for i in indices:
                related_sets.setdefault(i, set()).update(indices - {i})
        return {i: sorted(others) for i, others in related_sets.items()}

    @staticmethod
    def _format_related(
        related_indices: list[int],
        proposed_changes: list[dict[str, Any]],
        with_related: bool,
    ) -> str:
        """格式化关联操作：默认只给提醒，with_related=True 时才注入完整内容。"""
        if not related_indices:
            return "（无关联操作）"
        if with_related:
            payload = [{"index": i, **proposed_changes[i]} for i in related_indices]
            return json.dumps(payload, ensure_ascii=False, indent=2)
        lines = [
            f"- #{i} {proposed_changes[i].get('type', '')}: "
            f"{proposed_changes[i].get('reason', '')}"
            for i in related_indices
        ]
        return (
            "以下操作与本条涉及相同记忆，仅作提醒，默认不附完整内容；"
            "如确需其完整内容才能判断，请将 needs_context 置为 true：\n"
            + "\n".join(lines)
        )

    async def _get_op_source_data(
        self, op: dict[str, Any], cache: dict[str, dict[str, Any] | None]
    ) -> dict[str, Any]:
        """拉取本操作涉及的源记忆（仅本条操作，单条内容截断兜底防超长）。

        cache 为单次 review() 运行共享：同一 URI 只拉取一次，含未命中的负缓存。
        """
        memories: dict[str, Any] = {}
        for uri in self._op_uris(op)[:10]:
            if uri not in cache:
                try:
                    cache[uri] = await self._memory_mgr._get_memory_by_uri(uri)
                except Exception:
                    cache[uri] = None
            mem = cache[uri]
            if mem:
                memories[uri] = {
                    "content": str(mem.get("content", ""))[:2000],
                    "metadata": mem.get("metadata", {}),
                }
        return {"memories": memories}

    async def _get_persona_summary(self) -> str:
        """获取人格摘要。"""
        mode = self._config.get("persona_mode", "auto")
        if mode == "manual":
            return self._config.get("persona_summary", "")
        elif mode == "off":
            return ""
        else:
            # auto: 从主人格提取（Phase 5 完善）
            return ""
