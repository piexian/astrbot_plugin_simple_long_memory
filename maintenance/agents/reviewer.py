"""审核员：复核整理师和分析师的操作建议。

核心职责：
1. 复核 merge / archive / update / new_link 操作建议
2. approve / reject 每个操作
3. 对争议操作标记 controversial，触发人工升级

审核原则：
- 破坏性操作（merge/archive/update）：默认 reject，只有证据充分时才 approve
- 非破坏性操作（new_link）：默认 approve，只在明显不合理时 reject
- 不确定时选择 reject，保持现状永远比误删安全
"""

from __future__ import annotations

from typing import Any

from ..llm import MaintenanceLLM
from ..prompts import DEFAULT_REVIEWER_PROMPT, build_prompt


class ReviewerAgent:
    """审核员 Agent。"""

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

    async def review(
        self,
        proposed_changes: list[dict[str, Any]],
        original_data: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """复核操作建议，返回 verdicts。

        Args:
            proposed_changes: 待审核的操作列表
            original_data: 原始输入数据（用于上下文）

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
            return []

        # 组装 prompt
        variables = {
            "proposed_changes": self._format_changes(proposed_changes),
            "original_data": self._format_original_data(original_data),
            "persona_summary": await self._get_persona_summary(),
            "admin_guides": "",  # Phase 5 完善：从 KV 拉取管理员历史指引
        }
        prompt = build_prompt(
            default_template=DEFAULT_REVIEWER_PROMPT,
            variables=variables,
            prompt_override=self._config.get(
                "maintenance_reviewer_prompt_override", ""
            ),
            prompt_extra=self._config.get("maintenance_reviewer_prompt_extra", ""),
        )

        # 调用 LLM
        model_id = self._config.get(
            "maintenance_reviewer_model_id", ""
        ) or self._config.get("maintenance_model_id", "")
        raw = await self._llm._chat(
            system_prompt="你是记忆操作审核员，只输出结构化 JSON。",
            user_prompt=prompt,
            model_id=model_id,
        )

        verdicts: list[dict[str, Any]] = []
        if raw:
            parsed = self._llm._parse_json(raw)
            if parsed:
                verdicts = parsed.get("verdicts", [])
                # 补充 controversial 标记
                for v in verdicts:
                    confidence = v.get("confidence", 1.0)
                    if confidence < self._controversial_threshold:
                        v["controversial"] = True
                    else:
                        v["controversial"] = False

        return verdicts

    def _format_changes(self, changes: list[dict[str, Any]]) -> str:
        """格式化待审核的操作列表。"""
        import json

        return json.dumps(changes, ensure_ascii=False, indent=2)

    def _format_original_data(self, data: dict[str, Any] | None) -> str:
        """格式化原始数据。"""
        if not data:
            return "{}"
        import json

        return json.dumps(data, ensure_ascii=False, indent=2)

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
