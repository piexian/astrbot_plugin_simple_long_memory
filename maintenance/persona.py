"""主人格边界摘要：审核员与整理师共用的只读 persona 读取器。

persona_mode 语义：
- manual：直接使用配置 persona_summary
- off：返回空串，不参与判定
- auto：从 context.persona_manager 读取当前默认人格并提炼有界摘要

auto 摘要在提供器实例生命周期内按 UMO 缓存，整理周期开始时由宿主清空。
"""

from __future__ import annotations

from typing import Any

from astrbot.api import logger

PERSONA_NAME_MAX_CHARS = 50
PERSONA_PROMPT_MAX_CHARS = 600


def summarize_persona(persona: Any) -> str:
    """从人格对象提炼有界的 name/prompt 摘要，不含工具与对话预设。"""
    if not isinstance(persona, dict):
        return ""
    name = str(persona.get("name", "") or "").strip()[:PERSONA_NAME_MAX_CHARS]
    prompt = " ".join(str(persona.get("prompt", "") or "").split())
    prompt = prompt[:PERSONA_PROMPT_MAX_CHARS]
    if name and prompt:
        return f"{name}：{prompt}"
    return prompt or name


class PersonaSummaryProvider:
    """周期内共享的人格摘要读取器。"""

    def __init__(self, context: Any, config: dict[str, Any]) -> None:
        self._context = context
        self._config = config
        self._cache: dict[str, str] = {}

    def clear(self) -> None:
        """清空周期内缓存（整理周期开始时调用）。"""
        self._cache.clear()

    async def get(self, umo: str = "") -> str:
        """读取人格边界摘要；manual/off 保持配置语义，auto 读真实主人格。"""
        mode = str(self._config.get("persona_mode", "auto") or "auto").strip().lower()
        if mode == "off":
            return ""
        if mode == "manual":
            return str(self._config.get("persona_summary", "") or "")
        cache_key = umo or ""
        if cache_key in self._cache:
            return self._cache[cache_key]
        summary = await self._read_default_persona(cache_key)
        self._cache[cache_key] = summary
        return summary

    async def _read_default_persona(self, umo: str) -> str:
        persona_manager = getattr(self._context, "persona_manager", None)
        if persona_manager is None:
            return ""
        try:
            persona = await persona_manager.get_default_persona_v3(umo or None)
        except Exception as e:
            logger.debug(
                "[简单长期记忆] 读取默认人格失败: error_type=%s", type(e).__name__
            )
            return ""
        return summarize_persona(persona)
