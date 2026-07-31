"""后台整理唯一 LLM 入口。

所有后台整理 Agent 只能经由本模块调用模型，不直接访问 provider。
硬约束（对齐 Memento LLMClient）：
  - 仅后台整理模块可调用
  - 候选对先经向量预筛，再交 LLM 裁决
  - pair-hash 磁盘缓存，同对同模型永不重判
  - 三态裁决：link / merge / none / None（不可用/解析失败保持现状）
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astrbot.api import logger


@dataclass
class LLMVerdict:
    """LLM 对一对候选的裁决。

    verdict:
      "link"  —— 两条记忆语义相关，应建边
      "merge" —— 两条记忆几乎同一件事，应融合（fused_text 给出合成文本）
      "none"  —— 不相关或重复度不够，什么都不做
      None    —— LLM 不可用 / 解析失败 / 超时（调用方保守地不做改动）
    """

    verdict: str | None
    reason: str = ""
    weight: float = 0.0  # link 时建议的边权
    fused_text: str = ""  # merge 时合成的文本


class MaintenanceLLM:
    """后台整理专用 LLM 客户端，带磁盘缓存和调用统计。"""

    def __init__(
        self,
        context: Any,
        cache_dir: str | os.PathLike | None = None,
        cache_enabled: bool = True,
        default_model_id: str = "",
        max_calls_per_cycle: int = 50,
    ) -> None:
        self._context = context
        self._default_model_id = default_model_id
        self._max_calls_per_cycle = max_calls_per_cycle
        self._cache_enabled = cache_enabled
        self._calls = 0
        self._cache_hits = 0
        self._errors = 0

        # 缓存目录：默认用 AstrBot 数据目录
        if cache_dir is None:
            try:
                from astrbot.core.utils.astrbot_path import get_astrbot_data_path

                cache_dir = Path(get_astrbot_data_path()) / "plugin_data" / "simple_long_memory" / "llm_cache"
            except Exception:
                cache_dir = Path("data") / "llm_cache"
        self._cache_dir = Path(cache_dir)
        if cache_enabled:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def available(self) -> bool:
        """LLM 是否可用（有 context 即可，具体模型调用时确定）。"""
        return self._context is not None

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def errors(self) -> int:
        return self._errors

    @property
    def remaining_calls(self) -> int:
        return max(0, self._max_calls_per_cycle - self._calls)

    def reset_cycle_stats(self) -> None:
        """每个整理周期开始时重置统计。"""
        self._calls = 0
        self._cache_hits = 0
        self._errors = 0

    # ─── 缓存 ───────────────────────────────────────────────

    def _cache_key(self, text_a: str, text_b: str, task: str, model_id: str) -> str:
        """规范化对称文本对 + 模型 + 任务 → sha256。"""
        a = re.sub(r"\s+", " ", text_a).strip().lower()
        b = re.sub(r"\s+", " ", text_b).strip().lower()
        if a > b:
            a, b = b, a
        payload = {"model": model_id, "task": task, "a": a, "b": b}
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        if not self._cache_enabled:
            return None
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            path.unlink(missing_ok=True)
            return None

    def _cache_put(self, key: str, data: dict[str, Any]) -> None:
        if not self._cache_enabled:
            return
        try:
            self._cache_path(key).write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.debug(f"[简单长期记忆] LLM 缓存写入失败: {e}")

    # ─── JSON 解析 ──────────────────────────────────────────

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any] | None:
        """从 LLM 输出提取 JSON。"""
        candidate = raw.strip()
        # 尝试提取 ```json ... ``` 块
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
        if m:
            candidate = m.group(1)
        else:
            # 尝试提取第一个 { 到最后一个 }
            start = candidate.find("{")
            end = candidate.rfind("}")
            if 0 <= start < end:
                candidate = candidate[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None

    # ─── 核心调用 ───────────────────────────────────────────

    async def _chat(
        self, system_prompt: str, user_prompt: str, model_id: str = ""
    ) -> str | None:
        """调用 LLM，返回原始文本。失败返回 None。"""
        if self._calls >= self._max_calls_per_cycle:
            logger.warning(
                f"[简单长期记忆] 本周期 LLM 调用已达上限 {self._max_calls_per_cycle}，跳过"
            )
            return None

        provider_id = model_id or self._default_model_id
        if not provider_id:
            logger.debug("[简单长期记忆] 未配置整理模型，跳过 LLM 调用")
            return None

        # 在调用前计数，确保失败也计入限额
        self._calls += 1
        try:
            # AstrBot context.llm_generate 接口
            response = await self._context.llm_generate(
                chat_provider_id=provider_id,
                prompt=f"{system_prompt}\n\n{user_prompt}",
            )
            return getattr(response, "completion_text", "") or ""
        except Exception as e:
            self._errors += 1
            logger.warning(f"[简单长期记忆] LLM 调用失败: {e}")
            return None
    # ─── 任务接口 ───────────────────────────────────────────

    async def judge_relation(
        self,
        text_a: str,
        text_b: str,
        cosine: float = 0.0,
        model_id: str = "",
    ) -> LLMVerdict:
        """裁决两条记忆是 link / merge / none。

        候选对在调用前已做余弦预筛（由调用方负责），这里 LLM 只看文本。
        """
        key = self._cache_key(text_a, text_b, task="judge_relation", model_id=model_id)
        cached = self._cache_get(key)
        if cached is not None:
            self._cache_hits += 1
            return LLMVerdict(
                verdict=cached.get("verdict"),
                reason=cached.get("reason", ""),
                weight=float(cached.get("weight", 0.0)),
                fused_text=cached.get("fused_text", ""),
            )

        system = (
            "你是记忆库的离线整理员。给你两条记忆，判断它们的关系，只回 JSON。"
            "字段：verdict（'link'/'merge'/'none'）、reason（<=30字）、"
            "weight（0~1，link 时建议边权）、fused_text（merge 时合成一条简洁文本）。"
            "link=语义相关应建边；merge=几乎同一件事应合并；none=都不。"
        )
        user = (
            f"记忆A：\n{text_a}\n\n"
            f"记忆B：\n{text_b}\n\n"
            f"（向量余弦 {cosine:.3f}）\n"
            "只回 JSON。"
        )
        raw = await self._chat(system, user, model_id)
        if raw is None:
            return LLMVerdict(verdict=None, reason="LLM unavailable")

        parsed = self._parse_json(raw)
        if parsed is None:
            return LLMVerdict(verdict=None, reason="parse failed")

        verdict = LLMVerdict(
            verdict=parsed.get("verdict"),
            reason=parsed.get("reason", ""),
            weight=float(parsed.get("weight", 0.0) or 0.0),
            fused_text=parsed.get("fused_text", ""),
        )
        # 规范化 verdict
        if verdict.verdict not in {"link", "merge", "none"}:
            verdict.verdict = "none"
        self._cache_put(
            key,
            {
                "verdict": verdict.verdict,
                "reason": verdict.reason,
                "weight": verdict.weight,
                "fused_text": verdict.fused_text,
            },
        )
        return verdict

    def stats(self) -> dict[str, Any]:
        """本周期统计。"""
        return {
            "calls": self._calls,
            "cache_hits": self._cache_hits,
            "errors": self._errors,
            "remaining": self.remaining_calls,
        }
