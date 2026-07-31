"""分析师：关联发现、矛盾检测。

核心职责：
1. 找出记忆之间应该建立的关联（因果、场景、主题、时间）
2. 发现互相矛盾的记忆对（新的 supersede 旧的）
3. 识别对话中应该关联到已有记忆的新信息

预筛策略：余弦 ≥0.7 且排除已有关联边的对，才交 LLM 裁决。
"""

from __future__ import annotations

from typing import Any

from astrbot.api import logger

from ..llm import MaintenanceLLM


class AnalystAgent:
    """分析师 Agent。"""

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
        self._link_cosine_threshold = 0.7  # link 候选预筛阈值
        self._max_new_links = config.get("maintenance_analyst_max_new_links", 20)
        self._detect_contradiction = config.get(
            "maintenance_analyst_detect_contradiction", True
        )

    async def run(self, owner_filter: dict[str, Any] | None = None) -> dict[str, Any]:
        """运行分析师，返回 manifest。

        Args:
            owner_filter: 限定处理的用户范围

        Returns:
            {
                "new_links": [{"source": "uri_a", "target": "uri_b", "relation": "...", "reason": "..."}],
                "contradictions": [{"old_uri": "...", "new_uri": "...", "reason": "..."}],
                "notes": "...",
                "candidates_screened": int,
                "llm_calls": int,
            }
        """
        manifest: dict[str, Any] = {
            "new_links": [],
            "contradictions": [],
            "notes": "",
            "candidates_screened": 0,
            "llm_calls": 0,
        }

        # 1. 拉取活跃记忆
        memories = await self._get_active_memories(owner_filter)
        if len(memories) < 2:
            manifest["notes"] = "记忆数量不足，无需分析"
            return manifest

        # 2. 拉取对话历史（Phase 4 核心新增）
        conversation_history = await self._get_conversation_history(owner_filter)

        # 3. 两级预筛：余弦 ≥0.7 且排除已连边
        candidate_pairs = await self._screen_link_candidates(memories)
        manifest["candidates_screened"] = len(candidate_pairs)

        # 4. 逐对调 LLM 裁决（受调用上限约束）
        links_created = 0
        for mem_a, mem_b, cosine in candidate_pairs:
            if self._llm.remaining_calls <= 0 or links_created >= self._max_new_links:
                break

            verdict = await self._llm.judge_relation(
                text_a=mem_a["content"],
                text_b=mem_b["content"],
                cosine=cosine,
                model_id=self._config.get("maintenance_model_id", ""),
            )
            manifest["llm_calls"] = self._llm.calls

            if verdict.verdict == "link":
                manifest["new_links"].append(
                    {
                        "source": mem_a["uri"],
                        "target": mem_b["uri"],
                        "relation": "related",  # 默认类型，Phase 5 完善分类
                        "reason": verdict.reason,
                        "confidence": verdict.weight,
                    }
                )
                links_created += 1

        # 5. 矛盾检测（简化版：基于对话历史和记忆内容的时序分析）
        if self._detect_contradiction and conversation_history:
            contradictions = await self._detect_contradictions(
                memories, conversation_history
            )
            manifest["contradictions"] = contradictions

        return manifest

    async def _get_active_memories(
        self, owner_filter: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        """分页拉取全部活跃记忆。"""
        if not hasattr(self._memory_mgr, "get_all_active_memories"):
            logger.warning(
                "[简单长期记忆] memory_mgr 缺少 get_all_active_memories 接口"
            )
            return []
        all_memories: list[dict[str, Any]] = []
        page_size = 100
        offset = 0
        try:
            while True:
                page = await self._memory_mgr.get_all_active_memories(
                    owner_filter, limit=page_size, offset=offset
                )
                if not page:
                    break
                all_memories.extend(page)
                if len(page) < page_size:
                    break
                offset += page_size
        except Exception as e:
            logger.warning(f"[简单长期记忆] 拉取记忆列表失败: {e}")
        return all_memories

    async def _get_conversation_history(
        self, owner_filter: dict[str, Any] | None
    ) -> str:
        """拉取对话历史。

        使用 AstrBot 的 conversation_manager API，按 UMO 限定范围。
        三个维度取最严格：max_rounds / max_chars / max_age_days
        """
        # Phase 5 实现对话历史拉取时使用这些配置
        # max_rounds = self._config.get("context_max_rounds", 50)
        # max_chars = self._config.get("context_max_chars", 30000)
        # max_age_days = self._config.get("context_max_age_days", 7)

        try:
            # 检查 context 是否有 conversation_manager
            if not hasattr(self._context, "conversation_manager"):
                logger.debug(
                    "[简单长期记忆] conversation_manager 不可用，跳过对话历史拉取"
                )
                return ""

            # conv_mgr = self._context.conversation_manager

            # 按 owner_filter 限定 UMO（简化版：拉取所有对话，Phase 5 完善按 UMO 过滤）
            # 这里需要 AstrBot 提供按 UMO 查询对话的接口
            # 目前先返回空，等 AstrBot API 确认后再实现
            logger.debug("[简单长期记忆] 对话历史拉取接口待确认，暂时返回空")
            return ""

        except Exception as e:
            logger.warning(f"[简单长期记忆] 拉取对话历史失败: {e}")
            return ""

    async def _screen_link_candidates(
        self, memories: list[dict[str, Any]]
    ) -> list[tuple[dict[str, Any], dict[str, Any], float]]:
        """两级预筛：余弦 ≥ 阈值且排除已连边的候选对。

        Returns:
            [(mem_a, mem_b, cosine), ...] 按余弦降序
        """
        candidates: list[tuple[dict[str, Any], dict[str, Any], float]] = []

        # 提取向量
        vectors = []
        valid_memories = []
        for mem in memories:
            vec = mem.get("vector")
            if vec is not None:
                vectors.append(vec)
                valid_memories.append(mem)

        if len(valid_memories) < 2:
            return candidates

        # 计算余弦相似度矩阵
        try:
            import numpy as np

            vecs = np.array(vectors, dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
            sims = vecs @ vecs.T

            # 收集 ≥ 阈值的对
            n = len(valid_memories)
            for i in range(n):
                for j in range(i + 1, n):
                    # 按 scope 实际边界分组，避免跨租户配对
                    mem_a = valid_memories[i]
                    mem_b = valid_memories[j]
                    scope_a = mem_a.get("metadata", {}).get("memory_scope", "")
                    scope_b = mem_b.get("metadata", {}).get("memory_scope", "")
                    if scope_a != scope_b:
                        continue
                    if scope_a == "personal":
                        owner_a = mem_a.get("metadata", {}).get("owner_user_id", "")
                        owner_b = mem_b.get("metadata", {}).get("owner_user_id", "")
                        if owner_a != owner_b:
                            continue
                    elif scope_a == "group":
                        session_a = mem_a.get("metadata", {}).get(
                            "owner_session_id", ""
                        )
                        session_b = mem_b.get("metadata", {}).get(
                            "owner_session_id", ""
                        )
                        if session_a != session_b:
                            continue
                    elif scope_a == "conversation":
                        umo_a = mem_a.get("metadata", {}).get("umo", "")
                        umo_b = mem_b.get("metadata", {}).get("umo", "")
                        if umo_a != umo_b:
                            continue
                    # global：无 per-user 限制

                    cosine = float(sims[i, j])
                    if cosine >= self._link_cosine_threshold:
                        candidates.append((mem_a, mem_b, cosine))

            # 按余弦降序
            candidates.sort(key=lambda x: x[2], reverse=True)

        except ImportError:
            logger.warning("[简单长期记忆] numpy 不可用，跳过向量预筛")
        except Exception as e:
            logger.warning(f"[简单长期记忆] 向量预筛失败: {e}")

        # 排除已有关联边的对（避免重复提议）
        link_mgr = getattr(self._memory_mgr, "_link_manager", None)
        if link_mgr and candidates:
            filtered: list[tuple[dict[str, Any], dict[str, Any], float]] = []
            # 批量收集候选 URI 的已有边
            existing_pairs: set[tuple[str, str]] = set()
            checked_uris: set[str] = set()
            for mem_a, mem_b, _ in candidates:
                for uri in (mem_a["uri"], mem_b["uri"]):
                    if uri not in checked_uris:
                        checked_uris.add(uri)
                        try:
                            links = await link_mgr.get_links_for_uri(
                                uri, injectable_only=False, limit=0
                            )
                            for lk in links:
                                src = lk.get("source_uri", "")
                                tgt = lk.get("target_uri", "")
                                existing_pairs.add((src, tgt))
                                existing_pairs.add((tgt, src))
                        except Exception:
                            pass
            for mem_a, mem_b, cosine in candidates:
                pair = (mem_a["uri"], mem_b["uri"])
                if pair not in existing_pairs:
                    filtered.append((mem_a, mem_b, cosine))
            candidates = filtered

        return candidates

    async def _detect_contradictions(
        self, memories: list[dict[str, Any]], conversation_history: str
    ) -> list[dict[str, Any]]:
        """检测矛盾记忆对。

        简化版：基于时间戳和内容相似度，找出可能矛盾的记忆。
        Phase 5 完善：结合对话历史做更精确的时序分析。
        """
        contradictions: list[dict[str, Any]] = []

        # 简化实现：找出内容高度相似但时间戳差异较大的记忆对
        # 这些可能是"新记忆取代旧记忆"的矛盾对
        for i, mem_a in enumerate(memories):
            for mem_b in memories[i + 1 :]:
                # 检查内容是否涉及同一主题（简化版：检查关键词重叠）
                content_a = mem_a.get("content", "").lower()
                content_b = mem_b.get("content", "").lower()

                # 如果两条记忆内容高度相似，但创建时间差距大，可能是矛盾
                time_a = mem_a.get("metadata", {}).get("created_at", "")
                time_b = mem_b.get("metadata", {}).get("created_at", "")

                if time_a and time_b and time_a != time_b:
                    # 简化版：如果内容有 50% 以上的关键词重叠，标记为潜在矛盾
                    words_a = set(content_a.split())
                    words_b = set(content_b.split())
                    if words_a and words_b:
                        overlap = len(words_a & words_b) / max(
                            len(words_a), len(words_b)
                        )
                        if overlap > 0.5:
                            contradictions.append(
                                {
                                    "old_uri": mem_a["uri"]
                                    if time_a < time_b
                                    else mem_b["uri"],
                                    "new_uri": mem_b["uri"]
                                    if time_a < time_b
                                    else mem_a["uri"],
                                    "reason": f"内容高度相似（重叠度 {overlap:.2f}）但时间不同",
                                    "confidence": overlap,
                                }
                            )

        return contradictions
