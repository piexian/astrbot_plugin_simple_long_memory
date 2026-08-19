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
        self._max_contradictions = config.get(
            "maintenance_analyst_max_contradictions", 20
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
            "memory_count": 0,
            "vector_count": 0,
            "vector_missing": 0,
            "conversation_chars": 0,
        }

        # 1. 拉取活跃记忆
        memories = await self._get_active_memories(owner_filter)
        manifest["memory_count"] = len(memories)
        manifest["vector_count"] = sum(
            mem.get("vector") is not None for mem in memories
        )
        manifest["vector_missing"] = manifest["memory_count"] - manifest["vector_count"]
        logger.debug(
            "[简单长期记忆] 分析师记忆输入: memories=%s, vectors=%s, missing=%s",
            manifest["memory_count"],
            manifest["vector_count"],
            manifest["vector_missing"],
        )
        if len(memories) < 2:
            manifest["notes"] = "记忆数量不足，无需分析"
            return manifest

        # 2. 拉取对话历史（Phase 4 核心新增）
        conversation_history = await self._get_conversation_history(owner_filter)
        manifest["conversation_chars"] = len(conversation_history)
        logger.debug(
            "[简单长期记忆] 分析师对话输入: chars=%s",
            manifest["conversation_chars"],
        )
        # 3. 两级预筛：余弦 ≥0.7 且排除已连边
        candidate_pairs = await self._screen_link_candidates(memories)
        manifest["candidates_screened"] = len(candidate_pairs)
        logger.debug(
            "[简单长期记忆] 分析师预筛完成: threshold=%.2f, candidates=%s",
            self._link_cosine_threshold,
            manifest["candidates_screened"],
        )

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

        # 5. 矛盾检测（基于预筛候选对，排除已提议建边的对，受数量上限约束）
        if self._detect_contradiction and conversation_history:
            linked_pairs = {
                (lnk["source"], lnk["target"]) for lnk in manifest["new_links"]
            }
            manifest["contradictions"] = self._detect_contradictions(
                candidate_pairs, exclude_pairs=linked_pairs
            )
        logger.debug(
            "[简单长期记忆] 分析师完成: candidates=%s, links=%s, contradictions=%s, "
            "llm_calls=%s",
            manifest["candidates_screened"],
            len(manifest["new_links"]),
            len(manifest["contradictions"]),
            manifest["llm_calls"],
        )
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

        try:
            if not hasattr(self._context, "conversation_manager"):
                logger.debug(
                    "[简单长期记忆] conversation_manager 不可用，跳过对话历史拉取"
                )
                return ""

            import json as _json

            max_rounds = self._config.get("context_max_rounds", 50)
            max_chars = self._config.get("context_max_chars", 30000)

            conv_mgr = self._context.conversation_manager
            # 按 UMO 限定范围（owner_filter 中可能有 umo）
            umo = (owner_filter or {}).get("umo")
            conversations = await conv_mgr.get_conversations(unified_msg_origin=umo)

            lines: list[str] = []
            total_chars = 0
            total_rounds = 0
            for conv in conversations:
                history_raw = getattr(conv, "history", None)
                if not history_raw:
                    continue
                try:
                    history = (
                        _json.loads(history_raw)
                        if isinstance(history_raw, str)
                        else history_raw
                    )
                except Exception:
                    continue
                if not isinstance(history, list):
                    continue
                for entry in reversed(history):  # 最新的在前
                    if total_rounds >= max_rounds or total_chars >= max_chars:
                        break
                    role = entry.get("role", "")
                    text = entry.get("content", "")
                    if not text or role not in ("user", "assistant"):
                        continue
                    line = f"[{role}]: {text}"
                    if total_chars + len(line) > max_chars:
                        break
                    lines.append(line)
                    total_chars += len(line)
                    total_rounds += 1
                if total_rounds >= max_rounds or total_chars >= max_chars:
                    break

            lines.reverse()  # 恢复时间顺序
            result = "\n".join(lines)
            logger.debug(
                f"[简单长期记忆] 拉取对话历史: {total_rounds} 轮, {total_chars} 字符"
            )
            return result

        except Exception as e:
            logger.warning(f"[简单长期记忆] 拉取对话历史失败: {e}")
            return ""

    @staticmethod
    def _same_scope_group(mem_a: dict[str, Any], mem_b: dict[str, Any]) -> bool:
        """两条记忆是否属于同一 scope 分组（link/矛盾配对共用的租户隔离规则）。"""
        meta_a = mem_a.get("metadata", {})
        meta_b = mem_b.get("metadata", {})
        scope_a = meta_a.get("memory_scope", "")
        if scope_a != meta_b.get("memory_scope", ""):
            return False
        if scope_a == "personal":
            return meta_a.get("owner_user_id", "") == meta_b.get("owner_user_id", "")
        if scope_a == "group":
            return meta_a.get("owner_session_id", "") == meta_b.get(
                "owner_session_id", ""
            )
        if scope_a == "conversation":
            return meta_a.get("umo", "") == meta_b.get("umo", "")
        # global：无 per-user 限制
        return True

    async def _screen_link_candidates(
        self, memories: list[dict[str, Any]]
    ) -> list[tuple[dict[str, Any], dict[str, Any], float]]:
        """两级预筛：余弦 ≥ 阈值且排除已连边的候选对。

        Returns:
            [(mem_a, mem_b, cosine), ...] 按余弦降序
        """
        candidates: list[tuple[dict[str, Any], dict[str, Any], float]] = []

        # 按 batch_size 截断，避免大 KB 分配巨大相似度矩阵
        batch_size = self._config.get("maintenance_analyst_batch_size", 200)
        if len(memories) > batch_size:
            import random

            memories = random.sample(memories, batch_size)

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
                    # 按 scope 实际边界分组，避免跨租户配对
                    if not self._same_scope_group(mem_a, mem_b):
                        continue
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

    def _detect_contradictions(
        self,
        candidate_pairs: list[tuple[dict[str, Any], dict[str, Any], float]],
        exclude_pairs: set[tuple[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        """从预筛候选对中检测矛盾记忆对。

        输入已是同 scope 分组、余弦达标、未连边的候选对，不再做 O(n²) 裸配对。
        简化版启发式：内容关键词高度重叠但创建时间不同 → 可能是新记忆取代旧记忆。
        数量受 maintenance_analyst_max_contradictions 上限约束。
        Phase 5 完善：结合对话历史做更精确的时序分析。
        """
        exclude_pairs = exclude_pairs or set()
        contradictions: list[dict[str, Any]] = []

        for mem_a, mem_b, _cosine in candidate_pairs:
            if len(contradictions) >= self._max_contradictions:
                break
            pair = (mem_a["uri"], mem_b["uri"])
            if pair in exclude_pairs or (pair[1], pair[0]) in exclude_pairs:
                continue

            time_a = mem_a.get("metadata", {}).get("created_at", "")
            time_b = mem_b.get("metadata", {}).get("created_at", "")
            if not time_a or not time_b or time_a == time_b:
                continue

            # 内容关键词重叠超过一半 → 疑似同一事实的新旧两个版本
            words_a = set(mem_a.get("content", "").lower().split())
            words_b = set(mem_b.get("content", "").lower().split())
            if not words_a or not words_b:
                continue
            overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
            if overlap <= 0.5:
                continue

            contradictions.append(
                {
                    "old_uri": mem_a["uri"] if time_a < time_b else mem_b["uri"],
                    "new_uri": mem_b["uri"] if time_a < time_b else mem_a["uri"],
                    "reason": f"内容高度相似（重叠度 {overlap:.2f}）但时间不同",
                    "confidence": overlap,
                }
            )

        return contradictions
