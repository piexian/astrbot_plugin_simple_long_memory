"""整理师：去重合并、质量精炼。

核心职责：
1. 找出重复或高度相似的记忆对，建议合并（merge）
2. 找出太短、太泛、无信息量的记忆，建议归档（archive）
3. 找出措辞可以精简的记忆，建议更新（update）

merge 走 supersede 语义：不物理删除旧记忆，而是标 deprecated + 建立 supersedes 边。
"""

from __future__ import annotations

from typing import Any

from astrbot.api import logger

from ..llm import MaintenanceLLM


class OrganizerAgent:
    """整理师 Agent。"""

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
        self._merge_cosine_threshold = 0.9  # merge 候选预筛阈值（破坏性操作，阈值高）
        self._batch_size = config.get("maintenance_organizer_batch_size", 30)

    async def run(self, owner_filter: dict[str, Any] | None = None) -> dict[str, Any]:
        """运行整理师，返回 manifest。

        Args:
            owner_filter: 限定处理的用户范围（如 {"user_id": "xxx"}），None 表示全部

        Returns:
            {
                "merge": [{"uris": ["a", "b"], "merged_content": "...", "reason": "..."}],
                "archive": [{"uri": "...", "reason": "..."}],
                "update": [{"uri": "...", "new_content": "...", "reason": "..."}],
                "notes": "...",
                "candidates_screened": int,
                "llm_calls": int,
            }
        """
        manifest: dict[str, Any] = {
            "merge": [],
            "archive": [],
            "update": [],
            "notes": "",
            "candidates_screened": 0,
            "llm_calls": 0,
        }

        # 1. 拉取候选记忆（简化版：先拉取所有活跃记忆，Phase 4 完善按 owner 过滤）
        memories = await self._get_active_memories(owner_filter)
        if len(memories) < 2:
            manifest["notes"] = "记忆数量不足，无需整理"
            return manifest

        # 2. 两级预筛：向量余弦 ≥0.9 的候选对
        candidate_pairs = await self._screen_merge_candidates(memories)
        manifest["candidates_screened"] = len(candidate_pairs)

        if not candidate_pairs:
            manifest["notes"] = "预筛后无 merge 候选对"
            return manifest

        # 3. 逐对调 LLM 裁决（受调用上限约束）
        consumed_uris: set[str] = set()  # 已进入 merge 的 URI，防止重复
        for mem_a, mem_b, cosine in candidate_pairs:
            # 跳过已消费的 URI（防止 A+B, A+C, B+C 重叠合并）
            if mem_a["uri"] in consumed_uris or mem_b["uri"] in consumed_uris:
                continue
            if self._llm.remaining_calls <= 0:
                logger.warning("[简单长期记忆] 整理师 LLM 调用已达上限，跳剩余候选")
                break

            verdict = await self._llm.judge_relation(
                text_a=mem_a["content"],
                text_b=mem_b["content"],
                cosine=cosine,
                model_id=self._config.get("maintenance_model_id", ""),
            )
            manifest["llm_calls"] = self._llm.calls

            if verdict.verdict == "merge":
                consumed_uris.add(mem_a["uri"])
                consumed_uris.add(mem_b["uri"])
                manifest["merge"].append(
                    {
                        "uris": [mem_a["uri"], mem_b["uri"]],
                        "merged_content": verdict.fused_text or mem_a["content"],
                        "reason": verdict.reason,
                        "confidence": verdict.weight,
                    }
                )
            # verdict == "link" 留给分析师处理
            # verdict == "none" / None 不做任何操作

        # 4. 批量质量检查（archive/update）——简化版，Phase 5 完善
        # 目前只处理 merge，archive/update 需要更复杂的 prompt，留 Phase 5

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

    async def _screen_merge_candidates(
        self, memories: list[dict[str, Any]]
    ) -> list[tuple[dict[str, Any], dict[str, Any], float]]:
        """两级预筛：找出向量余弦 ≥ 阈值的候选对。

        Returns:
            [(mem_a, mem_b, cosine), ...] 按余弦降序
        """
        candidates: list[tuple[dict[str, Any], dict[str, Any], float]] = []

        # 提取向量（假设记忆有 vector 字段）
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

            # 收集 ≥ 阈值的对（无向，去重，按 owner 分组）
            n = len(valid_memories)
            for i in range(n):
                for j in range(i + 1, n):
                    # 按 scope 实际边界分组，避免误过滤
                    mem_a = valid_memories[i]
                    mem_b = valid_memories[j]
                    scope_a = mem_a.get("metadata", {}).get("memory_scope", "")
                    scope_b = mem_b.get("metadata", {}).get("memory_scope", "")

                    # scope 必须相同
                    if scope_a != scope_b:
                        continue

                    # 根据 scope 类型确定分组键
                    if scope_a == "personal":
                        # personal：按 owner_user_id 分组
                        owner_a = mem_a.get("metadata", {}).get("owner_user_id", "")
                        owner_b = mem_b.get("metadata", {}).get("owner_user_id", "")
                        if owner_a != owner_b:
                            continue
                    elif scope_a == "group":
                        # group：按 session 分组（session_id 含群聊标识）
                        session_a = mem_a.get("metadata", {}).get(
                            "owner_session_id", ""
                        )
                        session_b = mem_b.get("metadata", {}).get(
                            "owner_session_id", ""
                        )
                        if session_a != session_b:
                            continue
                    elif scope_a == "conversation":
                        # conversation：按完整 UMO 精确分组，避免私聊/群聊 ID 碰撞
                        umo_a = mem_a.get("metadata", {}).get("umo", "")
                        umo_b = mem_b.get("metadata", {}).get("umo", "")
                        if umo_a != umo_b:
                            continue
                    # global：无 per-user 限制，直接通过
                    cosine = float(sims[i, j])
                    if cosine >= self._merge_cosine_threshold:
                        candidates.append((mem_a, mem_b, cosine))
            # 按余弦降序
            candidates.sort(key=lambda x: x[2], reverse=True)

        except ImportError:
            logger.warning("[简单长期记忆] numpy 不可用，跳过向量预筛")
        except Exception as e:
            logger.warning(f"[简单长期记忆] 向量预筛失败: {e}")

        return candidates
