"""分析师：中文语义关系分类。

核心职责：
1. 对同 scope/owner、余弦预筛、已连边排除后的有界候选对做
   merge / link / conflict / none 四分类
2. link 输出真实 relation_type（supports/supersedes/context/related）
3. conflict 按 created_at/updated_at 定向后进入既有矛盾人工裁决

分类优先走真实 Decision（choose 语义的 DecisionChoiceQuestion，经
MaintenanceLLM 受控计数）；未配置或单次失败时走 maintenance_model_id
的批量结构化 Chat JSON 降级。中文语义不再依赖空格分词重叠。
"""

from __future__ import annotations

import json
from typing import Any

from astrbot.api import logger

try:  # AstrBot 4.30+ 提供 Decision Choice 类型
    from astrbot.api.provider import DecisionChoiceAnswer, DecisionChoiceQuestion
except ImportError:  # AstrBot 4.29 仅走 Chat 降级
    DecisionChoiceAnswer = None  # type: ignore[assignment]
    DecisionChoiceQuestion = None  # type: ignore[assignment]

from ..llm import MaintenanceLLM
from ..memory_gate import _DECISION_PROVIDER_TYPE
from ..prompts import ANALYST_CLASSIFY_PROMPT, build_prompt
from ..relations import (
    PAIR_VERDICTS,
    RELATION_TYPES,
    orient_conflict,
    valid_unit_number,
)
from ..verifier import VerificationTask

try:  # 正常插件加载（包内相对导入）
    from ..prompts import sanitize_memory_content
except ImportError:  # 测试环境：仓库根作为顶层目录直接在 sys.path
    from prompts import sanitize_memory_content

_CLASSIFY_SYSTEM_PROMPT = "你是记忆库的关系分析师，只输出结构化 JSON。"

_CLASSIFY_CHUNK_SIZE = 10
"""Chat 降级路径每批分类的候选对数上限，约束单次 prompt 体积。"""


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
        self._max_contradictions = config.get(
            "maintenance_analyst_max_contradictions", 20
        )
        self._max_merges = config.get("maintenance_analyst_max_merges", 5)
        self._verify_confidence_threshold = float(
            config.get("maintenance_verification_confidence_threshold", 0.6)
        )
        # Decision provider 未配置时缓存不可用状态，本周期直接走 Chat 降级
        self._decision_unavailable = False

    async def run(
        self,
        owner_filter: dict[str, Any] | None = None,
        *,
        exclude_uris: set[str] | None = None,
    ) -> dict[str, Any]:
        """运行分析师，返回 manifest。

        Args:
            owner_filter: 限定处理的用户范围
            exclude_uris: 已被其他角色消费的 URI（如整理师 merge 对），跳过相关提案

        Returns:
            {
                "new_links": [{"source": "uri_a", "target": "uri_b", "relation": "...", "reason": "..."}],
                "contradictions": [{"old_uri": "...", "new_uri": "...", "reason": "..."}],
                "merges": [{"uris": [...], "merged_content": "...", "reason": "..."}],
                "notes": "...",
                "candidates_screened": int,
                "llm_calls": int,
                "verification_tasks": [VerificationTask, ...],
            }
        """
        manifest: dict[str, Any] = {
            "new_links": [],
            "contradictions": [],
            "merges": [],
            "notes": "",
            "candidates_screened": 0,
            "llm_calls": 0,
            "memory_count": 0,
            "vector_count": 0,
            "vector_missing": 0,
            "decision_calls": 0,
            "chat_fallback_calls": 0,
            "verification_tasks": [],
        }
        excluded = set(exclude_uris or set())

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

        # 2. 两级预筛：余弦 >=0.7 且排除已连边（同 scope/owner 分组）
        candidate_pairs = await self._screen_link_candidates(memories)
        manifest["candidates_screened"] = len(candidate_pairs)
        logger.debug(
            "[简单长期记忆] 分析师预筛完成: threshold=%.2f, candidates=%s",
            self._link_cosine_threshold,
            manifest["candidates_screened"],
        )

        # 3. 逐对分类：Decision 优先，失败的对收集进 Chat 降级批量
        consumed_uris: set[str] = set()
        fallback_pairs: list[tuple[dict[str, Any], dict[str, Any], float]] = []
        decision_ready = await self._decision_ready()
        for mem_a, mem_b, cosine in candidate_pairs:
            if self._caps_reached(manifest):
                break
            if self._llm.remaining_calls <= 0:
                logger.info("[简单长期记忆] 分析师 LLM 预算耗尽，剩余候选留待下周期")
                break
            if mem_a["uri"] in consumed_uris or mem_b["uri"] in consumed_uris:
                continue
            if mem_a["uri"] in excluded or mem_b["uri"] in excluded:
                continue
            classification = None
            if decision_ready:
                classification = await self._classify_by_decision(mem_a, mem_b, cosine)
                manifest["decision_calls"] += 1
            if classification is None:
                fallback_pairs.append((mem_a, mem_b, cosine))
                continue
            self._apply_classification(manifest, mem_a, mem_b, classification)
            consumed_uris.update((mem_a["uri"], mem_b["uri"]))
            manifest["llm_calls"] = self._llm.calls

        # 4. Chat 降级：批量结构化 JSON 分类（Decision 未配置或单次失败的对）
        for chunk_start in range(0, len(fallback_pairs), _CLASSIFY_CHUNK_SIZE):
            if self._caps_reached(manifest) or self._llm.remaining_calls <= 0:
                break
            chunk = fallback_pairs[chunk_start : chunk_start + _CLASSIFY_CHUNK_SIZE]
            classifications = await self._classify_by_chat(chunk)
            manifest["chat_fallback_calls"] += 1
            manifest["llm_calls"] = self._llm.calls
            for (mem_a, mem_b, _cosine), classification in zip(chunk, classifications):
                if classification is None:
                    continue
                if self._caps_reached(manifest):
                    break
                if mem_a["uri"] in consumed_uris or mem_b["uri"] in consumed_uris:
                    continue
                self._apply_classification(manifest, mem_a, mem_b, classification)
                consumed_uris.update((mem_a["uri"], mem_b["uri"]))

        logger.debug(
            "[简单长期记忆] 分析师完成: candidates=%s, links=%s, contradictions=%s, "
            "merges=%s, decision=%s, chat_fallback=%s, llm_calls=%s",
            manifest["candidates_screened"],
            len(manifest["new_links"]),
            len(manifest["contradictions"]),
            len(manifest["merges"]),
            manifest["decision_calls"],
            manifest["chat_fallback_calls"],
            manifest["llm_calls"],
        )
        return manifest

    def _caps_reached(self, manifest: dict[str, Any]) -> bool:
        """三类产物的周期上限全部达到时停止评估新候选对。"""
        return (
            len(manifest["new_links"]) >= self._max_new_links
            and len(manifest["contradictions"]) >= self._max_contradictions
            and len(manifest["merges"]) >= self._max_merges
        )

    def _apply_classification(
        self,
        manifest: dict[str, Any],
        mem_a: dict[str, Any],
        mem_b: dict[str, Any],
        classification: dict[str, Any],
    ) -> None:
        """应用单对分类结果；低于置信阈值的复杂提案转核验而非直接执行。"""
        verdict = classification["verdict"]
        if verdict == "link":
            if len(manifest["new_links"]) >= self._max_new_links:
                return
            op = {
                "type": "new_link",
                "source": mem_a["uri"],
                "target": mem_b["uri"],
                "relation": classification["relation_type"],
                "reason": classification["reason"],
                "confidence": classification["confidence"],
            }
        elif verdict == "conflict":
            if len(manifest["contradictions"]) >= self._max_contradictions:
                return
            old_uri, new_uri = orient_conflict(mem_a, mem_b)
            op = {
                "type": "contradiction",
                "old_uri": old_uri,
                "new_uri": new_uri,
                "reason": classification["reason"],
                "confidence": classification["confidence"],
            }
        elif verdict == "merge":
            if len(manifest["merges"]) >= self._max_merges:
                return
            op = {
                "type": "merge",
                "uris": [mem_a["uri"], mem_b["uri"]],
                "merged_content": classification["merged_content"],
                "reason": classification["reason"],
                "confidence": classification["confidence"],
            }
        else:  # none：不做任何操作
            return
        if classification["confidence"] < self._verify_confidence_threshold:
            manifest["verification_tasks"].append(
                VerificationTask(
                    kind=verdict,
                    uris=(mem_a["uri"], mem_b["uri"]),
                    umo=str(mem_a.get("metadata", {}).get("umo", "") or ""),
                    question=(
                        f"分类提案 kind={verdict}，理由：{classification['reason']}"
                    ),
                )
            )
            return
        if verdict == "link":
            manifest["new_links"].append(op)
        elif verdict == "conflict":
            manifest["contradictions"].append(op)
        else:
            manifest["merges"].append(op)

    async def _decision_ready(self) -> bool:
        """Decision 路径是否可尝试；已确认未配置则本周期直接走 Chat 降级。"""
        if self._decision_unavailable:
            return False
        provider_manager = getattr(self._context, "provider_manager", None)
        decision_service = getattr(self._context, "decision", None)
        if (
            provider_manager is None
            or decision_service is None
            or _DECISION_PROVIDER_TYPE is None
        ):
            self._decision_unavailable = True
            return False
        try:
            provider = await provider_manager.get_using_provider_async(
                _DECISION_PROVIDER_TYPE
            )
        except Exception as e:
            logger.debug(
                "[简单长期记忆] 分析师探测 Decision provider 失败: error_type=%s",
                type(e).__name__,
            )
            provider = None
        if provider is None:
            self._decision_unavailable = True
            logger.debug("[简单长期记忆] 未配置 Decision provider，分类走 Chat 降级")
            return False
        return True

    async def _classify_by_decision(
        self, mem_a: dict[str, Any], mem_b: dict[str, Any], cosine: float
    ) -> dict[str, Any] | None:
        """真实 Decision 四分类：单次问答同时问关系与 link 类型，计入预算。"""
        if DecisionChoiceAnswer is None or DecisionChoiceQuestion is None:
            return None
        questions = {
            "relation": DecisionChoiceQuestion(
                instructions=(
                    "判断记忆A与记忆B的关系："
                    "merge=几乎同一事实应合并；link=语义相关应建边；"
                    "conflict=互相矛盾、新记忆取代旧记忆；none=无明显关系"
                ),
                criteria={
                    "merge": "几乎同一事实，应合并为一条",
                    "link": "语义相关但不是同一事实，应建立关联边",
                    "conflict": "互相矛盾，新的取代旧的",
                    "none": "无明显关系，不做任何操作",
                },
            ),
            "link_type": DecisionChoiceQuestion(
                instructions=(
                    "若上一问答案为 link，这条边的真实关系类型是什么；否则选择 related"
                ),
                criteria={
                    "supports": "记忆B佐证或支持记忆A",
                    "supersedes": "记忆B取代记忆A的信息",
                    "context": "记忆B提供记忆A的背景上下文",
                    "related": "一般性相关",
                },
            ),
        }
        state = {
            "memory_a": self._pair_state(mem_a),
            "memory_b": self._pair_state(mem_b),
            "cosine": round(float(cosine), 3),
        }
        answers = await self._llm.ask_decisions(questions, state)
        if not isinstance(answers, dict):
            return None
        relation = answers.get("relation")
        if not isinstance(relation, DecisionChoiceAnswer):
            return None
        verdict = str(relation.choice or "")
        if verdict not in PAIR_VERDICTS:
            return None
        confidence = self._bounded_confidence(
            getattr(relation, "confidence", None),
            getattr(relation, "probability", None),
        )
        if confidence is None:
            return None
        result: dict[str, Any] = {
            "verdict": verdict,
            "confidence": confidence,
            "reason": f"decision choose: {verdict} (p={relation.probability:.2f})",
        }
        if verdict == "link":
            link_type = answers.get("link_type")
            if not isinstance(link_type, DecisionChoiceAnswer):
                return None
            relation_type = str(link_type.choice or "")
            if relation_type not in RELATION_TYPES:
                return None
            result["relation_type"] = relation_type
            result["reason"] += f" link_type={relation_type}"
        if verdict == "merge":
            # Decision 无法合成文本，合并内容取较新一条（新版本覆盖旧版本）
            result["merged_content"] = self._newer_content(mem_a, mem_b)
        return result

    async def _classify_by_chat(
        self, chunk: list[tuple[dict[str, Any], dict[str, Any], float]]
    ) -> list[dict[str, Any] | None]:
        """Chat 降级：一批候选对一次结构化 JSON 分类；无效输出按 none 处理。"""
        candidates_payload = []
        for index, (mem_a, mem_b, cosine) in enumerate(chunk):
            candidates_payload.append(
                {
                    "id": str(index),
                    "memory_a": self._pair_state(mem_a),
                    "memory_b": self._pair_state(mem_b),
                    "cosine": round(float(cosine), 3),
                }
            )
        prompt = build_prompt(
            ANALYST_CLASSIFY_PROMPT,
            {"candidates": json.dumps(candidates_payload, ensure_ascii=False)},
        )
        parsed = await self._llm.chat_json(
            _CLASSIFY_SYSTEM_PROMPT,
            prompt,
            model_id=self._config.get("maintenance_model_id", ""),
        )
        items: dict[str, dict[str, Any]] = {}
        duplicated: set[str] = set()
        if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
            for item in parsed["results"]:
                if not isinstance(item, dict):
                    continue
                item_id = str(item.get("id", "") or "")
                if item_id in items:
                    duplicated.add(item_id)
                else:
                    items[item_id] = item
        classifications: list[dict[str, Any] | None] = []
        for index in range(len(chunk)):
            item = items.get(str(index))
            if item is None or str(index) in duplicated:
                classifications.append(None)
                continue
            classifications.append(self._classification_from_item(item))
        return classifications

    def _classification_from_item(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """校验单条 Chat 分类输出；verdict/关系类型/置信度/理由无效按 none。"""
        verdict = str(item.get("verdict", "") or "")
        if verdict not in PAIR_VERDICTS:
            return None
        reason = str(item.get("reason", "") or "").strip()
        confidence = self._bounded_confidence(item.get("confidence"), None)
        if not reason or confidence is None:
            return None
        if verdict == "link":
            relation_type = str(item.get("relation_type", "") or "")
            if relation_type not in RELATION_TYPES:
                return None
            return {
                "verdict": verdict,
                "relation_type": relation_type,
                "reason": reason[:200],
                "confidence": confidence,
            }
        if verdict == "merge":
            merged = sanitize_memory_content(str(item.get("merged_content", "") or ""))
            if not merged:
                return None
            return {
                "verdict": verdict,
                "merged_content": merged,
                "reason": reason[:200],
                "confidence": confidence,
            }
        return {"verdict": verdict, "reason": reason[:200], "confidence": confidence}

    @staticmethod
    def _bounded_confidence(*values: Any) -> float | None:
        """取第一个有效的 0 到 1 有限置信度。"""
        for value in values:
            confidence = valid_unit_number(value)
            if confidence is not None:
                return confidence
        return None

    @staticmethod
    def _pair_state(mem: dict[str, Any]) -> dict[str, Any]:
        """构造分类输入的单条记忆摘要。"""
        meta = mem.get("metadata", {})
        return {
            "uri": mem.get("uri", ""),
            "content": str(mem.get("content", ""))[:1000],
            "created_at": meta.get("created_at", ""),
            "updated_at": meta.get("updated_at", ""),
            "scope": meta.get("memory_scope", ""),
        }

    @staticmethod
    def _newer_content(mem_a: dict[str, Any], mem_b: dict[str, Any]) -> str:
        """Decision 路径 merge 的合成内容：取较新一条的正文。"""
        old_uri, _new_uri = orient_conflict(mem_a, mem_b)
        if old_uri == mem_a["uri"]:
            return str(mem_b.get("content", ""))
        return str(mem_a.get("content", ""))

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
        """两级预筛：余弦 >= 阈值且排除已连边的候选对。

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

            # 收集 >= 阈值的对
            n = len(valid_memories)
            for i in range(n):
                for j in range(i + 1, n):
                    # 按 scope 实际边界分组，避免跨租户配对
                    mem_a = valid_memories[i]
                    mem_b = valid_memories[j]
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
