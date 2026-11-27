"""整理师：去重合并、单条存量巡检与减法。

两条通道：
1. merge 通道：同 scope/owner 分组、余弦 >=0.9 预筛、batch_size 有界，
   已消费 URI 防重复合并（保留既有行为）
2. 单条巡检通道：按优先级抽检活跃记忆（单条记忆也可运行），先经
   MemoryGate 四头预筛，再用批量清洗提示词产出 keep/archive/update；
   无效输出 fail closed 为 keep，所有提案交审核员复核，不直接写库
"""

from __future__ import annotations

import json
import re
from typing import Any

from astrbot.api import logger

from ..llm import MaintenanceLLM
from ..memory_gate import (
    GATE_CONTENT_MAX_CHARS,
    GATE_HEADS,
    GateCandidate,
    MemoryGate,
    build_maintenance_gate,
)
from ..persona import PersonaSummaryProvider
from ..prompts import ORGANIZER_CLEANING_PROMPT, build_prompt
from ..relations import valid_unit_number
from ..verifier import VerificationTask

try:  # 正常插件加载（包内相对导入）
    from ..prompts import sanitize_memory_content
except ImportError:  # 测试环境：仓库根作为顶层目录直接在 sys.path
    from prompts import sanitize_memory_content

# 巡检优先的风险词：相对时间、指令腔、情绪记账
_INSPECTION_RISK_RE = re.compile(
    r"今天|明天|昨天|前天|后天|上周|下周|这周|大前天|大后天|刚刚|刚才|"
    r"以后|每次|只要|必须|别再|别用|不要|不准|禁止|帮我|"
    r"讨厌|生气|吵架|骂|拉黑|绝交|记仇|烦死|垃圾"
)

_CLEANING_SYSTEM_PROMPT = "你是记忆库的存量清洗员，只输出结构化 JSON。"

_CLEANING_ACTIONS = ("keep", "archive", "update")

_INSPECTION_METADATA_KEYS = (
    "memory_scope",
    "domain",
    "created_at",
    "updated_at",
    "recall_count",
    "created_by",
    "owner_user_id",
    "owner_session_id",
    "umo",
)


class OrganizerAgent:
    """整理师 Agent。"""

    def __init__(
        self,
        context: Any,
        memory_mgr: Any,
        llm: MaintenanceLLM,
        config: dict[str, Any],
        *,
        gate: MemoryGate | None = None,
        persona_provider: PersonaSummaryProvider | None = None,
    ) -> None:
        self._context = context
        self._memory_mgr = memory_mgr
        self._llm = llm
        self._config = config
        self._merge_cosine_threshold = 0.9  # merge 候选预筛阈值（破坏性操作，阈值高）
        self._batch_size = config.get("maintenance_organizer_batch_size", 30)
        self._verify_confidence_threshold = float(
            config.get("maintenance_verification_confidence_threshold", 0.6)
        )
        # 宿主传入周期内复用的闸门与人格读取器；单测直接构造时按同样规则自建
        self._gate = gate
        self._persona_provider = persona_provider

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
                "inspected": int,
                "verification_tasks": [VerificationTask, ...],
            }
        """
        manifest: dict[str, Any] = {
            "merge": [],
            "archive": [],
            "update": [],
            "notes": "",
            "candidates_screened": 0,
            "llm_calls": 0,
            "memory_count": 0,
            "vector_count": 0,
            "vector_missing": 0,
            "inspected": 0,
            "inspection_kept": 0,
            "inspection_unavailable": 0,
            "verification_tasks": [],
        }
        memories = await self._get_active_memories(owner_filter)
        manifest["memory_count"] = len(memories)
        manifest["vector_count"] = sum(
            mem.get("vector") is not None for mem in memories
        )
        manifest["vector_missing"] = manifest["memory_count"] - manifest["vector_count"]
        logger.debug(
            "[简单长期记忆] 整理师记忆输入: memories=%s, vectors=%s, missing=%s",
            manifest["memory_count"],
            manifest["vector_count"],
            manifest["vector_missing"],
        )

        consumed_uris: set[str] = set()
        if len(memories) >= 2:
            consumed_uris = await self._run_merge_channel(manifest, memories)
        else:
            manifest["notes"] = "记忆数量不足，跳过相似合并"

        # 单条存量巡检不受 len<2 限制；单条记忆也能评估
        await self._run_inspection(manifest, memories, consumed_uris)

        logger.debug(
            "[简单长期记忆] 整理师完成: candidates=%s, merge=%s, inspected=%s, "
            "archive=%s, update=%s, llm_calls=%s",
            manifest["candidates_screened"],
            len(manifest["merge"]),
            manifest["inspected"],
            len(manifest["archive"]),
            len(manifest["update"]),
            manifest["llm_calls"],
        )
        return manifest

    async def _run_merge_channel(
        self, manifest: dict[str, Any], memories: list[dict[str, Any]]
    ) -> set[str]:
        """merge 通道：余弦 >=0.9 预筛 + 逐对 LLM 裁决，返回已消费 URI 集合。"""
        candidate_pairs = await self._screen_merge_candidates(memories)
        manifest["candidates_screened"] = len(candidate_pairs)
        logger.debug(
            "[简单长期记忆] 整理师预筛完成: threshold=%.2f, candidates=%s",
            self._merge_cosine_threshold,
            manifest["candidates_screened"],
        )
        if not candidate_pairs:
            manifest["notes"] = "预筛后无 merge 候选对"
            return set()

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
        return consumed_uris

    async def _run_inspection(
        self,
        manifest: dict[str, Any],
        memories: list[dict[str, Any]],
        consumed_uris: set[str],
    ) -> None:
        """单条存量巡检：优先级选样 → MemoryGate 预筛 → 批量清洗判定。"""
        eligible = [
            mem
            for mem in memories
            if mem.get("uri") and mem.get("content") and mem["uri"] not in consumed_uris
        ]
        batch = self._select_inspection_batch(eligible)
        manifest["inspected"] = len(batch)
        if not batch:
            return

        gate = self._gate or build_maintenance_gate(
            self._context, self._llm, self._config
        )
        outcomes = await gate.evaluate(
            [
                GateCandidate(
                    candidate_id=mem["uri"],
                    content=str(mem["content"]),
                    evidence=self._inspection_evidence(mem),
                )
                for mem in batch
            ]
        )
        cleaning: list[dict[str, Any]] = []
        for mem in batch:
            outcome = outcomes.get(mem["uri"])
            if outcome is None or not outcome.approved:
                if outcome is not None and outcome.status == "rejected":
                    cleaning.append(mem)
                else:
                    # 闸门不可用：fail closed 保留
                    manifest["inspection_unavailable"] += 1
                    continue
            else:
                manifest["inspection_kept"] += 1
        if not cleaning:
            return

        if self._llm.remaining_calls <= 0:
            logger.info("[简单长期记忆] 巡检清洗预算耗尽，候选留待下周期")
            manifest["inspection_unavailable"] += len(cleaning)
            return
        decisions = await self._clean_batch(cleaning, outcomes)
        manifest["llm_calls"] = self._llm.calls
        for mem in cleaning:
            self._apply_cleaning_decision(manifest, mem, decisions.get(mem["uri"]))

    def _select_inspection_batch(
        self, memories: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """按巡检优先级限量选样：风险词 > 召回次数 > group > curator 产出。"""
        limit = max(1, int(self._batch_size))
        ranked = sorted(memories, key=self._inspection_priority, reverse=True)
        return ranked[:limit]

    @staticmethod
    def _inspection_priority(mem: dict[str, Any]) -> tuple[int, ...]:
        """巡检优先级键：含相对时间/指令腔/情绪词与高召回优先。"""
        meta = mem.get("metadata", {})
        content = str(mem.get("content", "") or "")
        try:
            recall = int(meta.get("recall_count") or 0)
        except (TypeError, ValueError):
            recall = 0
        return (
            1 if _INSPECTION_RISK_RE.search(content) else 0,
            recall,
            1 if meta.get("memory_scope") == "group" else 0,
            1 if meta.get("created_by") == "maintenance_curator" else 0,
        )

    @staticmethod
    def _inspection_evidence(mem: dict[str, Any]) -> str:
        """构造闸门证据摘要（脱敏，只含计数与标识类字段）。"""
        meta = mem.get("metadata", {})
        return (
            f"scope={meta.get('memory_scope', '')}, "
            f"domain={meta.get('domain', '')}, "
            f"recall_count={meta.get('recall_count', 0)}, "
            f"created_by={meta.get('created_by', '')}"
        )

    async def _clean_batch(
        self, cleaning: list[dict[str, Any]], gate_outcomes: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """批量清洗判定；解析失败返回空 dict，调用方按 keep 处理。"""
        persona_summary = await self._persona_summary()
        candidates_payload = []
        for mem in cleaning:
            outcome = gate_outcomes.get(mem["uri"])
            scores = outcome.scores if outcome is not None else {}
            meta = mem.get("metadata", {})
            candidates_payload.append(
                {
                    "id": mem["uri"],
                    "content": str(mem["content"])[:GATE_CONTENT_MAX_CHARS],
                    "metadata": {
                        key: meta.get(key) for key in _INSPECTION_METADATA_KEYS
                    },
                    "gate_scores": {head: scores.get(head) for head in GATE_HEADS},
                }
            )
        prompt = build_prompt(
            ORGANIZER_CLEANING_PROMPT,
            {
                "persona_summary": persona_summary or "（未读取到主人格）",
                "candidates": json.dumps(
                    candidates_payload, ensure_ascii=False, indent=1
                ),
            },
        )
        parsed = await self._llm.chat_json(
            _CLEANING_SYSTEM_PROMPT,
            prompt,
            model_id=self._cleaning_model_id(),
        )
        return self._parse_cleaning_results(parsed, [mem["uri"] for mem in cleaning])

    def _apply_cleaning_decision(
        self,
        manifest: dict[str, Any],
        mem: dict[str, Any],
        decision: dict[str, Any] | None,
    ) -> None:
        """应用单条清洗判定：无效或 keep 一律保留，archive/update 生成操作。"""
        uri = mem["uri"]
        action = str((decision or {}).get("action", "") or "")
        if action not in ("archive", "update"):
            manifest["inspection_kept"] += 1
            return
        reason = str((decision or {}).get("reason", "") or "").strip()[:200]
        if action == "archive":
            op = {"type": "archive", "uri": uri, "reason": reason}
        else:
            new_content = sanitize_memory_content(
                str((decision or {}).get("new_content", "") or "")
            )
            original = " ".join(str(mem.get("content", "") or "").split())
            if not new_content or " ".join(new_content.split()) == original:
                # 改写必须非空且与原文有实质差异，否则保留原记忆
                manifest["inspection_kept"] += 1
                return
            op = {
                "type": "update",
                "uri": uri,
                "new_content": new_content,
                "reason": reason,
            }
        confidence = valid_unit_number((decision or {}).get("confidence"))
        # 置信度缺失或非法（NaN/越界）按低置信处理：转核验而非直接执行
        needs_verification = (
            confidence is None
            or bool((decision or {}).get("needs_verification"))
            or confidence < self._verify_confidence_threshold
        )
        if needs_verification:
            manifest["verification_tasks"].append(
                VerificationTask(
                    kind=action,
                    uris=(uri,),
                    umo=str(mem.get("metadata", {}).get("umo", "") or ""),
                    question=f"巡检提案 kind={action}，理由：{reason or '（未给出）'}",
                )
            )
            return
        if action == "archive":
            manifest["archive"].append(op)
        else:
            manifest["update"].append(op)

    @staticmethod
    def _parse_cleaning_results(
        parsed: Any, valid_uris: list[str]
    ) -> dict[str, dict[str, Any]]:
        """校验清洗输出：未知/重复/无效条目丢弃，缺失按 keep 处理。"""
        if not isinstance(parsed, dict):
            return {}
        results = parsed.get("results")
        if not isinstance(results, list):
            return {}
        valid = set(valid_uris)
        decisions: dict[str, dict[str, Any]] = {}
        for item in results:
            if not isinstance(item, dict):
                continue
            uri = str(item.get("id", "") or "").strip()
            if uri not in valid or uri in decisions:
                continue
            action = str(item.get("action", "") or "")
            if action not in _CLEANING_ACTIONS:
                continue
            decisions[uri] = item
        return decisions

    def _cleaning_model_id(self) -> str:
        """清洗判定模型：organizer 专用模型缺省回落维护模型。"""
        return self._config.get(
            "maintenance_organizer_model_id", ""
        ) or self._config.get("maintenance_model_id", "")

    async def _persona_summary(self) -> str:
        """读取主人格边界摘要（Reviewer 与 Organizer 共用实现）。"""
        provider = self._persona_provider or PersonaSummaryProvider(
            self._context, self._config
        )
        return await provider.get()

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
        """两级预筛：找出向量余弦 >= 阈值的候选对。

        Returns:
            [(mem_a, mem_b, cosine), ...] 按余弦降序
        """
        candidates: list[tuple[dict[str, Any], dict[str, Any], float]] = []

        # 按 batch_size 截断，避免大 KB 分配巨大相似度矩阵
        batch_size = self._config.get("maintenance_organizer_batch_size", 200)
        if len(memories) > batch_size:
            import random

            memories = random.sample(memories, batch_size)

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

            # 收集 >= 阈值的对（无向，去重，按 owner 分组）
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
