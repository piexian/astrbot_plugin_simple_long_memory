"""后台整理执行管线。

一次整理周期 = 一条顺序管线：
  purge（纯逻辑）→ extract（分段员+整理师提取对话记忆）→ organizer → analyst
  → reviewer（复核前几个角色的 manifest）

所有 Agent 只输出结构化 manifest，Host 解析后才执行 DB 写入。
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from astrbot.api import logger

from .agents.analyst import AnalystAgent
from .agents.curator import CuratorAgent
from .agents.organizer import OrganizerAgent
from .agents.reviewer import ReviewerAgent
from .extraction_journal import ExtractionJournal
from .inbox import STATUS_PROCESSED, STATUS_SKIPPED
from .llm import MaintenanceLLM
from .memory_gate import (
    GATE_CONTENT_MAX_CHARS,
    GateCandidate,
    MemoryGate,
    build_maintenance_gate,
)
from .persona import PersonaSummaryProvider
from .verifier import MaintenanceVerifier, VerificationTask

try:  # 正常插件加载（包内相对导入）
    from ..extraction_utils import normalize_extracted_scope
    from ..memory_protocol import (
        MemoryScope,
        UMOInfo,
        normalize_memory_scope,
        validate_owner_ids,
    )
except ImportError:  # 测试环境：仓库根作为顶层目录直接在 sys.path
    from extraction_utils import normalize_extracted_scope
    from memory_protocol import (
        MemoryScope,
        UMOInfo,
        normalize_memory_scope,
        validate_owner_ids,
    )


@dataclass
class AgentManifest:
    """Agent 输出的结构化操作清单。"""

    agent_type: str  # organizer / analyst / reviewer
    operations: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    raw_response: str = ""
    parsed: bool = False
    error: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class MaintenanceReport:
    """一次整理周期的结构化报告。"""

    session_id: str
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    finished_at: str = ""
    duration_ms: float = 0.0

    # 各阶段统计
    purge_result: dict[str, Any] = field(default_factory=dict)
    dry_run: bool = False
    extract_manifest: AgentManifest | None = None
    organizer_manifest: AgentManifest | None = None
    analyst_manifest: AgentManifest | None = None
    reviewer_verdicts: list[dict[str, Any]] = field(default_factory=list)

    # LLM 统计
    llm_stats: dict[str, Any] = field(default_factory=dict)

    # 执行结果
    executed_ops: int = 0
    skipped_ops: int = 0
    failed_ops: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "dry_run": self.dry_run,
            "purge_result": self.purge_result,
            "extract": {
                "ops_count": len(self.extract_manifest.operations)
                if self.extract_manifest
                else 0,
                "parsed": self.extract_manifest.parsed
                if self.extract_manifest
                else False,
                "error": self.extract_manifest.error if self.extract_manifest else "",
                "metrics": self.extract_manifest.metrics
                if self.extract_manifest
                else {},
            },
            "organizer": {
                "ops_count": len(self.organizer_manifest.operations)
                if self.organizer_manifest
                else 0,
                "parsed": self.organizer_manifest.parsed
                if self.organizer_manifest
                else False,
                "error": self.organizer_manifest.error
                if self.organizer_manifest
                else "",
                "metrics": self.organizer_manifest.metrics
                if self.organizer_manifest
                else {},
            },
            "analyst": {
                "ops_count": len(self.analyst_manifest.operations)
                if self.analyst_manifest
                else 0,
                "parsed": self.analyst_manifest.parsed
                if self.analyst_manifest
                else False,
                "error": self.analyst_manifest.error if self.analyst_manifest else "",
                "metrics": self.analyst_manifest.metrics
                if self.analyst_manifest
                else {},
            },
            "reviewer_verdicts_count": len(self.reviewer_verdicts),
            "llm_stats": self.llm_stats,
            "executed_ops": self.executed_ops,
            "skipped_ops": self.skipped_ops,
            "failed_ops": self.failed_ops,
            "errors": self.errors,
        }


class _MaintenanceEvent:
    """后台提取用最小事件 shim：无真实消息事件，只提供 store_memory 所需字段。"""

    def __init__(self, umo: str, sender_id: str = "") -> None:
        self.unified_msg_origin = umo
        self._sender_id = sender_id

    def get_sender_id(self) -> str:
        return self._sender_id


class MaintenanceRunner:
    """后台整理执行管线。"""

    def __init__(
        self,
        context: Any,
        memory_mgr: Any,
        llm: MaintenanceLLM,
        config: dict[str, Any],
        kv_put: Any = None,
        kv_get: Any = None,
    ) -> None:
        self._context = context
        self._memory_mgr = memory_mgr
        self._llm = llm
        self._config = config
        self._kv_put = kv_put
        self._kv_get = kv_get
        self._running = False
        # 执行阶段与人工审批共用的互斥锁：
        # run_cycle 阶段 4 与 /memory review approve 都会写记忆存储，必须互斥
        self._op_lock = asyncio.Lock()
        # 最近一次提取 manifest（供 _run_reviewer 并入审核流，见 run_cycle）
        self._extract_manifest_for_review: AgentManifest | None = None
        # 最近一次审核未产出裁决的原因分布（由 _run_reviewer 回填）
        self._last_review_unresolved: dict[str, int] = {}
        self._extract_journal = ExtractionJournal(kv_get, kv_put)
        self._gate: MemoryGate | None = None
        # 周期内共享的人格摘要读取器与受控核验器（每周期清缓存/清计数）
        self._persona_provider: PersonaSummaryProvider | None = None
        self._verifier: MaintenanceVerifier | None = None
        self._verification_used = 0

    def _reset_cycle_state(self) -> None:
        """周期级状态复位：LLM 统计、人格缓存、核验计数。"""
        self._llm.reset_cycle_stats()
        self._verification_used = 0
        if self._persona_provider is not None:
            self._persona_provider.clear()

    def _get_persona_provider(self) -> PersonaSummaryProvider:
        """懒加载周期内共享的人格摘要读取器（Reviewer 与 Organizer 共用）。"""
        if self._persona_provider is None:
            self._persona_provider = PersonaSummaryProvider(self._context, self._config)
        return self._persona_provider

    def _get_verifier(self) -> MaintenanceVerifier:
        """懒加载受控核验器（真实 tool_loop_agent，预算经 MaintenanceLLM）。"""
        if self._verifier is None:
            self._verifier = MaintenanceVerifier(
                context=self._context,
                memory_mgr=self._memory_mgr,
                llm=self._llm,
                config=self._config,
            )
        return self._verifier

    async def _run_verifications(
        self, tasks: list[VerificationTask]
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """执行本阶段积累的核验任务，返回 (提案操作, 统计)。

        每周期核验次数受 maintenance_verification_max_per_cycle 限制；
        预算不足或超出上限的任务不执行（其原提案一并保留现状，留待下周期）。
        """
        needed = len(tasks)
        if needed == 0:
            return [], {
                "verification_needed": 0,
                "verification_executed": 0,
                "verification_budget_skipped": 0,
            }
        cap = int(self._config.get("maintenance_verification_max_per_cycle", 2))
        allowance = max(0, cap - self._verification_used)
        proposals, executed = await self._get_verifier().run(tasks, allowance=allowance)
        self._verification_used += executed
        return proposals, {
            "verification_needed": needed,
            "verification_executed": executed,
            "verification_budget_skipped": needed - executed,
        }

    async def run_cycle(self, *, dry_run: bool = False) -> MaintenanceReport:
        """执行一次完整整理周期；dry_run 时不执行阶段 4 写入。"""
        if self._running:
            logger.warning("[简单长期记忆] 整理周期已在运行，跳过")
            raise RuntimeError("maintenance cycle already running")

        self._running = True
        session_id = f"maint-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        report = MaintenanceReport(session_id=session_id, dry_run=dry_run)
        started = datetime.now(timezone.utc)
        previous_cache_enabled = getattr(self._llm, "_cache_enabled", None)
        if dry_run and previous_cache_enabled is not None:
            self._llm._cache_enabled = False
            logger.debug("[简单长期记忆] dry-run 已禁用 LLM 磁盘缓存")

        try:
            logger.info(f"[简单长期记忆] 开始整理周期: {session_id}")

            # 重置周期级状态（LLM 统计、人格缓存、核验计数）
            self._reset_cycle_state()
            logger.debug(
                "[简单长期记忆] 整理周期配置: session=%s, extract=%s, organizer=%s, "
                "analyst=%s, reviewer=%s, max_llm=%s, max_ops=%s",
                session_id,
                self._config.get("maintenance_extract_enabled", True),
                self._config.get("maintenance_organizer_enabled", True),
                self._config.get("maintenance_analyst_enabled", True),
                self._config.get("maintenance_reviewer_enabled", True),
                self._config.get("maintenance_max_llm_calls", 50),
                self._config.get("maintenance_max_ops_per_cycle", 100),
            )
            # ── 阶段 0: purge（纯逻辑，无 LLM）──
            if self._config.get("auto_purge_enabled", True):
                purge_days = self._config.get("auto_purge_after_days", 7)
                try:
                    purge_result = await self._memory_mgr.purge_deprecated(
                        after_days=purge_days, dry_run=dry_run
                    )
                    report.purge_result = purge_result
                    logger.info(
                        f"[简单长期记忆] purge 完成: {purge_result.get('purged', 0)} 条记忆"
                    )
                except Exception as e:
                    report.errors.append(f"purge: {e}")
                    logger.warning(f"[简单长期记忆] purge 失败: {e}")

            # ── 阶段 0.5: extract（分段员切对话块 + 整理师提取记忆）──
            if self._config.get("maintenance_extract_enabled", True):
                try:
                    report.extract_manifest = await self._run_extraction(
                        dry_run=dry_run
                    )
                except Exception as e:
                    report.errors.append(f"extract: {e}")
                    logger.warning(f"[简单长期记忆] 提取阶段失败: {e}")

            # ── 阶段 1: organizer（整理师）──
            if self._config.get("maintenance_organizer_enabled", True):
                try:
                    report.organizer_manifest = await self._run_organizer()
                except Exception as e:
                    report.errors.append(f"organizer: {e}")
                    logger.warning(f"[简单长期记忆] organizer 失败: {e}")

            # ── 阶段 2: analyst（分析师）──
            if self._config.get("maintenance_analyst_enabled", True):
                try:
                    report.analyst_manifest = await self._run_analyst(
                        exclude_uris=self._organizer_merge_uris(
                            report.organizer_manifest
                        )
                    )
                except Exception as e:
                    report.errors.append(f"analyst: {e}")
                    logger.warning(f"[简单长期记忆] analyst 失败: {e}")

            # ── 阶段 3: reviewer（审核员）──
            if self._config.get("maintenance_reviewer_enabled", True):
                try:
                    # extract 操作并入审核流头部；经实例属性传递以保持
                    # _run_reviewer(organizer, analyst) 调用签名不变
                    self._extract_manifest_for_review = report.extract_manifest
                    report.reviewer_verdicts = await self._run_reviewer(
                        organizer_manifest=report.organizer_manifest,
                        analyst_manifest=report.analyst_manifest,
                    )
                except Exception as e:
                    report.errors.append(f"reviewer: {e}")
                    logger.warning(f"[简单长期记忆] reviewer 失败: {e}")
                finally:
                    self._extract_manifest_for_review = None

            # ── 阶段 3.5: 互审模式（驳回理由回传修正，最多 2 轮）──
            max_revision_rounds = 2
            for round_num in range(max_revision_rounds):
                # 检查是否有 reject 且 controversial 的项
                controversial_items = [
                    v
                    for v in report.reviewer_verdicts
                    if v.get("verdict") == "reject" and v.get("controversial", False)
                ]
                if not controversial_items:
                    break

                logger.info(
                    f"[简单长期记忆] 第 {round_num + 1} 轮互审："
                    f"{len(controversial_items)} 个争议项回传修正"
                )

                # 回传修正（简化版：重新运行原角色，输入包含驳回理由）
                # Phase 5 完善：将驳回理由注入原角色的 prompt
                # 目前先跳过修正，直接标记为待人工审核
                for item in controversial_items:
                    item["needs_human_review"] = True

            # ── 阶段 4: Host 执行（根据审核结果执行操作）──
            if dry_run:
                logger.info(
                    "[简单长期记忆] 整理 dry-run 完成，跳过阶段 4 的所有写操作: session=%s",
                    session_id,
                )
            else:
                await self._execute_operations(report)

        except Exception as e:
            report.errors.append(f"cycle: {e}")
            logger.error(f"[简单长期记忆] 整理周期异常: {e}")
        finally:
            finished = datetime.now(timezone.utc)
            report.finished_at = finished.isoformat()
            report.duration_ms = (finished - started).total_seconds() * 1000
            self._running = False
            report.llm_stats = self._llm.stats()
            if dry_run and previous_cache_enabled is not None:
                self._llm._cache_enabled = previous_cache_enabled
            logger.debug(
                "[简单长期记忆] 整理周期收尾: session=%s, report=%s",
                session_id,
                report.to_dict(),
            )

            # 落 KV（Phase 2 先日志输出，Phase 3 接 KV）
            extract_info = ""
            if report.extract_manifest is not None:
                _em = report.extract_manifest.metrics
                extract_info = (
                    f", 提取=块{_em.get('blocks_processed', 0)}"
                    f"/新增{_em.get('created', 0)}/更新{_em.get('updated', 0)}"
                )
            logger.info(
                f"[简单长期记忆] 整理周期完成: {session_id}, "
                f"耗时 {report.duration_ms:.0f}ms, "
                f"LLM 调用 {report.llm_stats.get('calls', 0)} 次{extract_info}"
            )

        return report

    @staticmethod
    def _test_manifest_summary(manifest: AgentManifest) -> dict[str, Any]:
        """压缩测试输出，避免命令回显记忆正文或模型理由。"""
        return {
            "parsed": manifest.parsed,
            "operations": len(manifest.operations),
            "metrics": manifest.metrics,
            "error": manifest.error,
        }

    async def run_test_stage(self, stage: str) -> dict[str, Any]:
        """直接运行一项后台能力测试，不执行任何持久化写操作。"""
        normalized = stage.strip().lower()
        if normalized == "cycle":
            return (await self.run_cycle(dry_run=True)).to_dict()
        if normalized not in {"purge", "extract", "organizer", "analyst", "reviewer"}:
            raise ValueError(
                "未知后台测试项，合法值: purge, extract, organizer, analyst, reviewer, cycle"
            )
        if self._running:
            raise RuntimeError("maintenance cycle already running")

        self._running = True
        started = datetime.now(timezone.utc)
        result: dict[str, Any] = {"stage": normalized, "dry_run": True}
        self._reset_cycle_state()
        previous_cache_enabled = getattr(self._llm, "_cache_enabled", None)
        if previous_cache_enabled is not None:
            self._llm._cache_enabled = False
            logger.debug("[简单长期记忆] 后台测试已禁用 LLM 磁盘缓存")
        try:
            if normalized == "purge":
                result["purge"] = await self._memory_mgr.purge_deprecated(
                    after_days=self._config.get("auto_purge_after_days", 7),
                    dry_run=True,
                )
            elif normalized == "extract":
                result["extract"] = self._test_manifest_summary(
                    await self._run_extraction(dry_run=True)
                )
            elif normalized == "organizer":
                result["organizer"] = self._test_manifest_summary(
                    await self._run_organizer()
                )
            elif normalized == "analyst":
                result["analyst"] = self._test_manifest_summary(
                    await self._run_analyst()
                )
            else:
                organizer = await self._run_organizer()
                analyst = await self._run_analyst(
                    exclude_uris=self._organizer_merge_uris(organizer)
                )
                verdicts = await self._run_reviewer(organizer, analyst)
                result["organizer"] = self._test_manifest_summary(organizer)
                result["analyst"] = self._test_manifest_summary(analyst)
                result["reviewer"] = {
                    "verdicts": len(verdicts),
                    "approved": sum(v.get("verdict") == "approve" for v in verdicts),
                    "rejected": sum(v.get("verdict") == "reject" for v in verdicts),
                    "unresolved": dict(self._last_review_unresolved),
                }
        finally:
            result["duration_ms"] = (
                datetime.now(timezone.utc) - started
            ).total_seconds() * 1000
            result["llm_stats"] = self._llm.stats()
            if previous_cache_enabled is not None:
                self._llm._cache_enabled = previous_cache_enabled
            self._running = False

        logger.info(
            "[简单长期记忆] 后台测试完成: stage=%s, dry_run=True, duration_ms=%.0f, llm_calls=%s",
            normalized,
            result["duration_ms"],
            result["llm_stats"].get("calls", 0),
        )
        return result

    async def _execute_operations(self, report: MaintenanceReport) -> None:
        """阶段 4：按审核结果执行操作。

        与人工审批（execute_approved）共用 _op_lock，避免与审批路径并发写竞态。
        """
        async with self._op_lock:
            executed = 0
            skipped = 0
            failed = 0
            pending_items: list[tuple[dict[str, Any], dict[str, Any]]] = []

            # 收集所有操作和对应的审核结果（提取操作排在最前，与审核下标一致）
            all_operations = []
            if report.extract_manifest and report.extract_manifest.parsed:
                all_operations.extend(report.extract_manifest.operations)
            if report.organizer_manifest and report.organizer_manifest.parsed:
                all_operations.extend(report.organizer_manifest.operations)
            if report.analyst_manifest and report.analyst_manifest.parsed:
                all_operations.extend(report.analyst_manifest.operations)

            n_reviewed_ops = len(all_operations)
            # 周期操作总数硬上限：防异常放大，超出部分下周期重新评估
            max_ops = self._config.get("maintenance_max_ops_per_cycle", 100)
            if len(all_operations) > max_ops:
                logger.warning(
                    f"[简单长期记忆] 本周期操作数 {len(all_operations)} 超过上限 "
                    f"{max_ops}，超出部分本周期跳过（下周期会重新评估）"
                )
                all_operations = all_operations[:max_ops]

            reviewer_enabled = self._config.get("maintenance_reviewer_enabled", True)
            logger.debug(
                "[简单长期记忆] 执行阶段输入: operations=%s, reviewer_enabled=%s, "
                "verdicts=%s, max_ops=%s",
                len(all_operations),
                reviewer_enabled,
                len(report.reviewer_verdicts),
                max_ops,
            )
            # 审核员按序审核、预算耗尽即中断：未裁决下标落在尾部即预算耗尽操作
            budget_tail_start = max(
                0,
                n_reviewed_ops
                - int(self._last_review_unresolved.get("budget_exhausted", 0) or 0),
            )
            budget_deferred = 0
            # reviewer 启用时校验裁决完整性：
            # 下标越界/重复 → 真正不一致（LLM 乱序/错号），告警；
            # 仅数量不足 → 有意的部分评审（预算耗尽/输出无效），info 级给出原因分布
            if reviewer_enabled and all_operations:
                raw_indices = [v.get("index") for v in report.reviewer_verdicts]
                n_ops = len(all_operations)
                invalid = [
                    i
                    for i in raw_indices
                    if not isinstance(i, int) or i < 0 or i >= n_ops
                ]
                duplicated = len(raw_indices) - len(set(raw_indices))
                if invalid or duplicated:
                    logger.warning(
                        f"[简单长期记忆] 审核员裁决与操作不一致"
                        f"（越界下标 {invalid}，重复 {duplicated} 条，"
                        f"共 {n_ops} 条操作），相关操作将 fail closed 转待审"
                    )
                else:
                    verdict_indices = set(raw_indices)
                    missing = [i for i in range(n_ops) if i not in verdict_indices]
                    if missing:
                        unresolved = self._last_review_unresolved
                        logger.info(
                            f"[简单长期记忆] 审核员部分评审: {len(missing)}/{n_ops} 条未裁决"
                            f"（预算耗尽 {unresolved.get('budget_exhausted', 0)}, "
                            f"输出无效 {unresolved.get('invalid_output', 0)}），"
                            f"预算耗尽部分留待下周期，其余 fail closed 转待审"
                        )

            # 按审核结果执行操作
            for i, op in enumerate(all_operations):
                # 查找对应的审核结果
                verdict = None
                for v in report.reviewer_verdicts:
                    if v.get("index") == i:
                        verdict = v
                        break

                # 缺失裁决 → 拒绝执行（避免 reviewer 故障时放行破坏性操作）
                if verdict is None:
                    # additive 操作（new_link/create）仅在审核员整体禁用时直接放行；
                    # 审核员启用时的缺失裁决需区分：预算耗尽的尾部操作留待下周期，
                    # 输出无效等异常缺失才 fail closed 转待审
                    if (
                        op.get("type") in ("new_link", "create")
                        and not reviewer_enabled
                    ):
                        try:
                            success = await self._execute_operation(op)
                            if success:
                                await self._extract_journal.acknowledge(op)
                                executed += 1
                            else:
                                failed += 1
                                report.errors.append(
                                    f"op[{i}] {op.get('type')} 执行返回失败"
                                )
                        except Exception as e:
                            failed += 1
                            report.errors.append(
                                f"op[{i}] {op.get('type')}: {type(e).__name__}"
                            )
                        continue
                    # 预算耗尽而未轮到审核：转入待审队列，避免操作丢失或阻塞提取日志
                    if reviewer_enabled and i >= budget_tail_start:
                        budget_deferred += 1
                        pending_items.append(
                            (
                                op,
                                {
                                    "verdict": "pending",
                                    "reason": "审核预算耗尽，转人工审核",
                                },
                            )
                        )
                        skipped += 1
                        continue
                    # 其余情况（破坏性操作，或审核员启用但裁决缺失）→ 转待审队列
                    reason = (
                        "reviewer 未启用"
                        if not reviewer_enabled
                        else "reviewer 缺失裁决（LLM 漏判或输出异常）"
                    )
                    pending_items.append((op, {"verdict": "pending", "reason": reason}))
                    skipped += 1
                    continue

                if verdict.get("verdict") == "approve":
                    # global 记忆操作必须经管理员人工确认（不可配置）
                    if await self._op_touches_global(op):
                        pending_items.append((op, verdict))
                        skipped += 1
                        logger.info(
                            f"[简单长期记忆] 操作 {i} 涉及 global 记忆，转人工审批"
                        )
                        continue
                    # 矛盾裁决不自动执行：转人工确认，批准后由 execute_approved 真正废弃旧记忆
                    if op.get("type") == "contradiction":
                        reason = verdict.get("reason", "")
                        pending_items.append(
                            (op, {**verdict, "reason": f"矛盾待人工裁决: {reason}"})
                        )
                        skipped += 1
                        logger.info(
                            f"[简单长期记忆] 矛盾操作 {i} 转人工裁决: "
                            f"{op.get('old_uri')} vs {op.get('new_uri')}"
                        )
                        continue
                    try:
                        success = await self._execute_operation(op)
                        if success:
                            await self._extract_journal.acknowledge(op)
                            executed += 1
                        else:
                            failed += 1
                            report.errors.append(
                                f"op[{i}] {op.get('type')} 执行返回失败"
                            )
                    except Exception as e:
                        logger.warning(
                            "[简单长期记忆] 执行操作失败: index=%s, type=%s, "
                            "error_type=%s",
                            i,
                            op.get("type"),
                            type(e).__name__,
                        )
                        failed += 1
                        report.errors.append(
                            f"op[{i}] {op.get('type')}: {type(e).__name__}"
                        )
                else:
                    # 审核拒绝 → 跳过
                    skipped += 1
                    if verdict.get("needs_human_review"):
                        # 争议项 → 写入 KV 待审队列
                        pending_items.append((op, verdict))
                        logger.info(
                            "[简单长期记忆] 争议项待人工审核: index=%s, type=%s",
                            i,
                            op.get("type"),
                        )
                    elif verdict.get("verdict") == "reject":
                        await self._extract_journal.acknowledge(op)

            persisted = await self._flush_pending_review(
                pending_items, report.session_id
            )
            for op, _ in pending_items:
                if self._op_signature(op) in persisted:
                    await self._extract_journal.acknowledge(op)
            await self._settle_inbox_blocks()

            report.executed_ops = executed
            report.skipped_ops = skipped
            report.failed_ops = failed
            # LLM 统计
            report.llm_stats = self._llm.stats()
            logger.debug(
                "[简单长期记忆] 执行阶段完成: executed=%s, skipped=%s, failed=%s, "
                "pending=%s, budget_deferred=%s, llm=%s",
                executed,
                skipped,
                failed,
                len(pending_items),
                budget_deferred,
                report.llm_stats,
            )

    async def execute_approved(self, op: dict[str, Any]) -> bool:
        """公开入口：执行一条经管理员批准的操作。

        与整理周期阶段 4 共用 _op_lock，避免与后台周期并发写竞态。
        供 /memory review approve 等外部命令调用，不应用 _execute_operation。
        """
        async with self._op_lock:
            return await self._execute_operation(op)

    async def _run_extraction(self, *, dry_run: bool) -> AgentManifest:
        """提取阶段：Inbox 纯规则切块 → 闸门 → 整理师提取 create/update 操作。

        闸门明确拒收的块标 skipped；判定失败或预算耗尽的块保留 pending 下周期；
        已产出操作的块在操作最终执行完成后才由 _settle_inbox_blocks 标 processed。
        LLM 预算占每周期上限的 maintenance_extract_llm_budget_ratio（默认 60%），
        闸门与整理师共享提取预算，不再消耗任何分段判定调用。
        """
        manifest = AgentManifest(agent_type="extract")
        journal = (
            ExtractionJournal(self._kv_get, None) if dry_run else self._extract_journal
        )
        if await journal.load():
            # 上一周期未完成的操作先走完审核与执行，本周期不再新提取
            manifest.parsed = True
            manifest.operations = journal.pending
            manifest.metrics = journal.state.get("metrics", {})
            manifest.notes = journal.state.get("notes", "")
            return manifest

        inbox = getattr(self._memory_mgr, "inbox", None)
        if inbox is None or not inbox.ready:
            manifest.error = "maintenance inbox unavailable"
            logger.warning("[简单长期记忆] 提取阶段跳过: 缓冲表不可用")
            return manifest

        segment_max_chars = int(self._config.get("maintenance_segment_max_chars", 8000))
        collect_max_chars = min(max(64, segment_max_chars), GATE_CONTENT_MAX_CHARS)
        collect = await inbox.collect_blocks(
            gap_minutes=self._config.get("maintenance_segment_time_gap_minutes", 30),
            max_chars=collect_max_chars,
            min_user_chars=self._config.get("extraction_min_content_length", 150),
            max_blocks=self._config.get("maintenance_extract_max_blocks_per_cycle", 20),
            commit=not dry_run,
        )
        blocks = collect["blocks"]
        collect_stats = collect["stats"]

        gate = self._get_gate()
        gate_outcomes = await gate.evaluate(
            [
                GateCandidate(candidate_id=block.key, content=block.text)
                for block in blocks
            ]
        )
        approved_blocks = [
            block for block in blocks if gate_outcomes[block.key].approved
        ]

        max_calls = int(self._config.get("maintenance_max_llm_calls", 50))
        ratio = float(self._config.get("maintenance_extract_llm_budget_ratio", 0.6))
        extract_budget = max(0, int(max_calls * ratio))
        curator_budget = max(0, extract_budget - self._llm.calls)

        curator = CuratorAgent(
            context=self._context,
            memory_mgr=self._memory_mgr,
            llm=self._llm,
            config=self._config,
        )
        if approved_blocks:
            cur_result = await curator.run(approved_blocks, curator_budget)
        else:
            cur_result = {
                "create": [],
                "update": [],
                "notes": "",
                "blocks_processed": 0,
                "blocks_nothing": 0,
                "llm_calls": 0,
                "outcomes": [],
            }
        curator_outcomes = {
            block.key: outcome for block, outcome in cur_result.get("outcomes", [])
        }

        # 块级最终结局：gate 拒收 → skipped；gate 失败/预算耗尽 → 留 pending
        skipped_rows: list[int] = []
        processed_rows: list[int] = []
        failed_rows: list[int] = []
        op_block_rows: dict[str, list[int]] = {}
        for block in blocks:
            gate_outcome = gate_outcomes[block.key]
            if gate_outcome.status == "rejected":
                outcome = "skipped"
            elif gate_outcome.status == "unavailable":
                outcome = "deferred"
            else:
                outcome = curator_outcomes.get(block.key, "deferred")
            if outcome == "skipped":
                skipped_rows.extend(block.row_ids)
            elif outcome == "nothing":
                processed_rows.extend(block.row_ids)
            elif outcome in ("created", "updated"):
                op_block_rows[block.key] = block.row_ids
            elif outcome == "failed":
                failed_rows.extend(block.row_ids)

        operations = list(cur_result.get("create", [])) + list(
            cur_result.get("update", [])
        )
        gate_approved = len(approved_blocks)
        gate_rejected = sum(1 for o in gate_outcomes.values() if o.status == "rejected")
        gate_unavailable = len(blocks) - gate_approved - gate_rejected
        manifest.parsed = True
        manifest.notes = str(cur_result.get("notes", "") or "")
        manifest.operations = operations
        manifest.metrics = {
            "inbox": collect_stats,
            "blocks": len(blocks),
            "gate_approved": gate_approved,
            "gate_rejected": gate_rejected,
            "gate_unavailable": gate_unavailable,
            "blocks_processed": cur_result.get("blocks_processed", 0),
            "blocks_nothing": cur_result.get("blocks_nothing", 0),
            "created": len(cur_result.get("create", [])),
            "updated": len(cur_result.get("update", [])),
            "llm_calls": cur_result.get("llm_calls", 0),
        }

        if not dry_run:
            if skipped_rows:
                await inbox.mark_rows(skipped_rows, STATUS_SKIPPED)
            if processed_rows:
                await inbox.mark_rows(processed_rows, STATUS_PROCESSED)
            if failed_rows:
                await inbox.mark_failed(failed_rows)
            if operations:
                await journal.prepare(
                    operations=operations,
                    block_rows=op_block_rows,
                    metrics=manifest.metrics,
                    notes=manifest.notes,
                )
                manifest.operations = journal.pending
        logger.debug(
            "[简单长期记忆] 提取阶段完成: blocks=%s, approved=%s, rejected=%s, "
            "unavailable=%s, processed=%s, create=%s, update=%s, llm_calls=%s, "
            "dry_run=%s",
            len(blocks),
            gate_approved,
            gate_rejected,
            gate_unavailable,
            manifest.metrics["blocks_processed"],
            manifest.metrics["created"],
            manifest.metrics["updated"],
            manifest.metrics["llm_calls"],
            dry_run,
        )
        return manifest

    def _get_gate(self) -> MemoryGate:
        """后台闸门：懒加载并复用同一实例，Decision 未配置缓存跨提取调用有效。

        Decision 走 MaintenanceLLM 受控包装计入总预算；Chat 降级复用 maintenance_model_id。
        """
        gate = getattr(self, "_gate", None)
        if gate is None:
            gate = build_maintenance_gate(self._context, self._llm, self._config)
            self._gate = gate
        return gate

    async def _settle_inbox_blocks(self) -> None:
        """把操作已全部完成的块对应缓冲行标 processed，全部成功后才清理日志。

        缓冲表不可用或任一行状态提交失败时保留日志，下周期重试；
        避免行仍为 pending 时日志被清导致同一对话块重复提取。
        """
        journal = self._extract_journal
        if not journal.state:
            return
        settled = journal.settled_block_rows()
        if settled:
            inbox = getattr(self._memory_mgr, "inbox", None)
            if inbox is None:
                logger.warning("[简单长期记忆] 缓冲表不可用，提取日志保留待下周期重试")
                return
            for block_key, row_ids in settled.items():
                if not await inbox.mark_rows(row_ids, STATUS_PROCESSED):
                    logger.warning(
                        "[简单长期记忆] 缓冲行状态提交失败: block=%s, rows=%s，"
                        "提取日志保留待下周期重试",
                        block_key,
                        len(row_ids),
                    )
                    return
        await journal.clear_if_finished()

    async def _run_organizer(self) -> AgentManifest:
        """运行整理师：去重合并、单条巡检与质量精炼。"""
        manifest = AgentManifest(agent_type="organizer")

        organizer = OrganizerAgent(
            context=self._context,
            memory_mgr=self._memory_mgr,
            llm=self._llm,
            config=self._config,
            gate=self._get_gate(),
            persona_provider=self._get_persona_provider(),
        )

        try:
            result = await organizer.run()
            manifest.parsed = True
            # 将 organizer 的输出转换为 operations 格式
            for merge_op in result.get("merge", []):
                manifest.operations.append(
                    {
                        "type": "merge",
                        "uris": merge_op.get("uris", []),
                        "merged_content": merge_op.get("merged_content", ""),
                        "reason": merge_op.get("reason", ""),
                        "confidence": merge_op.get("confidence", 0.0),
                    }
                )
            for archive_op in result.get("archive", []):
                manifest.operations.append(
                    {
                        "type": "archive",
                        "uri": archive_op.get("uri", ""),
                        "reason": archive_op.get("reason", ""),
                    }
                )
            for update_op in result.get("update", []):
                manifest.operations.append(
                    {
                        "type": "update",
                        "uri": update_op.get("uri", ""),
                        "new_content": update_op.get("new_content", ""),
                        "reason": update_op.get("reason", ""),
                    }
                )
            manifest.metrics = {
                key: result.get(key)
                for key in (
                    "memory_count",
                    "vector_count",
                    "vector_missing",
                    "candidates_screened",
                    "llm_calls",
                    "inspected",
                    "inspection_kept",
                    "inspection_unavailable",
                )
            }
            await self._attach_verification_ops(manifest, result)
            manifest.metrics.update(self._op_type_counts(manifest.operations))
            manifest.notes = result.get("notes", "")
        except Exception as e:
            manifest.error = str(e)
            logger.warning(f"[简单长期记忆] 整理师执行失败: {e}")

        return manifest

    async def _run_analyst(self, exclude_uris: set[str] | None = None) -> AgentManifest:
        """运行分析师：中文语义四分类与矛盾发现。

        exclude_uris 为整理师 merge 已消费的 URI，避免对同 URI 重复提案。
        """
        manifest = AgentManifest(agent_type="analyst")

        analyst = AnalystAgent(
            context=self._context,
            memory_mgr=self._memory_mgr,
            llm=self._llm,
            config=self._config,
        )

        try:
            result = await analyst.run(exclude_uris=exclude_uris)
            manifest.parsed = True
            # 将 analyst 的输出转换为 operations 格式
            for link_op in result.get("new_links", []):
                manifest.operations.append(
                    {
                        "type": "new_link",
                        "source": link_op.get("source", ""),
                        "target": link_op.get("target", ""),
                        "relation": link_op.get("relation", "related"),
                        "reason": link_op.get("reason", ""),
                        "confidence": link_op.get("confidence", 0.0),
                    }
                )
            for contra_op in result.get("contradictions", []):
                manifest.operations.append(
                    {
                        "type": "contradiction",
                        "old_uri": contra_op.get("old_uri", ""),
                        "new_uri": contra_op.get("new_uri", ""),
                        "reason": contra_op.get("reason", ""),
                        "confidence": contra_op.get("confidence", 0.0),
                    }
                )
            for merge_op in result.get("merges", []):
                manifest.operations.append(
                    {
                        "type": "merge",
                        "uris": merge_op.get("uris", []),
                        "merged_content": merge_op.get("merged_content", ""),
                        "reason": merge_op.get("reason", ""),
                        "confidence": merge_op.get("confidence", 0.0),
                    }
                )
            manifest.metrics = {
                key: result.get(key)
                for key in (
                    "memory_count",
                    "vector_count",
                    "vector_missing",
                    "candidates_screened",
                    "llm_calls",
                    "decision_calls",
                    "chat_fallback_calls",
                )
            }
            await self._attach_verification_ops(manifest, result)
            manifest.metrics.update(self._op_type_counts(manifest.operations))
            manifest.notes = result.get("notes", "")
        except Exception as e:
            manifest.error = str(e)
            logger.warning(f"[简单长期记忆] 分析师执行失败: {e}")

        return manifest

    async def _attach_verification_ops(
        self, manifest: AgentManifest, result: dict[str, Any]
    ) -> None:
        """把核验产出的提案并入 manifest 操作与统计。

        核验提案仍交审核员复核；未被核验的任务不直接执行（保留现状，
        留待下周期重新评估）。
        """
        proposals, stats = await self._run_verifications(
            list(result.get("verification_tasks", []))
        )
        manifest.operations.extend(proposals)
        manifest.metrics.update(stats)
        manifest.metrics["verification_proposals"] = len(proposals)

    @staticmethod
    def _op_type_counts(operations: list[dict[str, Any]]) -> dict[str, int]:
        """按操作类型计数（dry-run 报告用，不含正文与理由）。"""
        counts: dict[str, int] = {}
        for op in operations:
            op_type = str(op.get("type", "") or "")
            key = f"op_{op_type}" if op_type else "op_unknown"
            counts[key] = counts.get(key, 0) + 1
        return counts

    @staticmethod
    def _organizer_merge_uris(manifest: AgentManifest | None) -> set[str]:
        """提取整理师 merge 操作已消费的 URI，供分析师去重。"""
        uris: set[str] = set()
        if manifest is None:
            return uris
        for op in manifest.operations:
            if op.get("type") != "merge":
                continue
            for uri in op.get("uris", []):
                if isinstance(uri, str) and uri:
                    uris.add(uri)
        return uris

    async def _run_reviewer(
        self,
        organizer_manifest: AgentManifest | None,
        analyst_manifest: AgentManifest | None,
    ) -> list[dict[str, Any]]:
        """运行审核员：复核操作建议。"""
        verdicts: list[dict[str, Any]] = []

        # 收集待审核的 manifest（提取操作排在最前，与执行阶段顺序一致）
        proposed_changes = []
        extract_manifest = self._extract_manifest_for_review
        if extract_manifest and extract_manifest.parsed:
            proposed_changes.extend(extract_manifest.operations)
        if organizer_manifest and organizer_manifest.parsed:
            proposed_changes.extend(organizer_manifest.operations)
        if analyst_manifest and analyst_manifest.parsed:
            proposed_changes.extend(analyst_manifest.operations)

        if not proposed_changes:
            return verdicts

        # 使用 ReviewerAgent 执行审核
        reviewer = ReviewerAgent(
            context=self._context,
            memory_mgr=self._memory_mgr,
            llm=self._llm,
            config=self._config,
            persona_provider=self._get_persona_provider(),
        )

        # 源记录由审核员按操作逐条拉取（见 ReviewerAgent._get_op_source_data）
        self._last_review_unresolved = {}
        try:
            verdicts = await reviewer.review(proposed_changes)
            self._last_review_unresolved = dict(reviewer.last_unresolved)
        except Exception as e:
            logger.warning(f"[简单长期记忆] 审核员执行失败: {e}")
        return verdicts

    # ─── 数据拉取辅助（Phase 3/4 完善）────────────────────

    async def _get_memories_text(self) -> str:
        """拉取记忆列表文本（简化版）。"""
        # TODO: Phase 3 从 memory_mgr 拉取实际记忆
        return "# 记忆池\n（暂无数据）"

    async def _get_memory_count(self) -> int:
        """当前活跃记忆总数。"""
        # TODO: Phase 3 从 memory_mgr 统计
        return 0

    async def _get_conversation_history(self) -> str:
        """拉取对话历史。"""
        # TODO: Phase 4 从 conversation_manager 拉取
        return "# 最近对话\n（暂无数据）"

    async def _op_touches_global(self, op: dict[str, Any]) -> bool:
        """检查操作是否涉及 global 作用域记忆（查询实际 metadata）。"""
        uris: list[str] = []
        if op.get("uris"):
            uris.extend(op["uris"])
        for key in ("uri", "source", "target", "old_uri", "new_uri"):
            val = op.get(key)
            if isinstance(val, str) and val:
                uris.append(val)
        for uri in uris:
            try:
                mem = await self._memory_mgr._get_memory_by_uri(uri)
                if mem:
                    scope = mem.get("metadata", {}).get("memory_scope", "")
                    if scope == "global":
                        return True
            except Exception:
                pass
        return False

    @staticmethod
    def _op_signature(op: dict[str, Any]) -> str:
        """操作签名（规范化 JSON 的短哈希），用于待审队列跨周期去重。"""
        import hashlib
        import json as _json

        raw = _json.dumps(op, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    async def _flush_pending_review(
        self,
        items: list[tuple[dict[str, Any], dict[str, Any]]],
        session_id: str,
    ) -> set[str]:
        """批量写入待审队列，并保存容量溢出的非提取操作。"""
        if not self._kv_put or not self._kv_get:
            return set()
        try:
            import time as _time

            queue = await self._kv_get("maintenance_pending_review", None) or []
            raw_overflow = (
                await self._kv_get("maintenance_pending_review_overflow", None) or []
            )
            if not isinstance(queue, list):
                queue = []
            if not isinstance(raw_overflow, list):
                raw_overflow = []
            overflow = [
                entry
                for entry in raw_overflow
                if isinstance(entry, dict) and isinstance(entry.get("op"), dict)
            ]

            settled_sigs = {
                it.get("op_signature") or self._op_signature(it["op"])
                for it in queue
                if isinstance(it, dict)
                and it.get("status") in {"approved", "rejected"}
                and isinstance(it.get("op"), dict)
                and it.get("op", {}).get("_extract_id")
            }
            max_seen_id = max(
                (it.get("id", 0) for it in queue if isinstance(it, dict)),
                default=0,
            )
            seq = await self._kv_get("maintenance_pending_review_seq", None)
            if not isinstance(seq, int) or seq < max_seen_id:
                seq = max_seen_id

            pruned = sum(
                1
                for it in queue
                if not isinstance(it, dict) or it.get("status") != "pending"
            )
            if pruned:
                queue = [
                    it
                    for it in queue
                    if isinstance(it, dict) and it.get("status") == "pending"
                ]

            existing_sigs = {
                it.get("op_signature") or self._op_signature(it.get("op", {}))
                for it in queue
            } | settled_sigs
            overflow_sigs = {self._op_signature(entry["op"]) for entry in overflow}
            queue_max = max(
                1, int(self._config.get("maintenance_pending_queue_max", 500))
            )
            added = 0
            drained = 0
            deduped = 0
            journal_deferred = 0
            overflow_kept: list[dict[str, Any]] = []

            def append_queue(
                op: dict[str, Any],
                verdict: dict[str, Any],
                item_session_id: str,
            ) -> None:
                nonlocal added, seq
                seq += 1
                queue.append(
                    {
                        "id": seq,
                        "session_id": item_session_id,
                        "op_type": op.get("type", ""),
                        "op": op,
                        "op_signature": self._op_signature(op),
                        "verdict_reason": verdict.get("reason", ""),
                        "controversial": verdict.get("controversial", False),
                        "created_at": _time.strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "pending",
                    }
                )
                added += 1

            for entry in overflow:
                op = entry["op"]
                sig = self._op_signature(op)
                if sig in existing_sigs:
                    continue
                if op.get("_extract_id") or len(queue) >= queue_max:
                    overflow_kept.append(entry)
                    continue
                append_queue(
                    op,
                    entry.get("verdict")
                    if isinstance(entry.get("verdict"), dict)
                    else {},
                    str(entry.get("session_id") or session_id),
                )
                existing_sigs.add(sig)
                overflow_sigs.discard(sig)
                drained += 1

            for op, verdict in items:
                sig = self._op_signature(op)
                if sig in existing_sigs or sig in overflow_sigs:
                    deduped += 1
                    continue
                if len(queue) >= queue_max:
                    if op.get("_extract_id"):
                        # ExtractionJournal remains the retry source for extraction ops.
                        journal_deferred += 1
                    else:
                        overflow_kept.append(
                            {
                                "session_id": session_id,
                                "op": op,
                                "verdict": verdict,
                            }
                        )
                        overflow_sigs.add(sig)
                    continue
                append_queue(op, verdict, session_id)
                existing_sigs.add(sig)

            overflow_changed = overflow_kept != raw_overflow
            if added or pruned:
                await self._kv_put("maintenance_pending_review", queue)
            if added:
                await self._kv_put("maintenance_pending_review_seq", seq)
            if overflow_changed:
                await self._kv_put("maintenance_pending_review_overflow", overflow_kept)

            if items or overflow or pruned:
                logger.info(
                    "[简单长期记忆] 待审入队: 新增 %s, 溢出转存 %s, "
                    "队列满保留 %s, 提取日志保留 %s, 去重跳过 %s, 清理终态 %s, 当前待审 %s",
                    added,
                    drained,
                    len(overflow_kept),
                    journal_deferred,
                    deduped,
                    pruned,
                    len(queue),
                )
            if added and self._config.get("review_notify_enabled", True):
                await self._send_review_notification(len(queue))
            return existing_sigs
        except Exception as e:
            logger.warning(f"[简单长期记忆] 写入待审队列失败: {e}")
            return set()

    async def _send_review_notification(self, pending_count: int) -> None:
        """向管理员推送待审通知。"""
        try:
            umo = self._config.get("review_notify_umo", "")
            if not umo:
                return
            from astrbot.api.event import MessageChain
            from astrbot.api.message_components import Plain

            msg = MessageChain(
                chain=[
                    Plain(
                        text=f"📝 记忆整理待审通知\n当前有 {pending_count} 条争议操作等待审查。\n回复 /memory review 查看详情并处理"
                    )
                ]
            )
            if hasattr(msg, "use_markdown"):
                msg.use_markdown(False)
            await self._context.send_message(umo, msg)
        except Exception as e:
            logger.debug(f"[简单长期记忆] 待审通知发送失败: {e}")

    async def _execute_operation(self, op: dict[str, Any]) -> bool:
        """执行单个操作。"""
        op_type = op.get("type", "")

        try:
            if op_type == "create":
                return await self._execute_create(op)
            elif op_type == "merge":
                return await self._execute_merge(op)
            elif op_type == "archive":
                return await self._execute_archive(op)
            elif op_type == "update":
                return await self._execute_update(op)
            elif op_type == "new_link":
                return await self._execute_new_link(op)
            elif op_type == "contradiction":
                # 管理员批准后的真实执行：废弃被取代的旧记忆
                old_uri = op.get("old_uri", "")
                if not old_uri:
                    return False
                return await self._memory_mgr.deprecate_memory(
                    old_uri, reason=op.get("reason") or "contradiction resolved"
                )
            else:
                logger.warning(f"[简单长期记忆] 未知操作类型: {op_type}")
                return False
        except Exception as e:
            logger.warning(
                "[简单长期记忆] 执行操作失败: type=%s, error_type=%s",
                op_type,
                type(e).__name__,
            )
            return False

    async def _execute_create(self, op: dict[str, Any]) -> bool:
        """执行 create 操作（整理师从对话块提取的新记忆）。"""
        content = str(op.get("content") or "").strip()
        umo = str(op.get("umo") or "")
        if not content or not umo:
            return False

        parsed = UMOInfo.parse(umo)
        if not parsed.is_valid:
            logger.warning("[简单长期记忆] create 拒绝无效 UMO")
            return False
        session_type = parsed.session_type
        raw_scope = normalize_memory_scope(str(op.get("scope") or ""))
        scope = normalize_extracted_scope(raw_scope, session_type)
        if raw_scope == MemoryScope.GLOBAL:
            # 双保险：curator 解析已降级 global，执行侧再拦一次并留痕
            logger.info(
                "[简单长期记忆] create 操作请求 global scope，已降级 personal: umo=%s",
                umo,
            )

        subjects = [str(s) for s in (op.get("subjects") or []) if s]
        subject = str(op.get("subject") or "") or (subjects[0] if subjects else "")
        allowed_sender_ids = list(op.get("sender_ids") or [])
        if parsed.session_type == "private" and "sender_ids" not in op:
            allowed_sender_ids = [parsed.session_id]
        if scope == MemoryScope.PERSONAL:
            try:
                subjects = validate_owner_ids(
                    subjects or ([subject] if subject else []), allowed_sender_ids
                )
            except ValueError:
                logger.warning("[简单长期记忆] create 拒绝个人记忆: untrusted_subject")
                return False
            subject = subjects[0]
        # 非个人记忆可用会话 ID 构造维护事件，但不能据此创建个人 owner。
        sender_id = subject or parsed.session_id
        if not sender_id:
            logger.warning("[简单长期记忆] create 操作缺少可归属的 sender: umo=%s", umo)
            return False
        event = _MaintenanceEvent(umo=umo, sender_id=sender_id)
        is_personal = scope == MemoryScope.PERSONAL
        try:
            extra = {"created_by": "maintenance_curator"}
            if op.get("_extract_id"):
                extra["extraction_id"] = op["_extract_id"]
            uri = await self._memory_mgr.store_memory(
                event=event,
                content=content,
                domain=str(op.get("domain") or "fact"),
                disclosure=str(op.get("disclosure") or ""),
                importance=int(op.get("importance", 3)),
                memory_scope=scope,
                subject=subject,
                entities=op.get("entities") or [],
                topics=op.get("topics") or [],
                owner_sender_id=subject if is_personal else None,
                owner_sender_ids=subjects if is_personal and subjects else None,
                extra_metadata=extra,
                allowed_sender_ids=allowed_sender_ids,
            )
            logger.debug(
                "[简单长期记忆] create 写入完成: uri=%s, scope=%s, umo=%s",
                uri,
                scope,
                umo,
            )
            return bool(uri)
        except Exception as e:
            logger.warning(
                "[简单长期记忆] create 写入失败: error_type=%s", type(e).__name__
            )
            return False

    async def _execute_merge(self, op: dict[str, Any]) -> bool:
        """执行 merge 操作（supersede 语义，委托 merge_memories 统一处理）。"""
        uris = op.get("uris", [])
        merged_content = op.get("merged_content", "")

        if not uris or not merged_content:
            logger.warning("[简单长期记忆] merge 操作缺少必要参数")
            return False

        result = await self._memory_mgr.merge_memories(
            source_uris=uris,
            merged_content=merged_content,
            reason=op.get("reason", "organizer merge"),
            created_by="organizer",
        )
        return result.get("success", False)

    async def _execute_archive(self, op: dict[str, Any]) -> bool:
        """执行 archive 操作（标记 deprecated）。"""
        uri = op.get("uri", "")
        reason = op.get("reason", "")

        if not uri:
            return False

        return await self._memory_mgr.deprecate_memory(uri, reason=reason or "archived")

    async def _execute_update(self, op: dict[str, Any]) -> bool:
        """执行 update 操作（通过 replace_memory 替换内容）。"""
        uri = op.get("uri", "")
        new_content = op.get("new_content", "")

        if not uri or not new_content:
            return False

        extraction_id = op.get("_extract_id")
        done = []
        if extraction_id:
            done = await self._memory_mgr.vec_db.document_storage.get_documents(
                metadata_filters={
                    "extraction_id": extraction_id,
                    "is_memory_record": True,
                },
                limit=1,
            )

        # 先拉取旧记忆的 metadata
        old_docs = await self._memory_mgr.vec_db.document_storage.get_documents(
            metadata_filters={"uri": uri, "is_memory_record": True},
            limit=1,
        )
        if not old_docs:
            if done:
                return True
            logger.warning(f"[简单长期记忆] update 找不到记忆: {uri}")
            return False

        raw_meta = old_docs[0].get("metadata", {})
        if isinstance(raw_meta, str):
            import json as _json

            try:
                raw_meta = _json.loads(raw_meta)
            except Exception:
                raw_meta = {}
        old_metadata = raw_meta if isinstance(raw_meta, dict) else {}
        if extraction_id:
            old_metadata = {**old_metadata, "extraction_id": extraction_id}
        try:
            new_uri = await self._memory_mgr.replace_memory(
                old_metadata=old_metadata,
                new_content=new_content,
                updated_by="maintenance_curator" if extraction_id else "organizer",
            )
            return bool(new_uri)
        except Exception as e:
            logger.warning(
                "[简单长期记忆] update 失败: error_type=%s", type(e).__name__
            )
            return False

    async def _execute_new_link(self, op: dict[str, Any]) -> bool:
        """执行 new_link 操作。"""
        source = op.get("source", "")
        target = op.get("target", "")
        relation = op.get("relation", "related")
        reason = op.get("reason", "")
        confidence = op.get("confidence", 1.0)

        if not source or not target:
            return False

        # 写入关联表
        if self._memory_mgr._link_manager:
            return await self._memory_mgr._link_manager.add_link(
                source_uri=source,
                target_uri=target,
                relation_type=relation,
                reason=reason,
                confidence=confidence,
                created_by="analyst",
            )
        else:
            logger.warning("[简单长期记忆] link_manager 不可用")
            return False
