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
from .agents.segmenter import CARRY_KV_KEY, CURSOR_KV_KEY, SegmenterAgent
from .extraction_journal import ExtractionJournal
from .llm import MaintenanceLLM

try:  # 正常插件加载（包内相对导入）
    from ..extraction_utils import normalize_extracted_scope
    from ..memory_protocol import MemoryScope, UMOInfo, normalize_memory_scope
except ImportError:  # 测试环境：仓库根作为顶层目录直接在 sys.path
    from extraction_utils import normalize_extracted_scope
    from memory_protocol import MemoryScope, UMOInfo, normalize_memory_scope


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

            # 重置 LLM 统计
            self._llm.reset_cycle_stats()
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
                    report.analyst_manifest = await self._run_analyst()
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
        self._llm.reset_cycle_stats()
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
                analyst = await self._run_analyst()
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
                            f"fail closed 转待审"
                        )

            # 按审核结果执行操作
            for i, op in enumerate(all_operations):
                # 查找对应的审核结果
                verdict = None
                for v in report.reviewer_verdicts:
                    if v.get("index") == i:
                        verdict = v
                        break

                # 缺失裁决 → fail closed，拒绝执行（避免 reviewer 故障时放行破坏性操作）
                if verdict is None:
                    # additive 操作（new_link/create）仅在审核员整体禁用时直接放行；
                    # 审核员启用却缺失裁决属于异常（LLM 漏判），同样 fail closed 转待审
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
                            report.errors.append(f"op[{i}] {op.get('type')}: {e}")
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
                        logger.warning(f"[简单长期记忆] 执行操作失败: {op}, {e}")
                        failed += 1
                        report.errors.append(f"op[{i}] {op.get('type')}: {e}")
                else:
                    # 审核拒绝 → 跳过
                    skipped += 1
                    if verdict.get("needs_human_review"):
                        # 争议项 → 写入 KV 待审队列
                        pending_items.append((op, verdict))
                        logger.info(
                            f"[简单长期记忆] 争议项待人工审核: {op.get('type')}, "
                            f"理由: {verdict.get('reason', '')}"
                        )
                    elif verdict.get("verdict") == "reject":
                        await self._extract_journal.acknowledge(op)

            persisted = await self._flush_pending_review(
                pending_items, report.session_id
            )
            for op, _ in pending_items:
                if self._op_signature(op) in persisted:
                    await self._extract_journal.acknowledge(op)
            await self._finish_extract_progress()

            report.executed_ops = executed
            report.skipped_ops = skipped
            report.failed_ops = failed
            # LLM 统计
            report.llm_stats = self._llm.stats()
            logger.debug(
                "[简单长期记忆] 执行阶段完成: executed=%s, skipped=%s, failed=%s, "
                "pending=%s, llm=%s",
                executed,
                skipped,
                failed,
                len(pending_items),
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
        """提取阶段：分段员切对话块 → 整理师提取 create/update 操作。

        LLM 预算占每周期上限的 maintenance_extract_llm_budget_ratio（默认 60%），
        分段最多使用提取预算的一半，其余留给整理师；两者也受全局预算约束。
        """
        manifest = AgentManifest(agent_type="extract")
        journal = (
            ExtractionJournal(self._kv_get, None) if dry_run else self._extract_journal
        )
        if await journal.load():
            manifest.parsed = True
            manifest.operations = journal.pending
            manifest.metrics = journal.state.get("metrics", {})
            manifest.notes = journal.state.get("notes", "")
            return manifest
        max_calls = int(self._config.get("maintenance_max_llm_calls", 50))
        ratio = float(self._config.get("maintenance_extract_llm_budget_ratio", 0.6))
        extract_budget = max(0, int(max_calls * ratio))

        segmenter = SegmenterAgent(
            context=self._context,
            llm=self._llm,
            config=self._config,
            kv_get=self._kv_get,
            kv_put=self._kv_put,
        )
        seg_result = await segmenter.collect_blocks(extract_budget // 2)
        seg_stats = seg_result.get("stats", {})
        blocks = seg_result.get("blocks", [])

        curator_budget = max(0, extract_budget - int(seg_stats.get("llm_calls", 0)))
        curator = CuratorAgent(
            context=self._context,
            memory_mgr=self._memory_mgr,
            llm=self._llm,
            config=self._config,
        )
        cur_result = await curator.run(blocks, curator_budget)

        manifest.parsed = True
        manifest.notes = str(cur_result.get("notes", "") or "")
        manifest.operations.extend(cur_result.get("create", []))
        manifest.operations.extend(cur_result.get("update", []))
        manifest.metrics = {
            "segmenter": seg_stats,
            "blocks": len(blocks),
            "blocks_processed": cur_result.get("blocks_processed", 0),
            "blocks_nothing": cur_result.get("blocks_nothing", 0),
            "created": len(cur_result.get("create", [])),
            "updated": len(cur_result.get("update", [])),
            "llm_calls": int(seg_stats.get("llm_calls", 0))
            + cur_result.get("llm_calls", 0),
        }
        if not dry_run:
            await journal.prepare(seg_result, cur_result, manifest.metrics)
            manifest.operations = journal.pending
            await self._finish_extract_progress()
        logger.debug(
            "[简单长期记忆] 提取阶段完成: blocks=%s, processed=%s, create=%s, "
            "update=%s, llm_calls=%s, dry_run=%s",
            len(blocks),
            manifest.metrics["blocks_processed"],
            manifest.metrics["created"],
            manifest.metrics["updated"],
            manifest.metrics["llm_calls"],
            dry_run,
        )
        return manifest

    async def _finish_extract_progress(self) -> None:
        journal = self._extract_journal
        if journal.state:
            await self._commit_extract_progress(
                journal.state["segment"], journal.outcomes()
            )
            await journal.clear_if_finished()

    async def _commit_extract_progress(
        self,
        seg_result: dict[str, Any],
        outcomes: list[tuple[Any, str]],
    ) -> None:
        """按块提交提取游标并应用 carry 更新（整字典读改写，数据量小）。

        游标只随已持久化完成的连续块推进，失败或未完成块及其后续块保留；
        无产出块会话（短块/毒块跳过）直接采用分段员游标，避免每周期重扫。
        """
        if not self._kv_put or not self._kv_get:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            cursors = await self._kv_get(CURSOR_KV_KEY, None)
            cursors = dict(cursors) if isinstance(cursors, dict) else {}
            for key, value in seg_result.get("discovered", {}).items():
                cursors.setdefault(key, value)
            if seg_result.get("discovered"):
                await self._kv_put(CURSOR_KV_KEY, dict(cursors))
            grouped: dict[str, list[tuple[Any, str]]] = {}
            for block, outcome in outcomes:
                grouped.setdefault(block.conv_key, []).append((block, outcome))
            for conv_key, items in grouped.items():
                for block, outcome in items:
                    if outcome not in {"created", "updated", "nothing"}:
                        break
                    if getattr(block, "source", "pmh") == "conv2":
                        # conv2 无行 id，游标为内容锚点 + 消息下标
                        cursors[conv_key] = {
                            "source": "conv2",
                            "anchor": str(getattr(block, "end_anchor", "") or ""),
                            "idx": block.last_id,
                            "ts": now_iso,
                        }
                    else:
                        cursors[conv_key] = {"id": block.last_id, "ts": now_iso}
                    if getattr(block, "last_offset", 0):
                        cursors[conv_key]["offset"] = block.last_offset
                    # 逐块提交：崩溃后已处理块不重拉、未处理块不丢
                    await self._kv_put(CURSOR_KV_KEY, dict(cursors))
            block_convs = set(grouped)
            for conv_key, value in (seg_result.get("cursor_updates") or {}).items():
                if conv_key in block_convs:
                    continue  # 有块会话由 outcome 驱动，避免跳过未提取块
                if isinstance(value, dict) and value.get("source") == "conv2":
                    cursors[conv_key] = {
                        "source": "conv2",
                        "anchor": str(value.get("anchor") or ""),
                        "idx": int(value.get("idx") or 0),
                        "ts": now_iso,
                    }
                    if value.get("offset"):
                        cursors[conv_key]["offset"] = value["offset"]
                elif isinstance(value, dict):
                    cursors[conv_key] = {**value, "ts": now_iso}
                else:
                    cursors[conv_key] = {"id": value, "ts": now_iso}
                await self._kv_put(CURSOR_KV_KEY, dict(cursors))

            carry_updates = seg_result.get("carry_updates") or {}
            if carry_updates:
                carries = await self._kv_get(CARRY_KV_KEY, None)
                carries = dict(carries) if isinstance(carries, dict) else {}
                for conv_key, value in carry_updates.items():
                    if value is None:
                        carries.pop(conv_key, None)  # None = 清除该键
                    else:
                        carries[conv_key] = value
                    await self._kv_put(CARRY_KV_KEY, carries)
        except Exception as e:
            logger.warning(f"[简单长期记忆] 提交提取进度失败: {e}")
            raise

    async def _run_organizer(self) -> AgentManifest:
        """运行整理师：去重合并、质量精炼。"""
        manifest = AgentManifest(agent_type="organizer")

        # 使用 OrganizerAgent 执行整理
        organizer = OrganizerAgent(
            context=self._context,
            memory_mgr=self._memory_mgr,
            llm=self._llm,
            config=self._config,
        )

        try:
            result = await organizer.run()
            manifest.metrics = {
                key: result.get(key)
                for key in (
                    "memory_count",
                    "vector_count",
                    "vector_missing",
                    "candidates_screened",
                    "llm_calls",
                )
            }
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
            manifest.notes = result.get("notes", "")
        except Exception as e:
            manifest.error = str(e)
            logger.warning(f"[简单长期记忆] 整理师执行失败: {e}")

        return manifest

    async def _run_analyst(self) -> AgentManifest:
        """运行分析师：关联发现、矛盾检测。"""
        manifest = AgentManifest(agent_type="analyst")

        # 使用 AnalystAgent 执行分析
        analyst = AnalystAgent(
            context=self._context,
            memory_mgr=self._memory_mgr,
            llm=self._llm,
            config=self._config,
        )

        try:
            result = await analyst.run()
            manifest.metrics = {
                key: result.get(key)
                for key in (
                    "memory_count",
                    "vector_count",
                    "vector_missing",
                    "conversation_chars",
                    "candidates_screened",
                    "llm_calls",
                )
            }
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
            manifest.notes = result.get("notes", "")
        except Exception as e:
            manifest.error = str(e)
            logger.warning(f"[简单长期记忆] 分析师执行失败: {e}")

        return manifest

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
        )

        # 源记录由审核员按操作逐条拉取（见 ReviewerAgent._get_op_source_data）
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
        """周期结束批量写入待审队列：单次 KV 读写 + 签名去重 + 终态清理 + pending 容量上限 + 单调 id。"""
        if not items or not self._kv_put or not self._kv_get:
            return set()
        try:
            import time as _time

            queue = await self._kv_get("maintenance_pending_review", None) or []
            # A prior queue write may have succeeded before its journal acknowledgement.
            settled_sigs = {
                it.get("op_signature") or self._op_signature(it["op"])
                for it in queue
                if it.get("status") in {"approved", "rejected"}
                and it.get("op", {}).get("_extract_id")
            }

            # 单调递增 id：从原始队列取最大 id 作为下限，清理终态后 id 不复用，
            # 避免与管理员已见条目撞号
            max_seen_id = max((it.get("id", 0) for it in queue), default=0)
            seq = await self._kv_get("maintenance_pending_review_seq", None)
            if not isinstance(seq, int) or seq < max_seen_id:
                seq = max_seen_id

            # 清理终态条目（approved/rejected/failed），只保留 pending
            pruned = sum(1 for it in queue if it.get("status") != "pending")
            if pruned:
                queue = [it for it in queue if it.get("status") == "pending"]

            existing_sigs = {
                it.get("op_signature") or self._op_signature(it.get("op", {}))
                for it in queue
            } | settled_sigs
            queue_max = self._config.get("maintenance_pending_queue_max", 500)

            added = 0
            deduped = 0
            dropped = 0
            for op, verdict in items:
                sig = self._op_signature(op)
                if sig in existing_sigs:
                    deduped += 1
                    continue
                # 容量只统计 pending（此处队列已清理终态，len 即 pending 数）
                if len(queue) >= queue_max:
                    dropped += 1
                    continue
                seq += 1
                queue.append(
                    {
                        "id": seq,
                        "session_id": session_id,
                        "op_type": op.get("type", ""),
                        "op": op,
                        "op_signature": sig,
                        "verdict_reason": verdict.get("reason", ""),
                        "controversial": verdict.get("controversial", False),
                        "created_at": _time.strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "pending",  # pending / approved / rejected
                    }
                )
                existing_sigs.add(sig)
                added += 1

            if added or pruned:
                await self._kv_put("maintenance_pending_review", queue)
            if added:
                await self._kv_put("maintenance_pending_review_seq", seq)
            logger.info(
                f"[简单长期记忆] 待审入队: 新增 {added}, 去重跳过 {deduped}, "
                f"队列满丢弃 {dropped}, 清理终态 {pruned}, 当前待审 {len(queue)} 条"
            )
            # 待审通知推送（每周期最多一次）
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
            from astrbot.api.message_components import Plain
            from astrbot.api.star import MessageChain

            msg = MessageChain(
                chain=[
                    Plain(
                        text=f"📝 记忆整理待审通知\n当前有 {pending_count} 条争议操作等待审查。\n回复 /memory review 查看详情并处理"
                    )
                ]
            )
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
            logger.warning(f"[简单长期记忆] 执行操作失败: {op_type}, {e}")
            return False

    async def _execute_create(self, op: dict[str, Any]) -> bool:
        """执行 create 操作（整理师从对话块提取的新记忆）。"""
        content = str(op.get("content") or "").strip()
        umo = str(op.get("umo") or "")
        if not content or not umo:
            return False

        parsed = UMOInfo.parse(umo)
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
        if scope == MemoryScope.PERSONAL and "sender_ids" in op:
            if not set(subjects or [subject]).issubset(op["sender_ids"]):
                return False
        # 私聊无 subject 时以 session_id 兜底（私聊 session_id 即对端用户 id），
        # 避免 store_memory 的 owner 推导拿到空 sender
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
            )
            logger.debug(
                "[简单长期记忆] create 写入完成: uri=%s, scope=%s, umo=%s",
                uri,
                scope,
                umo,
            )
            return bool(uri)
        except Exception as e:
            logger.warning(f"[简单长期记忆] create 写入失败: {e}")
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
            logger.warning(f"[简单长期记忆] update 失败: {uri}, {e}")
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
