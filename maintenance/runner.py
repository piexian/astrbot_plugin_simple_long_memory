"""后台整理执行管线。

一次整理周期 = 一条顺序管线：
  purge（纯逻辑）→ organizer → analyst → reviewer（复核前两个角色的 manifest）

所有 Agent 只输出结构化 manifest，Host 解析后才执行 DB 写入。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from astrbot.api import logger

from .agents.reviewer import ReviewerAgent
from .agents.analyst import AnalystAgent
from .agents.organizer import OrganizerAgent
from .llm import MaintenanceLLM

@dataclass
class AgentManifest:
    """Agent 输出的结构化操作清单。"""

    agent_type: str  # organizer / analyst / reviewer
    operations: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    raw_response: str = ""
    parsed: bool = False
    error: str = ""


@dataclass
class MaintenanceReport:
    """一次整理周期的结构化报告。"""

    session_id: str
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str = ""
    duration_ms: float = 0.0

    # 各阶段统计
    purge_result: dict[str, int] = field(default_factory=dict)
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
            "purge_result": self.purge_result,
            "organizer": {
                "ops_count": len(self.organizer_manifest.operations) if self.organizer_manifest else 0,
                "parsed": self.organizer_manifest.parsed if self.organizer_manifest else False,
                "error": self.organizer_manifest.error if self.organizer_manifest else "",
            },
            "analyst": {
                "ops_count": len(self.analyst_manifest.operations) if self.analyst_manifest else 0,
                "parsed": self.analyst_manifest.parsed if self.analyst_manifest else False,
                "error": self.analyst_manifest.error if self.analyst_manifest else "",
            },
            "reviewer_verdicts_count": len(self.reviewer_verdicts),
            "llm_stats": self.llm_stats,
            "executed_ops": self.executed_ops,
            "skipped_ops": self.skipped_ops,
            "failed_ops": self.failed_ops,
            "errors": self.errors,
        }


class MaintenanceRunner:
    """后台整理执行管线。"""

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
        self._running = False

    async def run_cycle(self) -> MaintenanceReport:
        """执行一次完整整理周期。"""
        if self._running:
            logger.warning("[简单长期记忆] 整理周期已在运行，跳过")
            raise RuntimeError("maintenance cycle already running")

        self._running = True
        session_id = f"maint-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        report = MaintenanceReport(session_id=session_id)
        started = datetime.now(timezone.utc)

        try:
            logger.info(f"[简单长期记忆] 开始整理周期: {session_id}")

            # 重置 LLM 统计
            self._llm.reset_cycle_stats()

            # ── 阶段 0: purge（纯逻辑，无 LLM）──
            if self._config.get("auto_purge_enabled", True):
                purge_days = self._config.get("auto_purge_after_days", 7)
                try:
                    purge_result = await self._memory_mgr.purge_deprecated(after_days=purge_days)
                    report.purge_result = purge_result
                    logger.info(
                        f"[简单长期记忆] purge 完成: {purge_result.get('purged', 0)} 条记忆"
                    )
                except Exception as e:
                    report.errors.append(f"purge: {e}")
                    logger.warning(f"[简单长期记忆] purge 失败: {e}")

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
                    report.reviewer_verdicts = await self._run_reviewer(
                        organizer_manifest=report.organizer_manifest,
                        analyst_manifest=report.analyst_manifest,
                    )
                except Exception as e:
                    report.errors.append(f"reviewer: {e}")
                    logger.warning(f"[简单长期记忆] reviewer 失败: {e}")

            # ── 阶段 3.5: 互审模式（驳回理由回传修正，最多 2 轮）──
            max_revision_rounds = 2
            for round_num in range(max_revision_rounds):
                # 检查是否有 reject 且 controversial 的项
                controversial_items = [
                    v for v in report.reviewer_verdicts
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
            executed = 0
            skipped = 0
            failed = 0

            # 收集所有操作和对应的审核结果
            all_operations = []
            if report.organizer_manifest and report.organizer_manifest.parsed:
                all_operations.extend(report.organizer_manifest.operations)
            if report.analyst_manifest and report.analyst_manifest.parsed:
                all_operations.extend(report.analyst_manifest.operations)

            # 按审核结果执行操作
            for i, op in enumerate(all_operations):
                # 查找对应的审核结果
                verdict = None
                for v in report.reviewer_verdicts:
                    if v.get("index") == i:
                        verdict = v
                        break

                # 无审核结果或审核通过 → 执行
                if verdict is None or verdict.get("verdict") == "approve":
                    try:
                        success = await self._execute_operation(op)
                        if success:
                            executed += 1
                        else:
                            failed += 1
                    except Exception as e:
                        logger.warning(f"[简单长期记忆] 执行操作失败: {op}, {e}")
                        failed += 1
                else:
                    # 审核拒绝 → 跳过
                    skipped += 1
                    if verdict.get("needs_human_review"):
                        # 争议项 → 写入待审队列（Phase 5 完善：接 KV 和通知）
                        logger.info(
                            f"[简单长期记忆] 争议项待人工审核: {op.get('type')}, "
                            f"理由: {verdict.get('reason', '')}"
                        )

            report.executed_ops = executed
            report.skipped_ops = skipped
            report.failed_ops = failed
            # LLM 统计
            report.llm_stats = self._llm.stats()

        except Exception as e:
            report.errors.append(f"cycle: {e}")
            logger.error(f"[简单长期记忆] 整理周期异常: {e}")
        finally:
            finished = datetime.now(timezone.utc)
            report.finished_at = finished.isoformat()
            report.duration_ms = (finished - started).total_seconds() * 1000
            self._running = False

            # 落 KV（Phase 2 先日志输出，Phase 3 接 KV）
            logger.info(
                f"[简单长期记忆] 整理周期完成: {session_id}, "
                f"耗时 {report.duration_ms:.0f}ms, "
                f"LLM 调用 {report.llm_stats.get('calls', 0)} 次"
            )

        return report

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
            manifest.parsed = True
            # 将 organizer 的输出转换为 operations 格式
            for merge_op in result.get("merge", []):
                manifest.operations.append({
                    "type": "merge",
                    "uris": merge_op.get("uris", []),
                    "merged_content": merge_op.get("merged_content", ""),
                    "reason": merge_op.get("reason", ""),
                    "confidence": merge_op.get("confidence", 0.0),
                })
            for archive_op in result.get("archive", []):
                manifest.operations.append({
                    "type": "archive",
                    "uri": archive_op.get("uri", ""),
                    "reason": archive_op.get("reason", ""),
                })
            for update_op in result.get("update", []):
                manifest.operations.append({
                    "type": "update",
                    "uri": update_op.get("uri", ""),
                    "new_content": update_op.get("new_content", ""),
                    "reason": update_op.get("reason", ""),
                })
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
            manifest.parsed = True
            # 将 analyst 的输出转换为 operations 格式
            for link_op in result.get("new_links", []):
                manifest.operations.append({
                    "type": "new_link",
                    "source": link_op.get("source", ""),
                    "target": link_op.get("target", ""),
                    "relation": link_op.get("relation", "related"),
                    "reason": link_op.get("reason", ""),
                    "confidence": link_op.get("confidence", 0.0),
                })
            for contra_op in result.get("contradictions", []):
                manifest.operations.append({
                    "type": "contradiction",
                    "old_uri": contra_op.get("old_uri", ""),
                    "new_uri": contra_op.get("new_uri", ""),
                    "reason": contra_op.get("reason", ""),
                    "confidence": contra_op.get("confidence", 0.0),
                })
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

        # 收集待审核的 manifest
        proposed_changes = []
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

        try:
            verdicts = await reviewer.review(proposed_changes)
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

    async def _execute_operation(self, op: dict[str, Any]) -> bool:
        """执行单个操作。"""
        op_type = op.get("type", "")

        try:
            if op_type == "merge":
                return await self._execute_merge(op)
            elif op_type == "archive":
                return await self._execute_archive(op)
            elif op_type == "update":
                return await self._execute_update(op)
            elif op_type == "new_link":
                return await self._execute_new_link(op)
            elif op_type == "contradiction":
                # 矛盾检测只记录，不执行实际操作
                logger.info(f"[简单长期记忆] 检测到矛盾: {op.get('old_uri')} vs {op.get('new_uri')}")
                return True
            else:
                logger.warning(f"[简单长期记忆] 未知操作类型: {op_type}")
                return False
        except Exception as e:
            logger.warning(f"[简单长期记忆] 执行操作失败: {op_type}, {e}")
            return False

    async def _execute_merge(self, op: dict[str, Any]) -> bool:
        """执行 merge 操作（supersede 语义）。"""
        uris = op.get("uris", [])
        merged_content = op.get("merged_content", "")

        if not uris or not merged_content:
            logger.warning("[简单长期记忆] merge 操作缺少必要参数")
            return False

        # 1. 新建合并后的记忆
        new_uri = f"facts://merged/{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        success = await self._memory_mgr.add_memory(
            content=merged_content,
            uri=new_uri,
            memory_type="fact",
            importance=3,
            scope="personal",
        )
        if not success:
            return False

        # 2. 旧记忆标 deprecated + 写 superseded_by 边
        for old_uri in uris:
            # 标记废弃
            await self._memory_mgr.mark_deprecated(old_uri, reason="merged")
            # 写 superseded_by 边
            if self._memory_mgr._link_manager:
                await self._memory_mgr._link_manager.add_link(
                    source_uri=new_uri,
                    target_uri=old_uri,
                    relation_type="supersedes",
                    reason=f"merged into {new_uri}",
                    confidence=op.get("confidence", 1.0),
                    created_by="organizer",
                )

        return True

    async def _execute_archive(self, op: dict[str, Any]) -> bool:
        """执行 archive 操作。"""
        uri = op.get("uri", "")
        reason = op.get("reason", "")

        if not uri:
            return False

        # 标记废弃
        return await self._memory_mgr.mark_deprecated(uri, reason=reason or "archived")

    async def _execute_update(self, op: dict[str, Any]) -> bool:
        """执行 update 操作。"""
        uri = op.get("uri", "")
        new_content = op.get("new_content", "")

        if not uri or not new_content:
            return False

        # 更新记忆内容（需要 memory_mgr 提供 update 接口）
        if hasattr(self._memory_mgr, "update_memory"):
            return await self._memory_mgr.update_memory(uri, new_content)
        else:
            logger.warning("[简单长期记忆] memory_mgr 缺少 update_memory 接口")
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
