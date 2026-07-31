"""后台整理执行管线。

一次整理周期 = 一条顺序管线：
  purge（纯逻辑）→ organizer → analyst → reviewer（复核前两个角色的 manifest）

所有 Agent 只输出结构化 manifest，Host 解析后才执行 DB 写入。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from astrbot.api import logger

from .agents.analyst import AnalystAgent
from .agents.organizer import OrganizerAgent
from .llm import MaintenanceLLM
from .prompts import build_prompt, DEFAULT_REVIEWER_PROMPT

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

            # ── 阶段 4: Host 执行（Phase 3/4/5 实现具体操作执行）──
            # 目前只记录 manifest，不执行实际操作
            report.executed_ops = 0
            report.skipped_ops = (
                len(report.organizer_manifest.operations) if report.organizer_manifest else 0
            ) + (
                len(report.analyst_manifest.operations) if report.analyst_manifest else 0
            )

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

        # 组装 prompt
        variables = {
            "proposed_changes": json.dumps(proposed_changes, ensure_ascii=False, indent=2),
            "original_data": "{}",  # Phase 5 完善
            "persona_summary": await self._get_persona_summary(),
            "admin_guides": "",
        }
        prompt = build_prompt(
            default_template=DEFAULT_REVIEWER_PROMPT,
            variables=variables,
            prompt_override=self._config.get("maintenance_reviewer_prompt_override", ""),
            prompt_extra=self._config.get("maintenance_reviewer_prompt_extra", ""),
        )

        # 调用 LLM
        model_id = self._config.get("maintenance_reviewer_model_id", "") or self._config.get("maintenance_model_id", "")
        raw = await self._llm._chat(
            system_prompt="你是记忆操作审核员，只输出结构化 JSON。",
            user_prompt=prompt,
            model_id=model_id,
        )

        if raw:
            parsed = self._llm._parse_json(raw)
            if parsed:
                verdicts = parsed.get("verdicts", [])

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
