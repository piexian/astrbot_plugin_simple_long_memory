"""后台整理执行管线。

一次整理周期 = 一条顺序管线：
  purge（纯逻辑）→ organizer → analyst → reviewer（复核前两个角色的 manifest）

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
from .agents.organizer import OrganizerAgent
from .agents.reviewer import ReviewerAgent
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
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
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
                "ops_count": len(self.organizer_manifest.operations)
                if self.organizer_manifest
                else 0,
                "parsed": self.organizer_manifest.parsed
                if self.organizer_manifest
                else False,
                "error": self.organizer_manifest.error
                if self.organizer_manifest
                else "",
            },
            "analyst": {
                "ops_count": len(self.analyst_manifest.operations)
                if self.analyst_manifest
                else 0,
                "parsed": self.analyst_manifest.parsed
                if self.analyst_manifest
                else False,
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
                    purge_result = await self._memory_mgr.purge_deprecated(
                        after_days=purge_days
                    )
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
            await self._execute_operations(report)

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

    async def _execute_operations(self, report: MaintenanceReport) -> None:
        """阶段 4：按审核结果执行操作。

        与人工审批（execute_approved）共用 _op_lock，避免与审批路径并发写竞态。
        """
        async with self._op_lock:
            executed = 0
            skipped = 0
            failed = 0

            # 收集所有操作和对应的审核结果
            all_operations = []
            if report.organizer_manifest and report.organizer_manifest.parsed:
                all_operations.extend(report.organizer_manifest.operations)
            if report.analyst_manifest and report.analyst_manifest.parsed:
                all_operations.extend(report.analyst_manifest.operations)

            reviewer_enabled = self._config.get("maintenance_reviewer_enabled", True)
            # reviewer 启用时裁决数应与操作数一致，不一致说明 LLM 漏判/乱序，提前告警
            if reviewer_enabled and all_operations:
                verdict_indices = {v.get("index") for v in report.reviewer_verdicts}
                missing = [
                    i for i in range(len(all_operations)) if i not in verdict_indices
                ]
                if missing:
                    logger.warning(
                        f"[简单长期记忆] 审核员启用但缺失 {len(missing)} 条裁决"
                        f"（共 {len(all_operations)} 条操作），相关操作将 fail closed 转待审"
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
                    # additive 操作（new_link）仅在审核员整体禁用时直接放行；
                    # 审核员启用却缺失裁决属于异常（LLM 漏判），同样 fail closed 转待审
                    if op.get("type") == "new_link" and not reviewer_enabled:
                        try:
                            success = await self._execute_operation(op)
                            if success:
                                executed += 1
                            else:
                                failed += 1
                                report.errors.append(f"op[{i}] new_link 执行返回失败")
                        except Exception as e:
                            failed += 1
                            report.errors.append(f"op[{i}] new_link: {e}")
                        continue
                    # 其余情况（破坏性操作，或审核员启用但裁决缺失）→ 转待审队列
                    reason = (
                        "reviewer 未启用"
                        if not reviewer_enabled
                        else "reviewer 缺失裁决（LLM 漏判或输出异常）"
                    )
                    await self._enqueue_pending_review(
                        op,
                        {"verdict": "pending", "reason": reason},
                        report.session_id,
                    )
                    skipped += 1
                    logger.info(
                        f"[简单长期记忆] 操作 {i} 缺少审核裁决，转待审: {op.get('type')}"
                    )
                    continue

                if verdict.get("verdict") == "approve":
                    # global 记忆操作必须经管理员人工确认（不可配置）
                    if await self._op_touches_global(op):
                        await self._enqueue_pending_review(
                            op, verdict, report.session_id
                        )
                        skipped += 1
                        logger.info(
                            f"[简单长期记忆] 操作 {i} 涉及 global 记忆，转人工审批"
                        )
                        continue
                    try:
                        success = await self._execute_operation(op)
                        if success:
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
                        await self._enqueue_pending_review(
                            op, verdict, report.session_id
                        )
                        logger.info(
                            f"[简单长期记忆] 争议项待人工审核: {op.get('type')}, "
                            f"理由: {verdict.get('reason', '')}"
                        )

            report.executed_ops = executed
            report.skipped_ops = skipped
            report.failed_ops = failed
            # LLM 统计
            report.llm_stats = self._llm.stats()

    async def execute_approved(self, op: dict[str, Any]) -> bool:
        """公开入口：执行一条经管理员批准的操作。

        与整理周期阶段 4 共用 _op_lock，避免与后台周期并发写竞态。
        供 /memory review approve 等外部命令调用，不应用 _execute_operation。
        """
        async with self._op_lock:
            return await self._execute_operation(op)

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

        # 拉取操作涉及的源记录，供审核员对比
        original_data: dict[str, Any] = {"memories": {}}
        referenced_uris: set[str] = set()
        for op in proposed_changes:
            for key in ("uris", "uri", "source", "target", "old_uri", "new_uri"):
                val = op.get(key)
                if isinstance(val, list):
                    referenced_uris.update(val)
                elif isinstance(val, str) and val:
                    referenced_uris.add(val)
        for uri in list(referenced_uris)[:20]:
            try:
                mem = await self._memory_mgr._get_memory_by_uri(uri)
                if mem:
                    original_data["memories"][uri] = {
                        "content": mem.get("content", ""),
                        "metadata": mem.get("metadata", {}),
                    }
            except Exception:
                pass

        try:
            verdicts = await reviewer.review(
                proposed_changes, original_data=original_data
            )
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

    async def _enqueue_pending_review(
        self, op: dict[str, Any], verdict: dict[str, Any], session_id: str
    ) -> None:
        """将争议操作写入 KV 待审队列。"""
        if not self._kv_put or not self._kv_get:
            return
        try:
            queue = await self._kv_get("maintenance_pending_review", None) or []
            import time as _time

            queue.append(
                {
                    "id": len(queue) + 1,
                    "session_id": session_id,
                    "op_type": op.get("type", ""),
                    "op": op,
                    "verdict_reason": verdict.get("reason", ""),
                    "controversial": verdict.get("controversial", False),
                    "created_at": _time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "pending",  # pending / approved / rejected
                }
            )
            await self._kv_put("maintenance_pending_review", queue)
            # 待审通知推送
            if self._config.get("review_notify_enabled", True):
                await self._send_review_notification(len(queue))
        except Exception as e:
            logger.warning(f"[简单长期记忆] 写入待审队列失败: {e}")

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
                logger.info(
                    f"[简单长期记忆] 检测到矛盾: {op.get('old_uri')} vs {op.get('new_uri')}"
                )
                return True
            else:
                logger.warning(f"[简单长期记忆] 未知操作类型: {op_type}")
                return False
        except Exception as e:
            logger.warning(f"[简单长期记忆] 执行操作失败: {op_type}, {e}")
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

        # 先拉取旧记忆的 metadata
        old_docs = await self._memory_mgr.vec_db.document_storage.get_documents(
            metadata_filters={"uri": uri, "is_memory_record": True},
            limit=1,
        )
        if not old_docs:
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
        try:
            new_uri = await self._memory_mgr.replace_memory(
                old_metadata=old_metadata,
                new_content=new_content,
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
