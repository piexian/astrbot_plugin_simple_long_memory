"""后台整理调度器。

使用 AstrBot 的 cron_manager 注册定时任务，
管理整理团队的生命周期（注册/注销/执行）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from astrbot.api import logger

if TYPE_CHECKING:
    from astrbot.core.star.context import Context

    from ..memory_manager import MemoryManager


class MaintenanceScheduler:
    """后台整理调度器，管理所有定时整理任务。"""

    def __init__(
        self,
        context: Context,
        memory_mgr: MemoryManager,
        config: dict[str, Any],
    ) -> None:
        self._context = context
        self._memory_mgr = memory_mgr
        self._config = config
        self._job_ids: list[str] = []
        self._running_tasks: set[str] = set()  # single-flight 防并发

    async def start(self) -> None:
        """注册所有定时任务到 cron_manager。"""
        if not self._config.get("maintenance_enabled", False):
            logger.debug("[简单长期记忆] 后台整理未启用")
            return

        cron_mgr = self._context.cron_manager
        if not cron_mgr:
            logger.warning("[简单长期记忆] cron_manager 不可用，后台整理无法启动")
            return

        # 自动清理（purge）
        if self._config.get("auto_purge_enabled", True):
            purge_days = self._config.get("auto_purge_after_days", 7)
            # 默认每天 05:00 执行
            purge_cron = self._config.get("auto_purge_cron", "0 5 * * *")
            try:
                job = await cron_mgr.add_basic_job(
                    name="memory_purge",
                    cron_expression=purge_cron,
                    handler=self._run_purge,
                    description="物理清理废弃超期记忆",
                    persistent=False,
                    enabled=True,
                    payload={"after_days": purge_days},
                )
                self._job_ids.append(job.job_id)
                logger.info(
                    f"[简单长期记忆] 自动清理已注册: cron={purge_cron}, "
                    f"超期={purge_days}天"
                )
            except Exception as e:
                logger.warning(f"[简单长期记忆] 注册自动清理任务失败: {e}")

        # TODO: Phase 3+ 注册整理师/分析师/审核员任务

    async def stop(self) -> None:
        """注销所有定时任务。"""
        cron_mgr = self._context.cron_manager
        if not cron_mgr:
            return
        remaining: list[str] = []
        for job_id in self._job_ids:
            try:
                await cron_mgr.delete_job(job_id)
            except Exception as e:
                logger.warning(f"[简单长期记忆] 注销任务失败: {job_id}, {e}")
                remaining.append(job_id)
        self._job_ids = remaining
        if not remaining:
            logger.info("[简单长期记忆] 后台整理调度器已停止")

    async def _run_purge(self, **kwargs: Any) -> None:
        """执行物理清理（single-flight）。"""
        if "purge" in self._running_tasks:
            logger.debug("[简单长期记忆] 清理任务已在运行，跳过")
            return
        self._running_tasks.add("purge")
        try:
            after_days = kwargs.get("after_days") or self._config.get(
                "auto_purge_after_days", 7
            )
            if not self._memory_mgr.is_kb_connected:
                logger.debug("[简单长期记忆] KB 未连接，跳过清理")
                return
            result = await self._memory_mgr.purge_deprecated(after_days=after_days)
            if result["purged"] > 0:
                logger.info(
                    f"[简单长期记忆] 定时清理完成: "
                    f"{result['purged']} 条记忆, "
                    f"{result['links_cleaned']} 条关联"
                )
        except Exception as e:
            logger.warning(f"[简单长期记忆] 定时清理失败: {e}")
        finally:
            self._running_tasks.discard("purge")
