"""后台整理调度器。

使用 AstrBot 的 cron_manager 注册定时任务，
管理整理团队的生命周期（注册/注销/执行）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from astrbot.api import logger

from .llm import MaintenanceLLM
from .runner import MaintenanceRunner

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
        kv_put: Any = None,
        kv_get: Any = None,
    ) -> None:
        self._context = context
        self._memory_mgr = memory_mgr
        self._config = config
        self._kv_put = kv_put
        self._kv_get = kv_get
        self._job_ids: list[str] = []
        self._running_tasks: set[str] = set()  # single-flight 防并发

        # 初始化 LLM 客户端和执行管线
        maintenance_model_id = config.get("maintenance_model_id", "")
        self._llm = MaintenanceLLM(
            context=context,
            default_model_id=maintenance_model_id,
            max_calls_per_cycle=config.get("maintenance_max_llm_calls", 50),
            llm_timeout=config.get("maintenance_llm_timeout", 120),
        )
        self._runner = MaintenanceRunner(
            context=context,
            memory_mgr=memory_mgr,
            llm=self._llm,
            config=config,
            kv_put=kv_put,
            kv_get=kv_get,
        )

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

        # 整理周期（完整管线：purge → organizer → analyst → reviewer）
        if self._config.get("maintenance_enabled", False):
            maintenance_cron = self._config.get("maintenance_cron", "0 3 * * *")
            try:
                job = await cron_mgr.add_basic_job(
                    name="memory_maintenance_cycle",
                    cron_expression=maintenance_cron,
                    handler=self._run_maintenance_cycle,
                    description="完整记忆整理周期（purge + 整理师 + 分析师 + 审核员）",
                    persistent=False,
                    enabled=True,
                )
                self._job_ids.append(job.job_id)
                logger.info(f"[简单长期记忆] 整理周期已注册: cron={maintenance_cron}")
            except Exception as e:
                logger.warning(f"[简单长期记忆] 注册整理周期任务失败: {e}")

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
            # 存在任何失败时抛出异常，让 cron manager 记录为失败
            if result.get("errors"):
                raise RuntimeError(f"purge 部分失败: {result['errors']}")
        except Exception as e:
            logger.warning(f"[简单长期记忆] 定时清理失败: {e}")
            raise  # 传播到 cron manager，记录为失败而非 completed
        finally:
            self._running_tasks.discard("purge")

    async def _run_maintenance_cycle(self, **kwargs: Any) -> None:
        """执行完整整理周期（single-flight）。"""
        if "maintenance_cycle" in self._running_tasks:
            logger.debug("[简单长期记忆] 整理周期已在运行，跳过")
            return
        self._running_tasks.add("maintenance_cycle")
        try:
            if not self._memory_mgr.is_kb_connected:
                logger.debug("[简单长期记忆] KB 未连接，跳过整理周期")
                return

            report = await self._runner.run_cycle()
            logger.info(
                f"[简单长期记忆] 整理周期完成: {report.session_id}, "
                f"purge={report.purge_result.get('purged', 0)}, "
                f"LLM 调用={report.llm_stats.get('calls', 0)}, "
                f"错误={len(report.errors)}"
            )
            if report.errors:
                raise RuntimeError(f"整理周期部分失败: {report.errors}")
        except Exception as e:
            logger.warning(f"[简单长期记忆] 整理周期失败: {e}")
            raise
        finally:
            self._running_tasks.discard("maintenance_cycle")
