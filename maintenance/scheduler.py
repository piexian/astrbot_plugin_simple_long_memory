"""后台整理调度器。

插件自管 asyncio 后台定时任务，不写 AstrBot cron_manager 的 DB，
避免重启后产生重复 job 堆积且无法触发的问题。
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from apscheduler.triggers.cron import CronTrigger

from astrbot.api import logger

from .llm import MaintenanceLLM
from .runner import MaintenanceRunner

if TYPE_CHECKING:
    from astrbot.core.star.context import Context

    from ..memory_manager import MemoryManager

# AstrBot cron_manager 使用的星期格式为 sun=0/7，此处保持一致
_WEEKDAY_NAMES = ("sun", "mon", "tue", "wed", "thu", "fri", "sat")
_WEEKDAY_RE = re.compile(r"^(?:(\*)|(\d+)(?:-(\d+))?)(?:/(\d+))?$")


def _normalize_dow(day_of_week: str) -> str:
    """将数字星期转为 APScheduler 友好的命名格式。"""
    parts: list[str] = []
    for raw in day_of_week.split(","):
        p = raw.strip().lower()
        m = _WEEKDAY_RE.fullmatch(p)
        if not m:
            parts.append(p)
            continue
        wildcard, start_t, end_t, step_t = m.groups()
        step = int(step_t or "1")
        if wildcard:
            parts.append("*" if step == 1 else f"*/{step}")
            continue
        start = int(start_t)
        end = int(end_t) if end_t else None
        if end is None:
            end = 7 if step_t else start
        names = [_WEEKDAY_NAMES[v if v != 7 else 0] for v in range(start, end + 1, step)]
        parts.append(",".join(dict.fromkeys(names)))
    return ",".join(parts)


def _next_run(cron_expression: str, after: datetime | None = None) -> datetime:
    """根据 cron 表达式计算下次执行时间（UTC）。"""
    minute, hour, day, month, dow = cron_expression.split()
    trigger = CronTrigger.from_crontab(
        " ".join([minute, hour, day, month, _normalize_dow(dow)])
    )
    base = after or datetime.now(timezone.utc)
    return trigger.get_next_fire_time(None, base)


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
        self._running_tasks: set[str] = set()  # single-flight 防并发
        self._loop_tasks: set[asyncio.Task] = set()

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

    @property
    def runner(self) -> MaintenanceRunner:
        """执行管线（供人工审批等外部入口使用，如 /memory review approve）。"""
        return self._runner

    async def start(self) -> None:
        """启动 asyncio 后台定时循环。"""
        if not self._config.get("maintenance_enabled", False):
            logger.debug("[简单长期记忆] 后台整理未启用")
            return

        # 自动清理（purge）
        if self._config.get("auto_purge_enabled", True):
            purge_days = self._config.get("auto_purge_after_days", 7)
            purge_cron = self._config.get("auto_purge_cron", "0 5 * * *")
            task = asyncio.create_task(
                self._loop(
                    name="memory_purge",
                    cron_expression=purge_cron,
                    handler=self._run_purge,
                    payload={"after_days": purge_days},
                )
            )
            self._loop_tasks.add(task)
            task.add_done_callback(self._loop_tasks.discard)
            logger.info(
                f"[简单长期记忆] 自动清理已启动: cron={purge_cron}, 超期={purge_days}天"
            )

        # 整理周期（完整管线：purge → organizer → analyst → reviewer）
        if self._config.get("maintenance_enabled", False):
            maintenance_cron = self._config.get("maintenance_cron", "0 3 * * *")
            task = asyncio.create_task(
                self._loop(
                    name="memory_maintenance_cycle",
                    cron_expression=maintenance_cron,
                    handler=self._run_maintenance_cycle,
                )
            )
            self._loop_tasks.add(task)
            task.add_done_callback(self._loop_tasks.discard)
            logger.info(f"[简单长期记忆] 整理周期已启动: cron={maintenance_cron}")

    async def stop(self) -> None:
        """取消所有后台定时循环。"""
        for task in self._loop_tasks:
            task.cancel()
        # 等待取消完成
        for task in list(self._loop_tasks):
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._loop_tasks.clear()
        logger.info("[简单长期记忆] 后台整理调度器已停止")

    async def _loop(
        self,
        *,
        name: str,
        cron_expression: str,
        handler: Any,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """按 cron 表达式定时执行 handler 的后台循环。"""
        while True:
            try:
                next_run = _next_run(cron_expression)
                now = datetime.now(timezone.utc)
                delay = (next_run - now).total_seconds()
                if delay > 0:
                    await asyncio.sleep(delay)
                # 二次校验，防止 sleep 期间配置变更
                result = handler(**payload) if payload else handler()
                if asyncio.iscoroutine(result):
                    await result
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"[简单长期记忆] {name} 执行失败: {e}")
                # 出错后短暂等待避免密集重试
                await asyncio.sleep(60)

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
            if result.get("errors"):
                raise RuntimeError(f"purge 部分失败: {result['errors']}")
        except Exception as e:
            logger.warning(f"[简单长期记忆] 定时清理失败: {e}")
            raise
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
