"""后台整理调度器。

插件自管 asyncio 后台定时任务，不写 AstrBot cron_manager 的 DB，
避免重启后产生重复 job 堆积且无法触发的问题。
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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
            if step == 1:
                parts.append("*")
            else:
                # 展开 */N 为 sun=0..sat=6 序列，避免 APScheduler 按 mon=0 解释
                names = [_WEEKDAY_NAMES[v] for v in range(0, 7, step)]
                parts.append(",".join(dict.fromkeys(names)))
            continue
        start = int(start_t)
        end = int(end_t) if end_t else None
        if end is None:
            end = 7 if step_t else start
        names = [
            _WEEKDAY_NAMES[v if v != 7 else 0] for v in range(start, end + 1, step)
        ]
        parts.append(",".join(dict.fromkeys(names)))
    return ",".join(parts)


def _next_run(
    cron_expression: str,
    after: datetime | None = None,
    tz_str: str | None = None,
) -> datetime:
    """根据 cron 表达式计算下次执行时间。"""
    parts = cron_expression.split()
    if len(parts) != 5:
        raise ValueError(
            f"cron 表达式需为 5 段格式(min hour day month dow)，got: {cron_expression!r}"
        )
    minute, hour, day, month, dow = parts
    tzinfo = None
    if tz_str:
        try:
            tzinfo = ZoneInfo(tz_str)
        except ZoneInfoNotFoundError:
            logger.warning("[简单长期记忆] 未知时区 %s，回退到系统本地时区", tz_str)
    trigger = CronTrigger.from_crontab(
        " ".join([minute, hour, day, month, _normalize_dow(dow)]),
        timezone=tzinfo,
    )
    base = after or datetime.now(timezone.utc)
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)
    # APScheduler treats ``previous_fire_time`` as the lower bound. Passing
    # None on every iteration makes an exact fire time repeat forever.
    previous_fire_time = base if after is not None else None
    next_time = trigger.get_next_fire_time(previous_fire_time, base)
    if next_time is not None and next_time <= base:
        # Startup can happen at the exact cron boundary. Always return a
        # strictly future fire time so the asyncio loop cannot spin.
        next_time = trigger.get_next_fire_time(next_time, base)
    if next_time is None:
        raise ValueError(f"cron 表达式永不匹配: {cron_expression!r}")
    return next_time


def _parse_maintenance_window(window: str) -> tuple[int, int] | None:
    """解析 HH:MM 或 HH:MM-HH:MM，返回分钟数范围。"""
    raw = (window or "").strip()
    if not raw:
        return None
    parts = [part.strip() for part in raw.split("-", 1)]
    if len(parts) == 1:
        parts.append(parts[0])
    if len(parts) != 2:
        raise ValueError(f"整理时间窗口格式无效: {window!r}")

    result: list[int] = []
    for value in parts:
        parsed = datetime.strptime(value, "%H:%M")
        result.append(parsed.hour * 60 + parsed.minute)
    return result[0], result[1]


def _is_in_maintenance_window(now: datetime, window: str) -> bool:
    """判断当前本地时间是否位于整理窗口内。"""
    parsed = _parse_maintenance_window(window)
    if parsed is None:
        return True
    start, end = parsed
    current = now.hour * 60 + now.minute
    if start == end:
        return current == start
    if start < end:
        return start <= current <= end
    # 支持跨午夜窗口，例如 23:00-02:00。
    return current >= start or current <= end


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
        # 读取 AstrBot 主配置的时区，与旧 cron_manager 行为一致
        self._tz_str: str | None = None
        try:
            self._tz_str = context.astrbot_config.get("timezone") or None
        except Exception:
            pass

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

        maintenance_window = self._config.get("maintenance_window", "")
        try:
            _parse_maintenance_window(maintenance_window)
        except ValueError:
            logger.exception(
                "[简单长期记忆] 整理时间窗口无效: %r，后台任务不启动",
                maintenance_window,
            )
            return

        # 自动清理（purge）
        if self._config.get("auto_purge_enabled", True):
            purge_days = self._config.get("auto_purge_after_days", 7)
            purge_cron = self._config.get("auto_purge_cron", "0 5 * * *")
            if not self._validate_cron(purge_cron, "memory_purge"):
                return
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

        # 整理周期（完整管线：purge → extract → organizer → analyst → reviewer）
        if self._config.get("maintenance_enabled", False):
            maintenance_cron = self._config.get("maintenance_cron", "0 3 * * *")
            if not self._validate_cron(maintenance_cron, "memory_maintenance_cycle"):
                return
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
        tasks = list(self._loop_tasks)
        logger.debug(
            "[简单长期记忆] 开始停止后台调度器: tasks=%s, running=%s",
            len(tasks),
            sorted(self._running_tasks),
        )
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._loop_tasks.clear()
        logger.info("[简单长期记忆] 后台整理调度器已停止")

    def _validate_cron(self, cron_expression: str, name: str) -> bool:
        """启动时预校验 cron 表达式，失败返回 False。"""
        try:
            _next_run(cron_expression, tz_str=self._tz_str)
            return True
        except Exception as e:
            logger.error(
                f"[简单长期记忆] {name} cron 表达式无效: {cron_expression!r}, {e}，该任务不启动"
            )
            return False

    async def _loop(
        self,
        *,
        name: str,
        cron_expression: str,
        handler: Any,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """按 cron 表达式定时执行 handler 的后台循环。"""
        prev_run: datetime | None = None
        while True:
            try:
                loop_started = time.monotonic()
                now = datetime.now(timezone.utc)
                next_run = _next_run(
                    cron_expression, after=prev_run, tz_str=self._tz_str
                )
                delay = (next_run - now).total_seconds()
                if delay <= 0:
                    # Handler 执行过久或系统从睡眠中恢复时，跳过已经错过的
                    # fire time，重新计算严格晚于当前时刻的下一次执行。
                    logger.warning(
                        "[简单长期记忆] %s 错过调度时间，跳过本次补执行: "
                        "now=%s, next=%s, overdue=%.3fs",
                        name,
                        now.isoformat(),
                        next_run.isoformat(),
                        -delay,
                    )
                    prev_run = now
                    next_run = _next_run(
                        cron_expression, after=now, tz_str=self._tz_str
                    )
                    delay = (next_run - datetime.now(timezone.utc)).total_seconds()
                    if delay <= 0:
                        raise RuntimeError(
                            f"{name} 无法计算严格晚于当前时刻的下次执行时间: "
                            f"{next_run.isoformat()}"
                        )

                logger.debug(
                    "[简单长期记忆] %s 调度决策: cron=%r, timezone=%r, "
                    "previous=%s, next=%s, delay=%.3fs",
                    name,
                    cron_expression,
                    self._tz_str,
                    prev_run.isoformat() if prev_run else None,
                    next_run.isoformat(),
                    delay,
                )
                if delay > 0:
                    await asyncio.sleep(delay)
                    logger.debug(
                        "[简单长期记忆] %s 调度唤醒: scheduled=%s, actual=%s",
                        name,
                        next_run.isoformat(),
                        datetime.now(timezone.utc).isoformat(),
                    )
                prev_run = next_run
                handler_started = time.monotonic()
                logger.debug(
                    "[简单长期记忆] %s handler 开始: scheduled=%s, payload_keys=%s",
                    name,
                    next_run.isoformat(),
                    sorted((payload or {}).keys()),
                )
                result = handler(**payload) if payload else handler()
                if asyncio.iscoroutine(result):
                    await result
                logger.debug(
                    "[简单长期记忆] %s handler 完成: scheduled=%s, "
                    "duration_ms=%.1f, loop_ms=%.1f",
                    name,
                    next_run.isoformat(),
                    (time.monotonic() - handler_started) * 1000,
                    (time.monotonic() - loop_started) * 1000,
                )
            except asyncio.CancelledError:
                logger.debug("[简单长期记忆] %s 调度循环收到取消", name)
                raise
            except Exception:
                logger.exception(
                    "[简单长期记忆] %s 调度循环异常: cron=%r, previous=%s",
                    name,
                    cron_expression,
                    prev_run.isoformat() if prev_run else None,
                )
                await asyncio.sleep(60)

    async def _run_purge(self, **kwargs: Any) -> None:
        """执行物理清理（single-flight）。"""
        if "purge" in self._running_tasks:
            logger.debug("[简单长期记忆] 清理任务已在运行，跳过")
            return
        self._running_tasks.add("purge")
        started = time.monotonic()
        try:
            after_days = kwargs.get("after_days") or self._config.get(
                "auto_purge_after_days", 7
            )
            if not self._memory_mgr.is_kb_connected:
                logger.debug(
                    "[简单长期记忆] purge 跳过: kb_connected=False, after_days=%s",
                    after_days,
                )
                return
            logger.debug(
                "[简单长期记忆] purge 开始: after_days=%s, kb=%s",
                after_days,
                getattr(self._memory_mgr, "current_kb_name", ""),
            )
            result = await self._memory_mgr.purge_deprecated(after_days=after_days)
            logger.debug(
                "[简单长期记忆] purge 结果: result=%s, duration_ms=%.1f",
                result,
                (time.monotonic() - started) * 1000,
            )
            if result["purged"] > 0:
                logger.info(
                    f"[简单长期记忆] 定时清理完成: "
                    f"{result['purged']} 条记忆, "
                    f"{result['links_cleaned']} 条关联"
                )
            if result.get("errors"):
                raise RuntimeError(f"purge 部分失败: {result['errors']}")
        except Exception:
            logger.exception("[简单长期记忆] 定时清理失败")
            raise
        finally:
            self._running_tasks.discard("purge")

    async def _run_maintenance_cycle(self, **kwargs: Any) -> None:
        """执行完整整理周期（single-flight）。"""
        if "maintenance_cycle" in self._running_tasks:
            logger.debug("[简单长期记忆] 整理周期已在运行，跳过")
            return
        self._running_tasks.add("maintenance_cycle")
        started = time.monotonic()
        try:
            window = self._config.get("maintenance_window", "")
            try:
                window_tz = ZoneInfo(self._tz_str) if self._tz_str else None
            except ZoneInfoNotFoundError:
                window_tz = None
                logger.debug(
                    "[简单长期记忆] 整理窗口时区无效，使用系统本地时区: %r",
                    self._tz_str,
                )
            now_local = (
                datetime.now(window_tz) if window_tz else datetime.now().astimezone()
            )
            if not _is_in_maintenance_window(now_local, window):
                logger.debug(
                    "[简单长期记忆] 整理周期在时间窗口外，跳过: now=%s, window=%r",
                    now_local.isoformat(),
                    window,
                )
                return
            if not self._memory_mgr.is_kb_connected:
                logger.debug(
                    "[简单长期记忆] 整理周期跳过: kb_connected=False, window=%r",
                    window,
                )
                return

            logger.debug(
                "[简单长期记忆] 整理周期 handler 开始: kb=%s, window=%r",
                getattr(self._memory_mgr, "current_kb_name", ""),
                window,
            )
            report = await self._runner.run_cycle()
            logger.debug(
                "[简单长期记忆] 整理周期完整报告: session=%s, report=%s, "
                "duration_ms=%.1f",
                report.session_id,
                report.to_dict(),
                (time.monotonic() - started) * 1000,
            )
            logger.info(
                f"[简单长期记忆] 整理周期完成: {report.session_id}, "
                f"purge={report.purge_result.get('purged', 0)}, "
                f"LLM 调用={report.llm_stats.get('calls', 0)}, "
                f"错误={len(report.errors)}"
            )
            if report.errors:
                raise RuntimeError(f"整理周期部分失败: {report.errors}")
        except Exception:
            logger.exception("[简单长期记忆] 整理周期失败")
            raise
        finally:
            self._running_tasks.discard("maintenance_cycle")
