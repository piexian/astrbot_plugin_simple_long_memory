"""Pytest 环境引导：在未安装 AstrBot 的独立 CI 环境中提供最小桩模块与路径设置。"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent
_PROJECT_PARENT = _PROJECT_ROOT.parent

for _path in (_PROJECT_PARENT, _PROJECT_ROOT, _TESTS_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)


def _identity_decorator(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


class _FilterStub:
    on_astrbot_loaded = staticmethod(_identity_decorator)
    on_llm_request = staticmethod(_identity_decorator)
    on_llm_response = staticmethod(_identity_decorator)
    on_agent_begin = staticmethod(_identity_decorator)
    on_agent_done = staticmethod(_identity_decorator)
    llm_tool = staticmethod(_identity_decorator)

    @staticmethod
    def command_group(*args, **kwargs):
        def decorator(func):
            func.command = _identity_decorator
            return func

        return decorator


class _SessionControllerStub:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class _SessionWaiterStub:
    def __init__(self, session_filter, session_id, record_history_chains):
        self.session_filter = session_filter
        self.session_id = session_id
        self.record_history_chains = record_history_chains
        self.session_controller = _SessionControllerStub()

    async def register_wait(self, handler, timeout=30):
        raise AssertionError("test must patch SessionWaiter before waiting")


def _install_astrbot_stubs_if_missing() -> None:
    """仅在运行环境未安装真实 astrbot 包时注入轻量桩，不覆盖真实依赖。"""
    if "astrbot" in sys.modules or importlib.util.find_spec("astrbot") is not None:
        return

    astrbot_module = types.ModuleType("astrbot")
    api_module = types.ModuleType("astrbot.api")
    event_module = types.ModuleType("astrbot.api.event")
    provider_module = types.ModuleType("astrbot.api.provider")
    star_module = types.ModuleType("astrbot.api.star")
    util_module = types.ModuleType("astrbot.api.util")

    api_module.logger = logging.getLogger("astrbot")
    event_module.AstrMessageEvent = type("AstrMessageEvent", (), {})
    event_module.filter = _FilterStub()
    provider_module.LLMResponse = type("LLMResponse", (), {})
    provider_module.ProviderRequest = type("ProviderRequest", (), {})
    star_module.Context = type("Context", (), {})
    star_module.Star = type("Star", (), {})
    util_module.SessionController = _SessionControllerStub
    util_module.SessionWaiter = _SessionWaiterStub

    sys.modules.update(
        {
            "astrbot": astrbot_module,
            "astrbot.api": api_module,
            "astrbot.api.event": event_module,
            "astrbot.api.provider": provider_module,
            "astrbot.api.star": star_module,
            "astrbot.api.util": util_module,
        }
    )


_install_astrbot_stubs_if_missing()
