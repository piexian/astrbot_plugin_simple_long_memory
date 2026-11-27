"""后台整理的 AstrBot 版本门控：仅 4.30+ 允许启用。

比较用发布元组而非 packaging 的整体版本序：PEP 440 下 4.30.0-beta.2 小于
4.30.0，若按整体序比较会把当前 4.30 beta 误判为不支持。
"""

from __future__ import annotations

MINIMUM_ASTRBOT_RELEASE = (4, 30)
"""后台整理（maintenance_enabled）要求的最低 AstrBot 发布版本。"""


def parse_release_tuple(version: str) -> tuple[int, ...]:
    """把 '4.30.0-beta.2' 解析为 (4, 30, 0)，预发布后缀不参与比较。"""
    core = str(version or "").strip().split("+", 1)[0].split("-", 1)[0]
    parts: list[int] = []
    for piece in core.split("."):
        if not piece.isdigit():
            break
        parts.append(int(piece))
    return tuple(parts) if parts else (0,)


def current_astrbot_version() -> str:
    """读取运行中的 astrbot.__version__ 真实值。"""
    import astrbot

    return str(getattr(astrbot, "__version__", "") or "")


def maintenance_supported(version: str | None = None) -> bool:
    """判断给定 AstrBot 版本是否支持后台整理（缺省读当前真实版本）。"""
    release = parse_release_tuple(version or current_astrbot_version())
    return release >= MINIMUM_ASTRBOT_RELEASE


def unsupported_maintenance_reason(version: str | None = None) -> str:
    """版本不足时返回给用户的升级提示，支持时返回空串。"""
    actual = str(version or current_astrbot_version() or "未知")
    return (
        f"后台记忆整理需要 AstrBot 4.30.0 及以上，当前版本 {actual}，"
        "请升级 AstrBot 后再启用 maintenance_enabled"
    )
