"""
记忆管理器 - 封装知识库操作，通过 metadata 实现用户隔离

核心功能:
- 记忆存储 (store_memory)
- 记忆召回 (recall_memories)
- 记忆删除 (forget_memory)
- 记忆列表 (list_memories)
- 智能更新 (smart_update_memory)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from astrbot.api import logger

from .memory_protocol import (
    MemoryMetadata,
    MemoryScope,
    MemoryType,
    MemoryURI,
    MemoryVisibility,
    UMOInfo,
    build_session_id,
    build_user_id,
    format_memory_content,
    normalize_memory_scope,
)

if TYPE_CHECKING:
    from astrbot.core.knowledge_base.kb_helper import KBHelper
    from astrbot.core.knowledge_base.kb_mgr import KnowledgeBaseManager
    from astrbot.core.platform import AstrMessageEvent

# KV 存储回调类型
KVPutFn = Callable[[str, Any], Any]
KVGetFn = Callable[[str, Any], Any]
KVDeleteFn = Callable[[str], Any]

SIMILARITY_THRESHOLD = 0.85  # 相似度阈值，用于记忆合并

# 允许的记忆域
_ALLOWED_DOMAINS = frozenset(
    [
        "user_profile",
        "preferences",
        "facts",
        "events",
        "context",
        "fact",
        "preference",
        "event",  # 别名支持
    ]
)

# 域别名映射
_DOMAIN_ALIASES = {
    "fact": "facts",
    "preference": "preferences",
    "event": "events",
}

# 允许的记忆类型
_ALLOWED_MEMORY_TYPES = frozenset(
    [
        MemoryType.NORMAL,
        MemoryType.PERMANENT,
        "normal",
        "permanent",
    ]
)

# 记忆类型别名映射
_MEMORY_TYPE_ALIASES = {
    "normal": MemoryType.NORMAL,
    "permanent": MemoryType.PERMANENT,
}


def _safe_parse_metadata(metadata: Any) -> dict[str, Any]:
    """安全解析 metadata，确保返回字典"""
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def normalize_domain(domain: str) -> str:
    """标准化记忆域名称"""
    domain = (domain or "").lower().strip()
    if domain in _DOMAIN_ALIASES:
        return _DOMAIN_ALIASES[domain]
    if domain in _ALLOWED_DOMAINS:
        return domain
    return "facts"  # 默认域


def normalize_memory_type(memory_type: str) -> str:
    """标准化记忆类型"""
    memory_type = (memory_type or "").lower().strip()
    if memory_type in _MEMORY_TYPE_ALIASES:
        return _MEMORY_TYPE_ALIASES[memory_type]
    if memory_type in _ALLOWED_MEMORY_TYPES:
        return memory_type
    return MemoryType.NORMAL


def normalize_visibility(visibility: str, memory_scope: str) -> str:
    """标准化记忆可见性"""
    visibility = (visibility or "").lower().strip()
    if visibility in (MemoryVisibility.PRIVATE, MemoryVisibility.GROUP):
        return visibility
    return (
        MemoryVisibility.GROUP
        if memory_scope in (MemoryScope.GLOBAL, MemoryScope.GROUP)
        else MemoryVisibility.PRIVATE
    )


def _normalize_sender_ids(sender_ids: list[str] | None, fallback: str) -> list[str]:
    values = sender_ids or [fallback]
    result = []
    for sender_id in values:
        text = str(sender_id).strip()
        if text:
            result.append(text)
    return list(dict.fromkeys(result))


def _clamp_importance(importance: int) -> int:
    """限制重要性范围在 1-5"""
    try:
        return max(1, min(5, int(importance)))
    except (TypeError, ValueError):
        return 3


class MemoryManager:
    """记忆管理器 - 封装单知识库操作，通过 metadata 实现用户隔离"""

    def __init__(
        self,
        kb_mgr: KnowledgeBaseManager,
        config: dict,
        kv_put: KVPutFn | None = None,
        kv_get: KVGetFn | None = None,
        kv_delete: KVDeleteFn | None = None,
    ):
        self.kb_mgr = kb_mgr
        self.config = config
        self._kb_helper: KBHelper | None = None
        self._kb_name: str = ""
        self._rebuilding = False  # 重建/迁移锁
        self._pending_writes: list[dict[str, Any]] = []  # 重建期间暂存的写入
        # KV 持久化回调（由 Star 插件注入）
        self._kv_put = kv_put
        self._kv_get = kv_get
        self._kv_delete = kv_delete

    # ---------- public state accessors ----------
    @property
    def is_kb_connected(self) -> bool:
        """KB 是否已连接"""
        return self._kb_helper is not None

    @property
    def current_kb_name(self) -> str:
        """当前绑定的 KB 名称"""
        return self._kb_name

    @property
    def is_rebuilding(self) -> bool:
        """当前是否正在执行重建/迁移。"""
        return self._rebuilding

    def load_pending_writes(self, records: list[dict[str, Any]]) -> None:
        """从外部恢复重建期间未落盘的写入缓冲（启动恢复用）"""
        self._pending_writes = list(records)

    def initialize(self) -> None:
        """初始化记忆管理器（仅校验配置，不连接 KB）

        Raises:
            ValueError: 知识库未配置
        """
        kb_name_raw = self.config.get("kb_name", [])
        kb_name = (
            kb_name_raw[0]
            if isinstance(kb_name_raw, list) and kb_name_raw
            else kb_name_raw
        )
        if not kb_name:
            raise ValueError("记忆知识库未配置，请在插件设置中选择一个知识库")

        self._kb_name = kb_name

    async def connect_kb(self) -> None:
        """连接知识库（需在 KB 模块就绪后调用）

        Raises:
            ValueError: 知识库不存在
        """
        kb = await self.kb_mgr.get_kb_by_name(self._kb_name)
        if not kb:
            raise ValueError(f"知识库 '{self._kb_name}' 不存在，请先在知识库管理中创建")

        self._kb_helper = kb
        logger.info(f"[简单长期记忆] 已连接知识库: {self._kb_name}")
        await self._migrate_patch_chunk_index()

    async def _migrate_patch_chunk_index(self) -> None:
        """迁移补丁：为缺少 chunk_index 字段的旧记忆条目写入默认值 0。

        旧版插件直接写入 vec_db 时未设置 chunk_index，导致 AstrBot 知识库检索
        界面调用稀疏检索时抛出 KeyError: 'chunk_index'。
        通过 SQLite json_set 原地修改 metadata，无需重新嵌入向量。
        覆盖范围：有 is_memory_record 标记的新版记录 + 有 uri 但无标记的更早记录。
        """
        try:
            doc_storage = self.vec_db.document_storage
            async with doc_storage.get_session() as session, session.begin():
                from sqlalchemy import text as sa_text

                result = await session.execute(
                    sa_text(
                        "UPDATE documents "
                        "SET metadata = json_set(metadata, '$.chunk_index', 0) "
                        "WHERE json_extract(metadata, '$.chunk_index') IS NULL "
                        "  AND (json_extract(metadata, '$.is_memory_record') = 1 "
                        "       OR json_extract(metadata, '$.uri') IS NOT NULL)"
                    )
                )
                patched = result.rowcount
                if patched:
                    logger.info(
                        f"[简单长期记忆] 迁移补丁：已为 {patched} 条旧记忆补写 chunk_index=0"
                    )
        except Exception as e:
            logger.warning(f"[简单长期记忆] 迁移补丁执行失败（不影响功能）: {e}")

    @property
    def vec_db(self):
        """获取向量数据库实例"""
        if not self._kb_helper:
            raise RuntimeError("记忆管理器未初始化")
        return self._kb_helper.vec_db

    async def _exec_metadata_update(
        self,
        set_clause: str,
        where_clause: str,
        params: dict[str, Any],
    ) -> int:
        """原地更新 documents.metadata（json_set），返回受影响行数。

        绕过 FaissVecDB 无 update_metadata API 的限制：只改 metadata JSON 列，
        不动 FAISS 向量、不动 FTS5 索引，廉价且安全。失败仅记录日志，不阻断检索。
        """
        if not self._kb_helper:
            return 0
        try:
            doc_storage = self.vec_db.document_storage
            async with doc_storage.get_session() as session, session.begin():
                from sqlalchemy import text as sa_text

                result = await session.execute(
                    sa_text(
                        f"UPDATE documents SET metadata = {set_clause} WHERE {where_clause}"
                    ),
                    params,
                )
                return int(result.rowcount or 0)
        except Exception as e:
            logger.warning(f"[简单长期记忆] metadata 原地更新失败（不影响检索）: {e}")
            return 0

    async def _bump_recall_stats(self, uris: list[str]) -> None:
        """递增给定 uri 记忆的 recall_count 并刷新 last_recalled_at（P0.1 召回反馈）。"""
        uris = [u for u in uris if u]
        if not uris:
            return
        now = datetime.now(timezone.utc).isoformat()
        set_clause = (
            "json_set(metadata, '$.recall_count', "
            "CAST(COALESCE(json_extract(metadata,'$.recall_count'),0) AS INTEGER) + 1, "
            "'$.last_recalled_at', :now)"
        )
        placeholders = ",".join(f":u{i}" for i in range(len(uris)))
        where_clause = (
            f"json_extract(metadata,'$.uri') IN ({placeholders}) "
            "AND json_extract(metadata,'$.is_memory_record') = 1"
        )
        params: dict[str, Any] = {"now": now}
        params.update({f"u{i}": u for i, u in enumerate(uris)})
        await self._exec_metadata_update(set_clause, where_clause, params)

    async def expire_stale_memories(self, ttl_days: int) -> int:
        """TTL 过期：把超过 ttl_days 且未废弃的记忆标记 deprecated=True（P1.1）。

        召回 filter 已排除 deprecated，标记后即从召回移除。返回标记条数。
        """
        if ttl_days <= 0 or not self._kb_helper:
            return 0
        kb_id = self._kb_helper.kb.kb_id
        cutoff_iso = (datetime.now(timezone.utc) - timedelta(days=ttl_days)).isoformat()
        set_clause = "json_set(metadata, '$.deprecated', 1)"
        where_clause = (
            "json_extract(metadata,'$.is_memory_record') = 1 "
            "AND json_extract(metadata,'$.deprecated') IS NOT 1 "
            "AND json_extract(metadata,'$.kb_id') = :kb_id "
            "AND json_extract(metadata,'$.created_at') < :cutoff "
            "AND json_extract(metadata,'$.memory_type') != 'permanent' "
            "AND json_extract(metadata,'$.memory_scope') != 'global'"
        )
        params: dict[str, Any] = {"kb_id": kb_id, "cutoff": cutoff_iso}
        return await self._exec_metadata_update(set_clause, where_clause, params)

    async def fetch_consolidation_candidates(
        self,
        event: AstrMessageEvent,
        min_age_days: int,
        max_recall: int = 1,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """取出当前用户低频老旧、未废弃未压缩的个人记忆，作为巩固候选（P1.2）。

        owner 与 memory_scope 由 event 推导并锁定为当前用户 personal，防止跨用户/跨作用域泄露。
        """
        if not self._kb_helper or limit <= 0:
            return []
        owner_user_id = self._current_owner_user_id(event)
        memory_scope = MemoryScope.PERSONAL
        kb_id = self._kb_helper.kb.kb_id
        cutoff_iso = (
            datetime.now(timezone.utc) - timedelta(days=min_age_days)
        ).isoformat()
        try:
            doc_storage = self.vec_db.document_storage
            async with doc_storage.get_session() as session:
                from sqlalchemy import text as sa_text

                rows = (
                    await session.execute(
                        sa_text(
                            "SELECT text, metadata FROM documents "
                            "WHERE json_extract(metadata,'$.is_memory_record') = 1 "
                            "AND json_extract(metadata,'$.deprecated') IS NOT 1 "
                            "AND json_extract(metadata,'$.compressed') IS NOT 1 "
                            "AND json_extract(metadata,'$.kb_id') = :kb_id "
                            "AND json_extract(metadata,'$.created_at') < :cutoff "
                            "AND CAST(COALESCE(json_extract(metadata,'$.recall_count'),0) AS INTEGER) <= :max_recall "
                            "AND json_extract(metadata,'$.owner_user_id') = :owner_user_id "
                            "AND json_extract(metadata,'$.memory_scope') = :memory_scope "
                            "ORDER BY json_extract(metadata,'$.created_at') ASC "
                            "LIMIT :limit"
                        ),
                        {
                            "kb_id": kb_id,
                            "cutoff": cutoff_iso,
                            "max_recall": max_recall,
                            "owner_user_id": owner_user_id,
                            "memory_scope": memory_scope,
                            "limit": limit,
                        },
                    )
                ).all()
        except Exception as e:
            logger.warning(f"[简单长期记忆] 读取巩固候选失败: {e}")
            return []
        candidates: list[dict[str, Any]] = []
        for row in rows:
            text_val = getattr(row, "text", "") or ""
            meta = _safe_parse_metadata(getattr(row, "metadata", {}) or {})
            candidates.append({"text": text_val, "metadata": meta})
        return candidates

    async def mark_consolidated(self, uris: list[str]) -> int:
        """把已巩固的原记忆标记为 deprecated+compressed（P1.2）。"""
        uris = [u for u in uris if u]
        if not uris:
            return 0
        set_clause = "json_set(metadata, '$.deprecated', 1, '$.compressed', 1)"
        placeholders = ",".join(f":u{i}" for i in range(len(uris)))
        where_clause = (
            f"json_extract(metadata,'$.uri') IN ({placeholders}) "
            "AND json_extract(metadata,'$.is_memory_record') = 1"
        )
        params: dict[str, Any] = {f"u{i}": u for i, u in enumerate(uris)}
        return await self._exec_metadata_update(set_clause, where_clause, params)

    def _rerank_by_signal(self, memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """根据 importance/recall_count/recency 对召回结果二次加权排序（P0.2）。

        纯本地计算，不依赖 AstrBot。权重可经配置调整。
        """
        if len(memories) <= 1:
            return memories
        now = time.time()
        half_life = max(self.config.get("recall_recency_halflife_days", 14), 1) * 86400
        w_importance = self.config.get("recall_weight_importance", 0.4)
        w_frequency = self.config.get("recall_weight_frequency", 0.3)
        w_recency = self.config.get("recall_weight_recency", 0.3)

        def _score(mem: dict[str, Any]) -> float:
            meta = mem.get("metadata", {})
            try:
                _imp = meta.get("importance")
                _imp = 3 if _imp is None or _imp == "" else _imp
                importance = (int(float(_imp)) - 1) / 4.0
            except (ValueError, TypeError):
                importance = 0.5
            try:
                _rc = meta.get("recall_count")
                _rc = 0 if _rc is None or _rc == "" else _rc
                frequency = min(math.log1p(max(0, int(float(_rc)))) / 3.0, 1.0)
            except (ValueError, TypeError):
                frequency = 0.0
            ts_str = meta.get("last_recalled_at") or meta.get("created_at")
            recency = 0.5
            if ts_str:
                try:
                    ts = datetime.fromisoformat(
                        str(ts_str).replace("Z", "+00:00")
                    ).timestamp()
                    recency = math.exp(-max(0.0, now - ts) / half_life)
                except (ValueError, TypeError):
                    recency = 0.5
            return (
                w_importance * importance
                + w_frequency * frequency
                + w_recency * recency
            )

        return sorted(memories, key=_score, reverse=True)

    # ==================== KB 文档注册 ====================

    async def _register_kb_document(
        self,
        doc_id: str,
        doc_name: str,
        content_size: int,
        kb_helper: KBHelper | None = None,
    ) -> None:
        """将记忆注册为 KB 文档，使其在知识库界面可见"""
        from astrbot.core.knowledge_base.models import KBDocument

        kb = kb_helper or self._kb_helper
        doc = KBDocument(
            doc_id=doc_id,
            kb_id=kb.kb.kb_id,
            doc_name=doc_name,
            file_type="memory",
            file_size=content_size,
            file_path="",
            chunk_count=1,
            media_count=0,
        )
        async with kb.kb_db.get_db() as session:
            async with session.begin():
                session.add(doc)
                await session.commit()

    async def _ensure_kb_document(
        self,
        doc_id: str,
        doc_name: str,
        content_size: int,
        kb_helper: KBHelper | None = None,
    ) -> bool:
        """确保向量文档对应的 KB 文档记录存在。"""
        if not doc_id:
            return False

        kb = kb_helper or self._kb_helper
        if not kb:
            return False

        try:
            existing = await kb.get_document(doc_id)
            if existing:
                return True
            await self._register_kb_document(
                doc_id,
                doc_name,
                content_size,
                kb_helper=kb,
            )
            return True
        except Exception as e:
            logger.warning(f"[简单长期记忆] KB 文档记录修复失败: {doc_id}, {e}")
            return False

    async def _find_memory_doc_by_uri(
        self,
        kb_helper: KBHelper,
        uri: str,
    ) -> dict[str, Any] | None:
        """按 URI 精确查找目标 KB 中已存在的记忆向量文档。"""
        if not uri:
            return None
        docs = await kb_helper.vec_db.document_storage.get_documents(
            metadata_filters={
                "uri": uri,
                "is_memory_record": True,
                "kb_id": kb_helper.kb.kb_id,
            },
            limit=1,
        )
        return docs[0] if docs else None

    async def _repair_kb_document_for_vector_doc(
        self,
        kb_helper: KBHelper,
        doc: dict[str, Any],
        fallback_name: str,
    ) -> bool:
        """向量文档已存在时，补齐缺失的 KB 文档记录。"""
        metadata = _safe_parse_metadata(doc.get("metadata", {}))
        doc_id = metadata.get("kb_doc_id")
        if not doc_id:
            return False
        doc_name = metadata.get("uri") or fallback_name or doc_id
        content_size = len(doc.get("text", "") or "")
        return await self._ensure_kb_document(
            doc_id,
            doc_name,
            content_size,
            kb_helper=kb_helper,
        )

    async def _unregister_kb_documents(
        self,
        doc_ids: list[str],
        kb_helper: KBHelper | None = None,
    ) -> None:
        """批量移除 KB 文档记录"""
        if not doc_ids:
            return
        from astrbot.core.knowledge_base.models import KBDocument
        from sqlmodel import col, delete

        kb = kb_helper or self._kb_helper
        async with kb.kb_db.get_db() as session:
            async with session.begin():
                stmt = delete(KBDocument).where(col(KBDocument.doc_id).in_(doc_ids))
                await session.execute(stmt)
                await session.commit()

    async def _sync_kb_stats(self, kb_helper: KBHelper | None = None) -> None:
        """同步知识库统计数据"""
        kb = kb_helper or self._kb_helper
        await kb.kb_db.update_kb_stats(
            kb_id=kb.kb.kb_id,
            vec_db=kb.vec_db,
        )
        await kb.refresh_kb()

    async def _delete_rebuild_source_records(
        self,
        kb_helper: KBHelper,
        memory_records: list[dict[str, Any]],
    ) -> None:
        kb_id = kb_helper.kb.kb_id
        await kb_helper.vec_db.delete_documents(
            metadata_filters={"is_memory_record": True, "kb_id": kb_id}
        )
        legacy_uris: set[str] = set()
        for record in memory_records:
            metadata = _safe_parse_metadata(record.get("metadata", {}))
            if (
                not metadata.get("is_memory_record")
                and metadata.get("kb_id") == kb_id
                and metadata.get("uri")
            ):
                legacy_uris.add(metadata["uri"])
        for uri in legacy_uris:
            await kb_helper.vec_db.delete_documents(
                metadata_filters={"uri": uri, "kb_id": kb_id}
            )

    def _build_user_filter(self, event: AstrMessageEvent) -> dict[str, Any]:
        """构建用户隔离的 metadata 过滤器

        Args:
            event: 消息事件

        Returns:
            metadata 过滤器字典
        """
        return {
            "user_id": build_user_id(event.get_platform_id(), event.get_sender_id()),
        }

    def _event_scope_ids(
        self, event: AstrMessageEvent, owner_sender_id: str | None = None
    ) -> tuple[UMOInfo, str, str]:
        parsed = UMOInfo.parse(event.unified_msg_origin)
        sender_id = owner_sender_id or event.get_sender_id()
        owner_user_id = build_user_id(parsed.platform_id, sender_id)
        owner_session_id = build_session_id(parsed.platform_id, parsed.session_id)
        return parsed, owner_user_id, owner_session_id

    def _build_owner_user_ids(
        self, platform_id: str, owner_sender_ids: list[str]
    ) -> list[str]:
        return [build_user_id(platform_id, sender_id) for sender_id in owner_sender_ids]

    def _current_owner_user_id(self, event: AstrMessageEvent) -> str:
        parsed = UMOInfo.parse(event.unified_msg_origin)
        return build_user_id(parsed.platform_id, event.get_sender_id())

    def _is_visible_shared_personal(
        self, event: AstrMessageEvent, metadata: dict[str, Any]
    ) -> bool:
        """多 owner personal 记忆只对 owner_user_ids 内的用户可见。"""
        if metadata.get("memory_scope") != MemoryScope.PERSONAL:
            return True
        if metadata.get("visibility") != MemoryVisibility.GROUP:
            return True
        owner_user_ids = metadata.get("owner_user_ids", [])
        if not isinstance(owner_user_ids, list):
            owner_user_ids = []
        return self._current_owner_user_id(event) in owner_user_ids

    def _filter_visible_shared_personal(
        self, event: AstrMessageEvent, memories: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        visible = []
        for memory in memories:
            metadata = _safe_parse_metadata(memory.get("metadata", {}))
            if self._is_visible_shared_personal(event, metadata):
                visible.append(memory)
        return visible

    def _scope_filter(
        self,
        event: AstrMessageEvent,
        memory_scope: str,
        global_memory: bool = True,
    ) -> dict[str, Any]:
        _, owner_user_id, owner_session_id = self._event_scope_ids(event)
        scope = normalize_memory_scope(memory_scope)

        if scope == MemoryScope.GLOBAL:
            filters = {
                "memory_scope": MemoryScope.GLOBAL,
            }
        elif scope == MemoryScope.GROUP:
            filters = {
                "memory_scope": MemoryScope.GROUP,
                "owner_session_id": owner_session_id,
            }
        elif scope == MemoryScope.CONVERSATION:
            filters = {
                "memory_scope": MemoryScope.CONVERSATION,
                "umo": event.unified_msg_origin,
            }
        else:
            filters = {
                "memory_scope": MemoryScope.PERSONAL,
                "owner_user_id": owner_user_id,
            }
            if not global_memory:
                filters["umo"] = event.unified_msg_origin

        filters["is_memory_record"] = True
        filters["deprecated"] = False
        return filters

    def _build_query_filter(
        self,
        event: AstrMessageEvent | None,
        *,
        all_users: bool,
        domain: str | None = None,
        include_deprecated: bool = False,
        respect_global: bool = False,
    ) -> dict[str, Any]:
        """统一构建查询/列表/清空使用的 metadata 过滤器。

        Args:
            event: 消息事件（all_users 为 True 时可为 None）
            all_users: True 时跳过用户隔离，使用 is_memory_record 标记
            domain: 可选记忆域过滤
            include_deprecated: 为 False 时排除 deprecated=True 的记忆
            respect_global: True 时按 self.config['global_memory'] 决定是否限定 umo
        """
        if all_users:
            filters: dict[str, Any] = {"is_memory_record": True}
        else:
            if event is None:
                raise ValueError("非 all_users 模式需要传入 event")
            if respect_global:
                global_memory = self.config.get("global_memory", True)
                filters = self._scope_filter(event, MemoryScope.PERSONAL, global_memory)
            else:
                filters = self._build_user_filter(event)
        if not include_deprecated:
            filters["deprecated"] = False
        if domain:
            filters["domain"] = domain
        return filters

    def _build_memory_metadata(
        self,
        event: AstrMessageEvent,
        **extra: Any,
    ) -> dict[str, Any]:
        """构建完整的记忆元数据

        Args:
            event: 消息事件
            **extra: 额外的元数据字段

        Returns:
            完整的元数据字典
        """
        umo = event.unified_msg_origin
        memory_scope = normalize_memory_scope(extra.pop("memory_scope", ""))
        visibility = normalize_visibility(extra.pop("visibility", ""), memory_scope)
        speaker_id = extra.pop("speaker_id", event.get_sender_id())
        owner_sender_id = extra.pop("owner_sender_id", None)
        owner_sender_ids = _normalize_sender_ids(
            extra.pop("owner_sender_ids", None),
            owner_sender_id or event.get_sender_id(),
        )
        parsed, owner_user_id, owner_session_id = self._event_scope_ids(
            event, owner_sender_ids[0]
        )
        owner_user_ids = self._build_owner_user_ids(
            parsed.platform_id, owner_sender_ids
        )

        return {
            "user_id": owner_user_id,
            "platform_id": parsed.platform_id,
            "sender_id": owner_sender_ids[0],
            "umo": umo,
            "session_type": parsed.session_type,
            "session_id": parsed.session_id,
            "memory_scope": memory_scope,
            "owner_user_id": owner_user_id,
            "owner_user_ids": owner_user_ids,
            "owner_session_id": owner_session_id,
            "visibility": visibility,
            "speaker_id": speaker_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_recalled_at": datetime.now(timezone.utc).isoformat(),
            "recall_count": 0,
            "compressed": False,
            **extra,
        }

    async def store_memory(
        self,
        event: AstrMessageEvent,
        content: str,
        domain: str,
        uri: str | None = None,
        memory_type: str = MemoryType.NORMAL,
        disclosure: str = "",
        importance: int = 3,
        memory_scope: str = MemoryScope.PERSONAL,
        visibility: str = "",
        subject: str = "",
        entities: list[str] | None = None,
        topics: list[str] | None = None,
        owner_sender_id: str | None = None,
        owner_sender_ids: list[str] | None = None,
    ) -> str:
        """存储记忆到知识库

        Args:
            event: 消息事件
            content: 记忆内容
            domain: 记忆域
            uri: 记忆 URI（可选，自动生成）
            memory_type: 记忆类型
            disclosure: 触发召回条件描述
            importance: 重要性等级 (1-5)

        Returns:
            存储的记忆 ID
        """
        # 标准化参数
        domain = normalize_domain(domain)
        memory_type = normalize_memory_type(memory_type)
        importance = _clamp_importance(importance)
        memory_scope = normalize_memory_scope(memory_scope)
        visibility = normalize_visibility(visibility, memory_scope)
        entities = entities or []
        topics = topics or []
        owner_sender_ids = _normalize_sender_ids(
            owner_sender_ids, owner_sender_id or event.get_sender_id()
        )
        if memory_scope == MemoryScope.PERSONAL and len(owner_sender_ids) > 1:
            visibility = MemoryVisibility.GROUP

        if uri is None:
            uri = str(MemoryURI.generate(domain))

        # 重建/迁移期间：暂存到本地缓冲区并持久化到 KV，完成后批量处理
        if self._rebuilding:
            umo = event.unified_msg_origin
            parsed, owner_user_id, owner_session_id = self._event_scope_ids(
                event, owner_sender_ids[0]
            )
            owner_user_ids = self._build_owner_user_ids(
                parsed.platform_id, owner_sender_ids
            )
            item = {
                "content": content,
                "domain": domain,
                "uri": uri,
                "memory_type": memory_type,
                "disclosure": disclosure,
                "importance": importance,
                "user_id": owner_user_id,
                "owner_user_ids": owner_user_ids,
                "platform_id": parsed.platform_id,
                "sender_id": owner_sender_ids[0],
                "umo": umo,
                "session_type": parsed.session_type,
                "session_id": parsed.session_id,
                "memory_scope": memory_scope,
                "owner_user_id": owner_user_id,
                "owner_session_id": owner_session_id,
                "visibility": visibility,
                "speaker_id": owner_sender_ids[0],
                "subject": subject,
                "entities": entities,
                "topics": topics,
            }
            self._pending_writes.append(item)
            # 持久化缓冲区到 KV，防进程重启丢失
            if self._kv_put:
                await self._kv_put("rebuild_pending_writes", self._pending_writes)
            logger.debug(f"[简单长期记忆] 重建进行中，已缓冲记忆: {uri}")
            return uri

        # URI 去重：同名 URI 已存在时，内容相同则跳过，内容不同则换新 URI
        existing = await self.vec_db.document_storage.get_documents(
            metadata_filters={"uri": uri}, limit=1
        )
        if existing:
            old_text = existing[0].get("text", "")
            if old_text.strip() == content.strip():
                logger.debug(f"[简单长期记忆] 内容重复，跳过写入: {uri}")
                return uri
            uri = str(MemoryURI.generate(domain))
            logger.debug(f"[简单长期记忆] URI 冲突且内容不同，已重新生成: {uri}")

        metadata = self._build_memory_metadata(
            event,
            domain=domain,
            uri=uri,
            version=1,
            deprecated=False,
            memory_type=memory_type,
            disclosure=disclosure,
            importance=importance,
            memory_scope=memory_scope,
            visibility=visibility,
            subject=subject,
            entities=entities,
            topics=topics,
            memory_content=content,
            owner_sender_ids=owner_sender_ids,
        )

        # 生成 KB 文档 ID 并关联到向量条目
        doc_id = str(uuid.uuid4())
        metadata["kb_doc_id"] = doc_id
        metadata["kb_id"] = self._kb_helper.kb.kb_id
        metadata["chunk_index"] = 0
        metadata["is_memory_record"] = True

        # 格式化内容
        formatted_content = format_memory_content(content, metadata)

        # 存储到向量数据库
        await self.vec_db.insert(
            content=formatted_content,
            metadata=metadata,
            id=doc_id,
        )

        # 注册为 KB 文档（界面可见）
        try:
            await self._register_kb_document(doc_id, uri, len(formatted_content))
            await self._sync_kb_stats()
        except Exception as e:
            logger.warning(f"[简单长期记忆] KB 文档注册失败（不影响记忆功能）: {e}")

        logger.debug(f"[简单长期记忆] 存储记忆: {uri}, 用户: {metadata['user_id']}")
        return uri

    async def recall_memories(
        self,
        event: AstrMessageEvent,
        query: str,
        domain: str | None = None,
        top_k: int | None = None,
        all_users: bool = False,
        memory_scope: str | None = None,
        bump: bool = False,
    ) -> list[dict[str, Any]]:
        """召回相关记忆（自动按用户隔离）

        Args:
            event: 消息事件
            query: 查询文本
            domain: 记忆域过滤（可选）
            top_k: 返回数量（可选，使用配置值）
            all_users: 为 True 时跳过用户过滤

        Returns:
            记忆列表，每项包含 'text' 和 'metadata'
        """
        if top_k is None:
            top_k = self.config.get("max_memories_per_inject", 5)
        fetch_k = max(top_k, min(top_k * 3, 20))

        if all_users:
            filters = {"is_memory_record": True, "deprecated": False}
            if domain:
                filters["domain"] = domain
            raw = await self._retrieve_with_filter(query, fetch_k, filters)
            memories = self._dedupe_memories(raw)[:top_k]
        else:
            global_memory = self.config.get("global_memory", True)
            filters_list = self._build_recall_filters(
                event,
                global_memory=global_memory,
                domain=domain,
                memory_scope=memory_scope,
            )

            tasks = [
                self._retrieve_with_filter(query, fetch_k, filters)
                for filters in filters_list
            ]
            results_list = await asyncio.gather(*tasks)
            results = [item for sublist in results_list for item in sublist]
            results = self._filter_visible_shared_personal(event, results)
            memories = self._dedupe_memories(results)[:top_k]

        # P0.2 信号加权重排（importance / recall_count / recency）
        memories = self._rerank_by_signal(memories)
        # P0.1 召回反馈：仅注入路径递增 recall_count（避免 search/selftest/工具调用污染频次信号）
        if bump:
            recalled_uris = [
                m.get("metadata", {}).get("uri")
                for m in memories
                if m.get("metadata", {}).get("uri")
            ]
            await self._bump_recall_stats(recalled_uris)
        logger.debug(f"[简单长期记忆] 召回 {len(memories)} 条记忆")
        return memories

    def _build_recall_filters(
        self,
        event: AstrMessageEvent,
        global_memory: bool,
        domain: str | None = None,
        memory_scope: str | None = None,
    ) -> list[dict[str, Any]]:
        parsed = UMOInfo.parse(event.unified_msg_origin)
        scopes = (
            [normalize_memory_scope(memory_scope)]
            if memory_scope
            else [MemoryScope.GLOBAL, MemoryScope.PERSONAL]
        )
        if not memory_scope:
            if parsed.session_type == "group":
                scopes.extend([MemoryScope.GROUP, MemoryScope.CONVERSATION])
            else:
                scopes.append(MemoryScope.CONVERSATION)

        filters_list = []
        for scope in scopes:
            filters = self._scope_filter(event, scope, global_memory)
            if domain:
                filters["domain"] = domain
            filters_list.append(filters)
            if scope == MemoryScope.PERSONAL:
                if parsed.session_type == "group":
                    group_personal_filters = {
                        "memory_scope": MemoryScope.PERSONAL,
                        "owner_session_id": build_session_id(
                            parsed.platform_id, parsed.session_id
                        ),
                        "visibility": MemoryVisibility.GROUP,
                        "is_memory_record": True,
                        "deprecated": False,
                    }
                    if domain:
                        group_personal_filters["domain"] = domain
                    filters_list.append(group_personal_filters)
        return filters_list

    async def _retrieve_with_filter(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        use_rerank = self.config.get("use_reranker", True)
        # 稠密检索（本阶段不 rerank，融合后再统一 rerank）
        dense_results = await self.vec_db.retrieve(
            query=query,
            k=top_k,
            rerank=False,
            metadata_filters=filters,
        )
        dense_memories: list[dict[str, Any]] = []
        for result in dense_results:
            data = result.data
            dense_memories.append(
                {
                    "text": data.get("text", ""),
                    "metadata": _safe_parse_metadata(data.get("metadata", {})),
                    "similarity": result.similarity,
                }
            )

        # P0.3 稀疏检索（FTS5）+ RRF 融合
        if self.config.get("recall_sparse_fusion", True):
            sparse_memories = await self._sparse_retrieve(query, top_k, filters)
            memories = (
                self._rrf_fuse(dense_memories, sparse_memories, limit=top_k)
                if sparse_memories
                else dense_memories
            )
        else:
            memories = dense_memories

        # 融合后统一 rerank（复用知识库配置的 rerank provider）
        if use_rerank and memories:
            rerank_provider = getattr(self.vec_db, "rerank_provider", None)
            if rerank_provider:
                try:
                    docs = [m.get("text", "") for m in memories]
                    reranked = await rerank_provider.rerank(query, docs)
                    reranked = sorted(
                        reranked, key=lambda x: x.relevance_score, reverse=True
                    )
                    memories = [
                        memories[r.index]
                        for r in reranked
                        if 0 <= r.index < len(memories)
                    ]
                except Exception as e:
                    logger.debug(f"[简单长期记忆] rerank 失败，使用融合结果: {e}")
        return memories

    async def _sparse_retrieve(
        self, query: str, top_k: int, filters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """FTS5 稀疏检索（P0.3）。不可用或失败时返回空列表，调用方回退纯稠密。"""
        tokens = self._tokenize_query(query)
        if not tokens:
            return []
        try:
            docs = await self.vec_db.document_storage.search_sparse(
                query_tokens=tokens, limit=max(top_k * 3, 20)
            )
        except Exception as e:
            logger.debug(f"[简单长期记忆] 稀疏检索失败，回退纯稠密: {e}")
            return []
        if not docs:
            return []
        out: list[dict[str, Any]] = []
        for doc in docs:
            meta = _safe_parse_metadata(doc.get("metadata", {}))
            if not self._matches_filters(meta, filters):
                continue
            out.append(
                {
                    "text": doc.get("text", ""),
                    "metadata": meta,
                    "similarity": -float(doc.get("score", 0) or 0),
                }
            )
        return out

    def _tokenize_query(self, query: str) -> list[str]:
        """分词（依赖缓存）。AstrBot 检索依赖不可用时返回空列表。"""
        cache = getattr(self, "_sparse_tokenize", None)
        if cache is None:
            try:
                import os

                from astrbot.core.knowledge_base.retrieval import (
                    sparse_retriever as _sr,
                )
                from astrbot.core.knowledge_base.retrieval.tokenizer import (
                    load_stopwords,
                    tokenize_text,
                )

                stopwords = load_stopwords(
                    os.path.join(os.path.dirname(_sr.__file__), "hit_stopwords.txt")
                )
                self._sparse_tokenize = (tokenize_text, stopwords)
                cache = self._sparse_tokenize
            except Exception as e:
                logger.debug(f"[简单长期记忆] 稀疏检索分词依赖不可用: {e}")
                self._sparse_tokenize = False
                return []
        if cache is False:
            return []
        tokenize_text, stopwords = cache
        return tokenize_text(query, stopwords)

    @staticmethod
    def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        """本地 metadata 过滤（稀疏检索结果不带 filter，需本地过滤）。"""
        for key, expected in filters.items():
            if key == "is_memory_record":
                if bool(metadata.get(key)) != bool(expected):
                    return False
            elif key == "deprecated":
                if bool(metadata.get(key, False)) != bool(expected):
                    return False
            elif metadata.get(key) != expected:
                return False
        return True

    def _rrf_fuse(
        self,
        dense: list[dict[str, Any]],
        sparse: list[dict[str, Any]],
        k: int = 60,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion 融合稠密与稀疏结果（P0.3）。

        score(doc) = sum(1 / (k + rank))，同 AstrBot rank_fusion.py 公式；uri 作去重主键。
        """
        scores: dict[str, float] = {}
        dense_by_uri: dict[str, dict[str, Any]] = {}
        sparse_by_uri: dict[str, dict[str, Any]] = {}

        for rank, mem in enumerate(dense):
            uri = mem.get("metadata", {}).get("uri") or mem.get("text", "")
            if not uri:
                continue
            scores[uri] = scores.get(uri, 0.0) + 1.0 / (k + rank + 1)
            dense_by_uri.setdefault(uri, mem)
        for rank, mem in enumerate(sparse):
            uri = mem.get("metadata", {}).get("uri") or mem.get("text", "")
            if not uri:
                continue
            scores[uri] = scores.get(uri, 0.0) + 1.0 / (k + rank + 1)
            sparse_by_uri.setdefault(uri, mem)

        ordered = sorted(scores, key=lambda u: scores[u], reverse=True)[:limit]
        fused: list[dict[str, Any]] = []
        for uri in ordered:
            mem = dense_by_uri.get(uri) or sparse_by_uri.get(uri)
            if mem:
                # 保留原始 similarity（dense cosine / sparse 负分），RRF 分独立存放；
                # 勿覆盖 similarity，否则 smart_update_memory 的 0.85 阈值与 _dedupe 排序失效
                fused.append({**mem, "rrf_score": scores[uri]})
        return fused

    def _dedupe_memories(self, memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen = set()
        deduped = []
        for mem in sorted(memories, key=lambda m: m.get("similarity", 0), reverse=True):
            metadata = mem.get("metadata", {})
            key = metadata.get("uri") or mem.get("text", "")
            if key in seen:
                continue
            seen.add(key)
            deduped.append(mem)
        return deduped

    async def forget_memory(self, event: AstrMessageEvent, uri: str) -> int:
        """删除当前用户指定 URI 的记忆（LLM 工具用，按 user_id 隔离）

        Args:
            event: 消息事件
            uri: 记忆 URI

        Returns:
            删除数。无权限和不存在都返回 0，避免泄漏跨用户 URI 存在性。
        """
        filters = self._build_user_filter(event)
        filters["uri"] = uri
        deleted = await self._delete_by_filters(filters, uri)
        if deleted > 0:
            return deleted

        shared_docs = await self.vec_db.document_storage.get_documents(
            metadata_filters={
                "uri": uri,
                "memory_scope": MemoryScope.PERSONAL,
                "visibility": MemoryVisibility.GROUP,
                "is_memory_record": True,
                "deprecated": False,
            },
            limit=20,
        )
        for doc in shared_docs:
            metadata = _safe_parse_metadata(doc.get("metadata", {}))
            if self._is_visible_shared_personal(event, metadata):
                return await self._delete_by_filters(
                    {"uri": uri, "kb_doc_id": metadata.get("kb_doc_id", "")},
                    uri,
                )
        return 0

    async def forget_memory_by_uri(self, uri: str) -> int:
        """管理员按 URI 删除所有匹配的记忆（不限用户）

        Args:
            uri: 记忆 URI

        Returns:
            实际删除的记录数
        """
        return await self._delete_by_filters({"uri": uri}, uri)

    async def _delete_by_filters(self, filters: dict[str, Any], uri: str) -> int:
        """按 filters 删除记忆并同步清理 KB 文档记录

        Args:
            filters: metadata 过滤条件
            uri: 用于日志的记忆 URI

        Returns:
            实际删除的记录数
        """
        doc_ids: list[str] = []
        deleted = 0
        try:
            doc_ids, deleted = await self._collect_kb_doc_ids_for_filters(filters)
        except Exception as e:
            logger.warning(f"[简单长期记忆] 查询待删除文档失败: {e}")
            try:
                deleted = await self.vec_db.count_documents(metadata_filter=filters)
            except Exception as ce:
                logger.warning(f"[简单长期记忆] 统计待删除文档失败: {ce}")

        await self.vec_db.delete_documents(metadata_filters=filters)

        # 同步删除 KB 文档记录
        try:
            await self._unregister_kb_documents(doc_ids)
            await self._sync_kb_stats()
        except Exception as e:
            logger.warning(f"[简单长期记忆] KB 文档删除失败: {e}")

        logger.info(f"[简单长期记忆] 删除记忆: {uri}, 实际删除 {deleted} 条")
        return deleted

    async def _collect_kb_doc_ids_for_filters(
        self,
        filters: dict[str, Any],
    ) -> tuple[list[str], int]:
        """分页收集匹配向量文档的 KB 文档 ID，避免固定上限截断。"""
        try:
            page_size = int(self.config.get("memory_delete_scan_page_size", 1000))
        except (TypeError, ValueError):
            page_size = 1000
        page_size = max(1, page_size)

        doc_ids: list[str] = []
        count = 0
        offset = 0
        while True:
            docs = await self.vec_db.document_storage.get_documents(
                metadata_filters=filters,
                offset=offset,
                limit=page_size,
            )
            if not docs:
                break
            count += len(docs)
            offset += len(docs)
            for doc in docs:
                md = _safe_parse_metadata(doc.get("metadata", {}))
                if md.get("kb_doc_id"):
                    doc_ids.append(md["kb_doc_id"])
            if len(docs) < page_size:
                break

        return list(dict.fromkeys(doc_ids)), count

    async def clear_memories(
        self,
        event: AstrMessageEvent,
        domain: str | None = None,
        all_users: bool = False,
    ) -> int:
        """清空记忆

        Args:
            event: 消息事件
            domain: 仅清空指定域的记忆（可选）
            all_users: 为 True 时清空所有用户记忆

        Returns:
            删除的记忆数量
        """
        filters = self._build_query_filter(
            event,
            all_users=all_users,
            domain=domain,
            include_deprecated=True,
        )
        scope = "全部" if all_users else filters.get("user_id", "unknown")
        return await self._clear_by_filters(filters, scope_label=scope)

    async def count_clear_memories(
        self,
        event: AstrMessageEvent,
        domain: str | None = None,
        all_users: bool = False,
    ) -> int:
        """统计 clear_memories 将影响的记录数。"""
        filters = self._build_query_filter(
            event,
            all_users=all_users,
            domain=domain,
            include_deprecated=True,
        )
        return await self.vec_db.count_documents(metadata_filter=filters)

    async def _clear_by_filters(
        self, filters: dict[str, Any], *, scope_label: str
    ) -> int:
        """底层清空逻辑：查询 doc_ids → 删除 → 反注册 → 同步统计"""
        doc_ids: list[str] = []
        count = 0
        try:
            doc_ids, count = await self._collect_kb_doc_ids_for_filters(filters)
        except Exception as e:
            logger.warning(f"[简单长期记忆] 查询待清空文档失败: {e}")
            try:
                count = await self.vec_db.count_documents(metadata_filter=filters)
            except Exception as ce:
                logger.warning(f"[简单长期记忆] 统计待清空文档失败: {ce}")

        await self.vec_db.delete_documents(metadata_filters=filters)

        try:
            await self._unregister_kb_documents(doc_ids)
            await self._sync_kb_stats()
        except Exception as e:
            logger.warning(f"[简单长期记忆] KB 文档批量删除失败: {e}")

        logger.info(f"[简单长期记忆] 清空 {count} 条记忆, 范围: {scope_label}")
        return count

    async def list_memories(
        self,
        event: AstrMessageEvent,
        domain: str | None = None,
        page: int = 1,
        page_size: int = 10,
        all_users: bool = False,
    ) -> tuple[list[dict[str, Any]], int, bool]:
        """列出用户的记忆（分页）

        Args:
            event: 消息事件
            domain: 记忆域过滤（可选）
            page: 页码（从 1 开始）
            page_size: 每页数量
            all_users: 为 True 时跳过用户过滤

        Returns:
            (记忆列表, 总数, 总数是否被扫描上限截断)
        """
        truncated = False
        if all_users:
            filters: dict[str, Any] = {
                "is_memory_record": True,
                "deprecated": False,
            }
            if domain:
                filters["domain"] = domain
            total = await self.vec_db.count_documents(metadata_filter=filters)
            offset = (page - 1) * page_size
            docs = await self.vec_db.document_storage.get_documents(
                metadata_filters=filters,
                offset=offset,
                limit=page_size,
            )
        else:
            parsed = UMOInfo.parse(event.unified_msg_origin)
            offset = (page - 1) * page_size
            if parsed.session_type == "group":
                docs, total, truncated = await self._list_visible_user_documents(
                    event, domain, page=page, page_size=page_size
                )
            else:
                filters = self._build_user_filter(event)
                filters["deprecated"] = False
                if domain:
                    filters["domain"] = domain
                total = await self.vec_db.count_documents(metadata_filter=filters)
                docs = await self.vec_db.document_storage.get_documents(
                    metadata_filters=filters,
                    offset=offset,
                    limit=page_size,
                )

        memories = []
        for doc in docs:
            metadata = _safe_parse_metadata(doc.get("metadata", {}))

            memories.append(
                {
                    "text": doc.get("text", ""),
                    "metadata": metadata,
                }
            )

        return memories, total, truncated

    async def _list_visible_user_documents(
        self,
        event: AstrMessageEvent,
        domain: str | None = None,
        *,
        page: int = 1,
        page_size: int = 10,
    ) -> tuple[list[dict[str, Any]], int, bool]:
        parsed = UMOInfo.parse(event.unified_msg_origin)
        scan_limit = self._memory_list_scan_limit(page, page_size)
        source_filters = [
            self._scope_filter(event, MemoryScope.PERSONAL),
            {
                "memory_scope": MemoryScope.PERSONAL,
                "owner_session_id": build_session_id(
                    parsed.platform_id, parsed.session_id
                ),
                "visibility": MemoryVisibility.GROUP,
                "is_memory_record": True,
                "deprecated": False,
            },
            self._scope_filter(event, MemoryScope.GROUP),
            self._scope_filter(event, MemoryScope.CONVERSATION),
        ]
        if domain:
            for filters in source_filters:
                filters["domain"] = domain

        docs_by_source = await asyncio.gather(
            *(
                self.vec_db.document_storage.get_documents(
                    metadata_filters=filters,
                    limit=scan_limit + 1,
                )
                for filters in source_filters
            )
        )

        visible = []
        seen = set()
        scanned_all_sources = True
        for docs in docs_by_source:
            if len(docs) > scan_limit:
                scanned_all_sources = False
            for doc in docs[:scan_limit]:
                metadata = _safe_parse_metadata(doc.get("metadata", {}))
                if not self._is_visible_shared_personal(event, metadata):
                    continue
                uri = metadata.get("uri") or doc.get("text", "")
                if uri in seen:
                    continue
                visible.append(doc)
                seen.add(uri)

        truncated = not scanned_all_sources
        total = len(visible) if not truncated else max(len(visible), scan_limit + 1)
        offset = (page - 1) * page_size
        return visible[offset : offset + page_size], total, truncated

    def _memory_list_scan_limit(self, page: int, page_size: int) -> int:
        try:
            configured = int(self.config.get("max_memory_list_scan", 200))
        except (TypeError, ValueError):
            configured = 200
        configured = max(1, configured)
        needed = max(1, page) * max(1, page_size)
        return min(configured, needed)

    async def get_memory_by_uri(
        self,
        event: AstrMessageEvent,
        uri: str,
    ) -> dict[str, Any] | None:
        """通过 URI 获取记忆

        Args:
            event: 消息事件
            uri: 记忆 URI

        Returns:
            记忆数据或 None
        """
        filters = self._build_user_filter(event)
        filters["uri"] = uri

        docs = await self.vec_db.document_storage.get_documents(
            metadata_filters=filters,
            limit=1,
        )

        if not docs:
            return None

        doc = docs[0]
        metadata = _safe_parse_metadata(doc.get("metadata", {}))

        return {
            "text": doc.get("text", ""),
            "metadata": metadata,
        }

    async def smart_update_memory(
        self,
        event: AstrMessageEvent,
        content: str,
        domain: str,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ) -> str:
        """智能更新记忆：相似则合并，否则新建

        Args:
            event: 消息事件
            content: 新的记忆内容
            domain: 记忆域
            similarity_threshold: 相似度阈值

        Returns:
            操作结果描述
        """
        # 检索相似记忆
        candidates = await self.recall_memories(
            event=event,
            query=content,
            domain=domain,
            top_k=5,
        )

        # 找到最相似的候选
        best_match = None
        best_score = 0.0

        for result in candidates:
            score = result.get("similarity", 0)
            if score > best_score and score >= similarity_threshold:
                best_score = score
                best_match = result

        if best_match:
            # 高相似度：创建新版本（简化处理，不自动合并）
            old_uri = best_match["metadata"].get("uri", "")
            logger.info(
                f"[简单长期记忆] 发现相似记忆: {old_uri}, 相似度: {best_score:.2f}"
            )
            # 返回提示，让调用方决定是否合并
            return f"found_similar:{old_uri}:{best_score:.2f}"
        else:
            # 低相似度：创建新记忆
            uri = await self.store_memory(
                event=event,
                content=content,
                domain=domain,
                uri=str(MemoryURI.generate(domain)),
                memory_type=MemoryType.NORMAL,
            )
            return f"created:{uri}"

    async def get_memory_stats(
        self,
        event: AstrMessageEvent,
        all_users: bool = False,
    ) -> dict[str, int]:
        """获取记忆统计

        Args:
            event: 消息事件
            all_users: 为 True 时统计全局数据

        Returns:
            统计信息字典
        """
        query_event = None if all_users else event
        filters = self._build_query_filter(query_event, all_users=all_users)

        # 总数
        total = await self.vec_db.count_documents(metadata_filter=filters)

        # 永久记忆数
        permanent_filters = {**filters, "memory_type": MemoryType.PERMANENT}
        permanent = await self.vec_db.count_documents(metadata_filter=permanent_filters)

        # 普通记忆数
        normal_filters = {**filters, "memory_type": MemoryType.NORMAL}
        normal = await self.vec_db.count_documents(metadata_filter=normal_filters)

        # 已压缩数
        compressed_filters = self._build_query_filter(
            query_event,
            all_users=all_users,
            include_deprecated=True,
        )
        compressed_filters["compressed"] = True
        compressed = await self.vec_db.count_documents(
            metadata_filter=compressed_filters
        )

        return {
            "total": total,
            "permanent": permanent,
            "normal": normal,
            "compressed": compressed,
        }

    async def forget_memory_by_user(
        self,
        event: AstrMessageEvent,
        uri: str,
        target_user_id: str,
    ) -> int:
        """按 user_id + uri 删除指定用户的记忆

        Args:
            event: 消息事件
            uri: 记忆 URI
            target_user_id: 目标用户 ID

        Returns:
            实际删除的记录数
        """
        filters: dict[str, Any] = {
            "user_id": target_user_id,
            "uri": uri,
            "is_memory_record": True,
        }
        return await self._delete_by_filters(filters, uri)

    async def clear_memories_by_user(
        self,
        event: AstrMessageEvent,
        target_user_id: str,
        domain: str | None = None,
    ) -> int:
        """按 user_id 清空指定用户全部记忆

        Args:
            event: 消息事件
            target_user_id: 目标用户 ID
            domain: 仅清空指定域的记忆（可选）

        Returns:
            删除的记忆数量
        """
        filters: dict[str, Any] = {
            "user_id": target_user_id,
            "is_memory_record": True,
        }
        if domain:
            filters["domain"] = domain
        return await self._clear_by_filters(
            filters, scope_label=f"管理员清空用户 {target_user_id}"
        )

    async def count_clear_memories_by_user(
        self,
        event: AstrMessageEvent,
        target_user_id: str,
        domain: str | None = None,
    ) -> int:
        """统计 clear_memories_by_user 将影响的记录数。"""
        filters: dict[str, Any] = {
            "user_id": target_user_id,
            "is_memory_record": True,
        }
        if domain:
            filters["domain"] = domain
        return await self.vec_db.count_documents(metadata_filter=filters)

    async def count_kb_memory_records_by_name(self, kb_name: str) -> int:
        """统计指定知识库中的插件记忆记录数。"""
        kb = await self.kb_mgr.get_kb_by_name(kb_name)
        if not kb:
            return 0
        return await kb.vec_db.count_documents(
            metadata_filter={"is_memory_record": True, "kb_id": kb.kb.kb_id}
        )

    async def kb_exists(self, kb_name: str) -> bool:
        """检查知识库是否存在。"""
        return bool(await self.kb_mgr.get_kb_by_name(kb_name))

    async def get_kb_id_by_name(self, kb_name: str | None) -> str:
        """按知识库名称获取稳定 KB ID。"""
        if not kb_name:
            return ""
        kb = await self.kb_mgr.get_kb_by_name(kb_name)
        return kb.kb.kb_id if kb else ""

    async def _resolve_rebuild_target_kb(
        self,
        rebuild_context: dict[str, Any] | None,
    ) -> KBHelper | None:
        """从重建上下文解析恢复目标，优先使用稳定 KB ID。"""
        if not rebuild_context:
            return self._kb_helper

        target_kb_id = rebuild_context.get("target_kb_id")
        if target_kb_id:
            resolved_by_id = await self.kb_mgr.get_kb(target_kb_id)
            if resolved_by_id:
                return resolved_by_id

        target_kb_name = rebuild_context.get("target_kb_name")
        if target_kb_name:
            return await self.kb_mgr.get_kb_by_name(target_kb_name)

        return self._kb_helper

    async def _resume_rebuild_from_snapshot(
        self,
        memory_records: list[dict[str, Any]],
        rebuild_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """从 KV 快照恢复中断的重建

        将快照中的记忆重新写入当前 KB（原地重建清空后或迁移目标已写入的场景）。
        写入失败的记录会保留，供下次启动继续恢复。

        Args:
            memory_records: 从 KV 恢复的记忆记录列表

        Returns:
            {"success": int, "failed": int, "remaining_records": list}
        """
        if not memory_records:
            return {
                "success": 0,
                "failed": 0,
                "remaining_records": list(memory_records or []),
            }

        target_kb = await self._resolve_rebuild_target_kb(rebuild_context)
        if not target_kb:
            target_label = (
                (rebuild_context or {}).get("target_kb_id")
                or (rebuild_context or {}).get("target_kb_name")
                or self._kb_name
                or "unknown"
            )
            logger.warning(f"[简单长期记忆] 快照恢复目标知识库不存在: {target_label}")
            return {
                "success": 0,
                "failed": len(memory_records),
                "remaining_records": list(memory_records),
                "error": f"目标知识库不存在: {target_label}",
            }

        expected_target_kb_id = (rebuild_context or {}).get("target_kb_id")
        if expected_target_kb_id and target_kb.kb.kb_id != expected_target_kb_id:
            logger.warning(
                "[简单长期记忆] 快照恢复目标知识库 ID 不匹配: "
                f"expected={expected_target_kb_id}, actual={target_kb.kb.kb_id}"
            )
            return {
                "success": 0,
                "failed": len(memory_records),
                "remaining_records": list(memory_records),
                "error": f"目标知识库 ID 不匹配: {target_kb.kb.kb_name}",
            }

        success = 0
        failed = 0
        remaining_records: list[dict[str, Any]] = []

        for record in memory_records:
            text = record.get("text", "")
            metadata = record.get("metadata", {})
            uri = metadata.get("uri", "")

            if not text:
                logger.warning("[简单长期记忆] 快照恢复跳过空内容记录")
                failed += 1
                continue

            try:
                if uri:
                    existing_doc = await self._find_memory_doc_by_uri(target_kb, uri)
                    if existing_doc:
                        if not await self._repair_kb_document_for_vector_doc(
                            target_kb,
                            existing_doc,
                            uri,
                        ):
                            failed += 1
                            remaining_records.append(record)
                            continue
                        success += 1
                        continue

                new_doc_id = str(uuid.uuid4())
                updated_metadata = self._normalize_rebuild_record_metadata(
                    text,
                    metadata,
                    kb_id=target_kb.kb.kb_id,
                    doc_id=new_doc_id,
                )
                content = updated_metadata.get("memory_content", "")
                uri = updated_metadata.get("uri", uri)
                formatted_content = format_memory_content(content, updated_metadata)

                await target_kb.vec_db.insert(
                    content=formatted_content,
                    metadata=updated_metadata,
                    id=new_doc_id,
                )
                await self._ensure_kb_document(
                    new_doc_id,
                    uri,
                    len(formatted_content),
                    kb_helper=target_kb,
                )
                success += 1
            except Exception as e:
                logger.warning(f"[简单长期记忆] 快照恢复写入失败 (URI: {uri}): {e}")
                failed += 1
                remaining_records.append(record)

        if success > 0:
            try:
                await self._sync_kb_stats(kb_helper=target_kb)
            except Exception as e:
                logger.warning(f"[简单长期记忆] 快照恢复后统计同步失败: {e}")

        logger.info(f"[简单长期记忆] 快照恢复完成: 成功 {success}, 失败 {failed}")
        return {
            "success": success,
            "failed": failed,
            "remaining_records": remaining_records,
        }

    def _normalize_rebuild_record_metadata(
        self,
        text: str,
        metadata: dict[str, Any],
        *,
        kb_id: str,
        doc_id: str,
    ) -> dict[str, Any]:
        """将重建缓存中的旧记录规范化为当前 metadata 结构。"""
        meta = MemoryMetadata.from_dict(metadata)
        content = meta.memory_content or self._memory_content_from_text(text)
        now = datetime.now(timezone.utc).isoformat()
        created_at = meta.created_at or now
        owner_user_id = meta.owner_user_id or meta.user_id
        owner_user_ids = meta.owner_user_ids or (
            [owner_user_id] if owner_user_id else []
        )

        normalized = {
            **metadata,
            **meta.to_dict(),
            "memory_content": content,
            "created_at": created_at,
            "last_recalled_at": meta.last_recalled_at or created_at,
            "memory_scope": normalize_memory_scope(meta.memory_scope),
            "owner_user_id": owner_user_id,
            "owner_user_ids": owner_user_ids,
            "visibility": normalize_visibility(meta.visibility, meta.memory_scope),
            "speaker_id": meta.speaker_id or meta.sender_id,
            "kb_doc_id": doc_id,
            "kb_id": kb_id,
            "chunk_index": 0,
            "is_memory_record": True,
            "deprecated": False,
        }
        if normalized["memory_scope"] in (MemoryScope.GLOBAL, MemoryScope.GROUP):
            normalized["visibility"] = MemoryVisibility.GROUP
        return normalized

    def _memory_content_from_text(self, text: str) -> str:
        for line in str(text).splitlines():
            if line.startswith("memory: "):
                return line.removeprefix("memory: ").strip()
        return str(text).strip()

    async def rebuild_memories(
        self,
        target_kb_name: str | None = None,
    ) -> dict[str, Any]:
        """重建或迁移所有记忆

        原地重建：拉取所有记忆到本地 → 持久化到 KV → 清空当前 KB → 从本地重新嵌入写入
        迁移模式：拉取所有记忆到本地 → 持久化到 KV → 写入目标 KB → 仅当全部成功时清空源 KB

        重建期间新的写入请求会被缓冲并持久化到 KV，完成后批量语义去重再写入。
        所有中间数据通过 KV 持久化，进程重启后可恢复。

        Args:
            target_kb_name: 目标知识库名称（为 None 时原地重建）

        Returns:
            {"total": int, "success": int, "failed": int,
             "target_kb": str, "is_migration": bool,
             "pending_flushed": int}

        Raises:
            ValueError: 目标知识库不存在
        """
        if self._rebuilding:
            raise RuntimeError("重建/迁移正在进行中，请等待完成")

        # 立即加锁，防止并发竞态；finally 会兜底释放，避免异常路径遗留锁
        self._rebuilding = True

        try:
            if self._kv_delete:
                for key in (
                    "rebuild_memory_records",
                    "rebuild_context",
                    "rebuild_status",
                ):
                    try:
                        await self._kv_delete(key)
                    except Exception as e:
                        logger.warning(f"[简单长期记忆] 清理旧重建缓存失败: {key}, {e}")

            source_kb = self._kb_helper
            source_kb_name = self._kb_name
            is_migration = (
                target_kb_name is not None and target_kb_name != source_kb_name
            )

            # 解析目标 KB
            if is_migration:
                target_kb = await self.kb_mgr.get_kb_by_name(target_kb_name)
                if not target_kb:
                    all_kbs = await self._list_all_kb_names()
                    available = ", ".join(all_kbs) if all_kbs else "(无)"
                    raise ValueError(
                        f"目标知识库 '{target_kb_name}' 不存在。"
                        f"当前可用知识库: {available}"
                    )
                target_existing = await target_kb.vec_db.count_documents(
                    metadata_filter={
                        "is_memory_record": True,
                        "kb_id": target_kb.kb.kb_id,
                    }
                )
                if target_existing > 0:
                    return await self._finalize_rebuild(
                        total=0,
                        success=0,
                        failed=0,
                        target_kb_name=target_kb_name,
                        is_migration=True,
                        error=(
                            f"目标知识库 '{target_kb_name}' 已存在 "
                            f"{target_existing} 条记忆记录，请先清空目标或更换知识库"
                        ),
                    )
            else:
                target_kb = source_kb
                target_kb_name = source_kb_name

            if self._kv_put:
                await self._kv_put(
                    "rebuild_context",
                    {
                        "source_kb_name": source_kb_name,
                        "source_kb_id": source_kb.kb.kb_id,
                        "target_kb_name": target_kb_name,
                        "target_kb_id": target_kb.kb.kb_id,
                        "is_migration": is_migration,
                    },
                )

            # ── 阶段 1: 拉取所有记忆到本地 + 持久化到 KV ──
            source_doc_ids: list[str] = []
            memory_records: list[dict[str, Any]] = []

            page_size = 5000
            seen_records: set[str] = set()

            async def collect_memory_records(metadata_filters: dict[str, Any]) -> None:
                offset = 0
                while True:
                    page_docs = await source_kb.vec_db.document_storage.get_documents(
                        offset=offset,
                        limit=page_size,
                        metadata_filters=metadata_filters,
                    )
                    if not page_docs:
                        break
                    offset += len(page_docs)
                    for doc in page_docs:
                        metadata = _safe_parse_metadata(doc.get("metadata", {}))
                        if not metadata.get("uri"):
                            continue
                        record_key = metadata.get("kb_doc_id") or metadata.get("uri")
                        if record_key in seen_records:
                            continue
                        seen_records.add(record_key)
                        old_doc_id = metadata.get("kb_doc_id", "")
                        if old_doc_id:
                            source_doc_ids.append(old_doc_id)
                        memory_records.append(
                            {
                                "text": doc.get("text", ""),
                                "metadata": metadata,
                            }
                        )

            source_kb_id = source_kb.kb.kb_id
            try:
                await collect_memory_records(
                    {"is_memory_record": True, "kb_id": source_kb_id}
                )
                await collect_memory_records(
                    {"deprecated": False, "kb_id": source_kb_id}
                )
            except Exception as e:
                logger.error(f"[简单长期记忆] 读取源知识库文档失败: {e}")
                return await self._finalize_rebuild(
                    total=0,
                    success=0,
                    failed=0,
                    target_kb_name=target_kb_name,
                    is_migration=is_migration,
                    error=f"读取源知识库失败: {e}",
                )

            total = len(memory_records)
            logger.info(
                f"[简单长期记忆] 已拉取 {total} 条记忆到本地, "
                f"模式: {'迁移' if is_migration else '重建'}"
            )

            # 安全检查：拉取 0 条但源 KB 有记忆记录时中止，防止误删
            if total == 0:
                source_count = await source_kb.vec_db.count_documents(
                    metadata_filter={"is_memory_record": True, "kb_id": source_kb_id}
                )
                if source_count > 0:
                    return await self._finalize_rebuild(
                        total=0,
                        success=0,
                        failed=0,
                        target_kb_name=target_kb_name,
                        is_migration=is_migration,
                        error=(
                            f"拉取到 0 条记忆，但源知识库中存在 {source_count} 条"
                            "记忆记录，中止重建以防止数据丢失。"
                            "请检查知识库配置后重试"
                        ),
                    )

            # 持久化拉取的数据到 KV（防进程崩溃丢失）
            if self._kv_put:
                await self._kv_put("rebuild_memory_records", memory_records)
                await self._kv_put("rebuild_status", "in_progress")

            # ── 阶段 2: 清空源 KB（原地重建时）或 留待后续清理（迁移时） ──
            if not is_migration:
                try:
                    await self._delete_rebuild_source_records(source_kb, memory_records)
                    if source_doc_ids:
                        await self._unregister_kb_documents(
                            source_doc_ids, kb_helper=source_kb
                        )
                    logger.info("[简单长期记忆] 已清空当前知识库旧记忆")
                except Exception as e:
                    logger.error(f"[简单长期记忆] 清空当前知识库失败: {e}")
                    return await self._finalize_rebuild(
                        total=total,
                        success=0,
                        failed=0,
                        target_kb_name=target_kb_name,
                        is_migration=is_migration,
                        error=f"清空当前知识库失败: {e}",
                    )

            # ── 阶段 3: 从本地缓存写入目标 KB ──
            success = 0
            failed = 0

            for record in memory_records:
                text = record["text"]
                metadata = record["metadata"]
                uri = metadata.get("uri", "")

                try:
                    new_doc_id = str(uuid.uuid4())
                    updated_metadata = self._normalize_rebuild_record_metadata(
                        text,
                        metadata,
                        kb_id=target_kb.kb.kb_id,
                        doc_id=new_doc_id,
                    )
                    content = updated_metadata.get("memory_content", "")
                    uri = updated_metadata.get("uri", uri)
                    formatted_content = format_memory_content(content, updated_metadata)

                    await target_kb.vec_db.insert(
                        content=formatted_content,
                        metadata=updated_metadata,
                        id=new_doc_id,
                    )
                    await self._ensure_kb_document(
                        new_doc_id,
                        uri,
                        len(formatted_content),
                        kb_helper=target_kb,
                    )

                    success += 1
                except Exception as e:
                    logger.warning(f"[简单长期记忆] 写入记忆失败 (URI: {uri}): {e}")
                    failed += 1

            # ── 阶段 4: 迁移模式 — 仅当全部成功时清空源 KB 并切换 ──
            migration_committed = False
            migration_commit_error = ""
            if is_migration:
                if failed == 0:
                    try:
                        await self._delete_rebuild_source_records(
                            source_kb, memory_records
                        )
                        if source_doc_ids:
                            await self._unregister_kb_documents(
                                source_doc_ids, kb_helper=source_kb
                            )
                        self._kb_helper = target_kb
                        self._kb_name = target_kb_name
                        migration_committed = True
                        logger.info(f"[简单长期记忆] 已迁移到知识库: {target_kb_name}")
                    except Exception as e:
                        migration_commit_error = f"清理源知识库失败: {e}"
                        logger.error(f"[简单长期记忆] {migration_commit_error}")
                elif failed > 0:
                    logger.warning(
                        f"[简单长期记忆] 存在 {failed} 条写入失败，"
                        "跳过源知识库清理以防止数据丢失"
                    )

            # ── 阶段 5: 同步统计 ──
            try:
                if is_migration:
                    await self._sync_kb_stats(kb_helper=source_kb)
                await self._sync_kb_stats(kb_helper=target_kb)
            except Exception as e:
                logger.warning(f"[简单长期记忆] 同步统计数据失败: {e}")

            # ── 阶段 6: 解锁 + 处理缓冲写入 ──
            self._rebuilding = False
            # 迁移未提交时插件仍使用 source KB，缓冲写入必须落到当前活跃 KB
            flush_target = (
                source_kb if is_migration and not migration_committed else None
            )
            pending_flushed = await self._flush_pending_writes(target_kb=flush_target)

            final_status = "completed" if failed == 0 else "partial"
            if migration_commit_error:
                final_status = "interrupted"
            if self._kv_put:
                await self._kv_put("rebuild_status", final_status)

            # ── 阶段 7: 完整性校验 ──
            verification_expected = success
            if not is_migration or migration_committed:
                verification_expected += pending_flushed
            verification = await self._verify_rebuild_integrity(
                target_kb,
                total,
                verification_expected,
            )

            logger.info(
                f"[简单长期记忆] 重建完成: 总计 {total}, 成功 {success}, "
                f"失败 {failed}, 目标: {target_kb_name}, "
                f"迁移: {is_migration}, 缓冲写入: {pending_flushed}, "
                f"校验: {'通过' if verification['passed'] else '不一致'}"
            )

            return {
                "status": final_status,
                "total": total,
                "success": success,
                "failed": failed,
                "target_kb": target_kb_name,
                "is_migration": is_migration,
                "pending_flushed": pending_flushed,
                "verification": verification,
                "migration_committed": migration_committed,
                "error": migration_commit_error,
            }
        finally:
            self._rebuilding = False

    async def _verify_rebuild_integrity(
        self,
        target_kb: KBHelper,
        expected_total: int,
        expected_count: int,
    ) -> dict[str, Any]:
        """校验目标 KB 中的记忆记录数是否与预期一致

        Args:
            target_kb: 目标 KBHelper
            expected_total: 拉取的记忆总数
            expected_count: 目标 KB 中预期存在的记忆数量

        Returns:
            {"passed": bool, "expected": int, "actual": int, "diff": int}
        """
        try:
            actual = await target_kb.vec_db.count_documents(
                metadata_filter={"is_memory_record": True, "kb_id": target_kb.kb.kb_id}
            )
        except Exception as e:
            logger.warning(f"[简单长期记忆] 完整性校验失败: {e}")
            return {
                "passed": False,
                "expected": expected_count,
                "actual": -1,
                "diff": -1,
                "error": str(e),
            }

        passed = actual == expected_count
        return {
            "passed": passed,
            "expected": expected_count,
            "actual": actual,
            "diff": actual - expected_count,
        }

    async def _finalize_rebuild(
        self,
        total: int,
        success: int,
        failed: int,
        target_kb_name: str,
        is_migration: bool,
        error: str = "",
    ) -> dict[str, Any]:
        """重建失败时安全收尾：解锁 + flush 缓冲写入

        KV 缓存数据保留不清理，等待用户确认后手动清理。

        Returns:
            包含 "status": "interrupted" 的结果字典
        """
        self._rebuilding = False
        pending_flushed = await self._flush_pending_writes()

        if self._kv_put:
            await self._kv_put("rebuild_status", "interrupted")

        logger.warning(
            f"[简单长期记忆] 重建异常终止: 总计 {total}, 成功 {success}, "
            f"失败 {failed}, 已 flush 缓冲 {pending_flushed} 条"
        )

        return {
            "status": "interrupted",
            "total": total,
            "success": success,
            "failed": failed,
            "target_kb": target_kb_name,
            "is_migration": is_migration,
            "pending_flushed": pending_flushed,
            "error": error,
        }

    async def _flush_pending_writes(self, target_kb: KBHelper | None = None) -> int:
        """处理重建期间缓冲的写入请求，语义去重后写入 KB

        Args:
            target_kb: 指定写入目标 KB（迁移未提交时使用 target KB），
                       为 None 时使用当前 self._kb_helper

        Returns:
            成功写入的缓冲条数
        """
        if not self._pending_writes:
            return 0

        write_kb = target_kb or self._kb_helper
        if not write_kb:
            return 0

        pending = list(self._pending_writes)
        self._pending_writes.clear()
        flushed = 0
        retry_pending: list[dict[str, Any]] = []

        for item in pending:
            content = item["content"]
            uri = item.get("uri", "")
            try:
                existing_doc = await self._find_memory_doc_by_uri(write_kb, uri)
                if existing_doc:
                    if await self._repair_kb_document_for_vector_doc(
                        write_kb,
                        existing_doc,
                        uri,
                    ):
                        flushed += 1
                        continue
                    retry_pending.append(item)
                    continue

                # 语义去重：召回相似记忆，高相似度则跳过
                memory_scope = normalize_memory_scope(
                    item.get("memory_scope", MemoryScope.PERSONAL)
                )
                filters: dict[str, Any] = {
                    "memory_scope": memory_scope,
                    "domain": item["domain"],
                    "is_memory_record": True,
                    "deprecated": False,
                }
                if memory_scope == MemoryScope.GLOBAL:
                    pass
                elif memory_scope == MemoryScope.GROUP:
                    filters["owner_session_id"] = item.get("owner_session_id", "")
                elif memory_scope == MemoryScope.CONVERSATION:
                    filters["umo"] = item["umo"]
                else:
                    filters["owner_user_id"] = item.get(
                        "owner_user_id", item["user_id"]
                    )
                candidates = await write_kb.vec_db.retrieve(
                    query=content,
                    k=1,
                    metadata_filters=filters,
                )
                if candidates and candidates[0].similarity >= SIMILARITY_THRESHOLD:
                    logger.debug(
                        f"[简单长期记忆] 缓冲写入去重跳过: {item['uri']}, "
                        f"相似度: {candidates[0].similarity:.2f}"
                    )
                    continue

                # 构建完整 metadata 并写入
                now = datetime.now(timezone.utc).isoformat()
                metadata = {
                    "user_id": item["user_id"],
                    "platform_id": item["platform_id"],
                    "sender_id": item["sender_id"],
                    "umo": item["umo"],
                    "session_type": item["session_type"],
                    "session_id": item["session_id"],
                    "created_at": now,
                    "last_recalled_at": now,
                    "recall_count": 0,
                    "compressed": False,
                    "memory_scope": item.get("memory_scope", MemoryScope.PERSONAL),
                    "owner_user_id": item.get("owner_user_id", item["user_id"]),
                    "owner_user_ids": item.get("owner_user_ids", [item["user_id"]]),
                    "owner_session_id": item.get("owner_session_id", ""),
                    "visibility": item.get("visibility", MemoryVisibility.PRIVATE),
                    "speaker_id": item.get("speaker_id", item["sender_id"]),
                    "domain": item["domain"],
                    "uri": item["uri"],
                    "version": 1,
                    "deprecated": False,
                    "memory_type": item["memory_type"],
                    "disclosure": item["disclosure"],
                    "importance": item["importance"],
                    "subject": item.get("subject", ""),
                    "entities": item.get("entities", []),
                    "topics": item.get("topics", []),
                    "memory_content": content,
                }

                doc_id = str(uuid.uuid4())
                metadata["kb_doc_id"] = doc_id
                metadata["kb_id"] = write_kb.kb.kb_id
                metadata["chunk_index"] = 0
                metadata["is_memory_record"] = True

                formatted_content = format_memory_content(content, metadata)
                await write_kb.vec_db.insert(
                    content=formatted_content,
                    metadata=metadata,
                    id=doc_id,
                )

                if not await self._ensure_kb_document(
                    doc_id,
                    item["uri"],
                    len(formatted_content),
                    kb_helper=write_kb,
                ):
                    retry_pending.append(item)
                    continue

                flushed += 1
                logger.debug(f"[简单长期记忆] 缓冲写入已处理: {item['uri']}")
            except Exception as e:
                logger.warning(f"[简单长期记忆] 缓冲写入处理失败: {item['uri']}, {e}")
                retry_pending.append(item)

        if flushed:
            try:
                await self._sync_kb_stats(kb_helper=write_kb)
            except Exception as e:
                logger.warning(f"[简单长期记忆] 缓冲写入后统计同步失败: {e}")

        self._pending_writes = retry_pending
        if retry_pending:
            if self._kv_put:
                await self._kv_put("rebuild_pending_writes", retry_pending)
        elif self._kv_delete:
            try:
                await self._kv_delete("rebuild_pending_writes")
            except Exception:
                pass

        logger.info(
            f"[简单长期记忆] 缓冲写入处理完成: "
            f"共 {len(pending)} 条, 写入 {flushed} 条, "
            f"待重试 {len(retry_pending)} 条"
        )
        return flushed

    async def _list_all_kb_names(self) -> list[str]:
        """获取所有知识库名称列表"""
        try:
            kbs = await self.kb_mgr.list_kbs()
            return [kb.kb_name for kb in kbs] if kbs else []
        except Exception:
            return []

    async def clear_rebuild_cache(self) -> dict[str, bool]:
        """清理重建/迁移过程中缓存的 KV 数据

        在确认重建结果无误后调用。

        Returns:
            各 KV 键的清理结果
        """
        result: dict[str, bool] = {}
        keys = [
            "rebuild_memory_records",
            "rebuild_pending_writes",
            "rebuild_status",
            "rebuild_context",
        ]
        for key in keys:
            try:
                if self._kv_delete:
                    await self._kv_delete(key)
                result[key] = True
            except Exception:
                result[key] = False
        logger.info(f"[简单长期记忆] 已清理重建缓存: {result}")
        return result

    def _hash_rebuild_cache_parts(self, *parts: Any) -> str:
        payload = json.dumps(
            parts,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def rebuild_cache_fingerprint(self, status: dict[str, Any]) -> str:
        """生成重建缓存确认指纹。"""
        cached = status.get("fingerprint")
        if cached:
            return str(cached)
        return self._hash_rebuild_cache_parts(
            status.get("memory_records", 0),
            status.get("pending_writes", 0),
            status.get("status"),
            status.get("source_kb"),
            status.get("target_kb"),
            status.get("is_migration"),
        )

    async def get_rebuild_cache_status(self) -> dict[str, Any]:
        """查看重建缓存的当前状态

        Returns:
            缓存状态信息
        """
        status: dict[str, Any] = {}
        if self._kv_get:
            records = await self._kv_get("rebuild_memory_records", None)
            pending = await self._kv_get("rebuild_pending_writes", None)
            rebuild_status = await self._kv_get("rebuild_status", None)
            rebuild_context = await self._kv_get("rebuild_context", None)
            status["memory_records"] = len(records) if records else 0
            status["pending_writes"] = len(pending) if pending else 0
            status["status"] = rebuild_status
            if isinstance(rebuild_context, dict):
                status["source_kb"] = rebuild_context.get("source_kb_name", "")
                status["target_kb"] = rebuild_context.get("target_kb_name", "")
                status["is_migration"] = bool(rebuild_context.get("is_migration"))
            status["fingerprint"] = self._hash_rebuild_cache_parts(
                records or [],
                pending or [],
                rebuild_status,
                rebuild_context or {},
            )
        return status
