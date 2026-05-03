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

import json
import logging
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .memory_protocol import (
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

logger = logging.getLogger("astrbot")

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
        if memory_scope == MemoryScope.GROUP
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

    def _build_memory_filter(
        self,
        event: AstrMessageEvent,
        global_memory: bool = True,
    ) -> dict[str, Any]:
        """构建记忆召回过滤器

        Args:
            event: 消息事件
            global_memory: 是否全局记忆模式

        Returns:
            metadata 过滤器字典
        """
        filters = self._build_user_filter(event)

        if not global_memory:
            # 非全局模式：仅召回当前会话的记忆
            filters["umo"] = event.unified_msg_origin

        return filters

    def _scope_filter(
        self,
        event: AstrMessageEvent,
        memory_scope: str,
        global_memory: bool = True,
    ) -> dict[str, Any]:
        _, owner_user_id, owner_session_id = self._event_scope_ids(event)
        scope = normalize_memory_scope(memory_scope)

        if scope == MemoryScope.GROUP:
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

    def _legacy_personal_filter(
        self,
        event: AstrMessageEvent,
        global_memory: bool = True,
    ) -> dict[str, Any]:
        filters = self._build_memory_filter(event, global_memory)
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
                filters = self._build_memory_filter(event, global_memory)
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
            speaker_id=owner_sender_ids[0],
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

        if all_users:
            filters = {"is_memory_record": True, "deprecated": False}
            if domain:
                filters["domain"] = domain
            return await self._retrieve_with_filter(query, top_k, filters)

        global_memory = self.config.get("global_memory", True)
        filters_list = self._build_recall_filters(
            event,
            global_memory=global_memory,
            domain=domain,
            memory_scope=memory_scope,
        )

        results = []
        for filters, legacy_personal, owner_user_id, require_owner_list in filters_list:
            results.extend(
                await self._retrieve_with_filter(
                    query,
                    top_k,
                    filters,
                    legacy_personal=legacy_personal,
                    owner_user_id=owner_user_id,
                    require_owner_list=require_owner_list,
                )
            )

        memories = self._dedupe_memories(results)[:top_k]
        logger.debug(f"[简单长期记忆] 召回 {len(memories)} 条记忆")
        return memories

    def _build_recall_filters(
        self,
        event: AstrMessageEvent,
        global_memory: bool,
        domain: str | None = None,
        memory_scope: str | None = None,
    ) -> list[tuple[dict[str, Any], bool, str | None, bool]]:
        parsed = UMOInfo.parse(event.unified_msg_origin)
        current_user_id = build_user_id(parsed.platform_id, event.get_sender_id())
        scopes = (
            [normalize_memory_scope(memory_scope)]
            if memory_scope
            else [MemoryScope.PERSONAL]
        )
        if not memory_scope and parsed.session_type == "group":
            scopes.extend([MemoryScope.GROUP, MemoryScope.CONVERSATION])
        elif not memory_scope and not global_memory:
            scopes.append(MemoryScope.CONVERSATION)

        filters_list = []
        for scope in scopes:
            filters = self._scope_filter(event, scope, global_memory)
            if domain:
                filters["domain"] = domain
            filters_list.append((filters, False, None, False))
            if scope == MemoryScope.PERSONAL:
                legacy_filters = self._legacy_personal_filter(event, global_memory)
                if domain:
                    legacy_filters["domain"] = domain
                filters_list.append((legacy_filters, True, current_user_id, False))
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
                    filters_list.append(
                        (group_personal_filters, True, current_user_id, True)
                    )
        return filters_list

    async def _retrieve_with_filter(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any],
        legacy_personal: bool = False,
        owner_user_id: str | None = None,
        require_owner_list: bool = False,
    ) -> list[dict[str, Any]]:
        use_rerank = self.config.get("use_reranker", True)
        results = await self.vec_db.retrieve(
            query=query,
            k=top_k,
            rerank=use_rerank,
            metadata_filters=filters,
        )

        # 解析结果
        memories = []
        for result in results:
            data = result.data
            metadata = _safe_parse_metadata(data.get("metadata", {}))
            if legacy_personal and not self._is_visible_personal_memory(
                metadata, owner_user_id, require_owner_list
            ):
                continue

            memories.append(
                {
                    "text": data.get("text", ""),
                    "metadata": metadata,
                    "similarity": result.similarity,
                }
            )

        return memories

    def _is_visible_personal_memory(
        self,
        metadata: dict[str, Any],
        owner_user_id: str | None,
        require_owner_list: bool = False,
    ) -> bool:
        scope = metadata.get("memory_scope")
        if scope not in (None, "", MemoryScope.PERSONAL):
            return False
        owner_user_ids = metadata.get("owner_user_ids")
        if isinstance(owner_user_ids, list) and owner_user_ids:
            return owner_user_id in owner_user_ids if owner_user_id else True
        if require_owner_list:
            return False
        return metadata.get("owner_user_id", metadata.get("user_id")) == owner_user_id

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

    async def forget_memory(
        self,
        event: AstrMessageEvent,
        uri: str,
    ) -> tuple[int, bool]:
        """删除当前用户指定 URI 的记忆（LLM 工具用，按 user_id 隔离）

        Args:
            event: 消息事件
            uri: 记忆 URI

        Returns:
            (删除数, URI是否存在但属于其他用户)
        """
        filters = self._build_user_filter(event)
        filters["uri"] = uri
        deleted = await self._delete_by_filters(filters, uri)
        if deleted > 0:
            return (deleted, False)
        # 检查该 URI 是否存在（属于其他用户）
        exists = await self.vec_db.count_documents(metadata_filter={"uri": uri})
        return (0, exists > 0)

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
        # 查询匹配记录的 kb_doc_id 以便同步删除 KB 文档记录
        doc_ids: list[str] = []
        deleted = 0
        try:
            docs = await self.vec_db.document_storage.get_documents(
                metadata_filters=filters, limit=100
            )
            deleted = len(docs)
            for doc in docs:
                md = _safe_parse_metadata(doc.get("metadata", {}))
                if md.get("kb_doc_id"):
                    doc_ids.append(md["kb_doc_id"])
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

    async def _clear_by_filters(
        self, filters: dict[str, Any], *, scope_label: str
    ) -> int:
        """底层清空逻辑：查询 doc_ids → 删除 → 反注册 → 同步统计"""
        doc_ids: list[str] = []
        count = 0
        try:
            docs = await self.vec_db.document_storage.get_documents(
                metadata_filters=filters, limit=10000
            )
            count = len(docs)
            for doc in docs:
                md = _safe_parse_metadata(doc.get("metadata", {}))
                if md.get("kb_doc_id"):
                    doc_ids.append(md["kb_doc_id"])
        except Exception:
            count = await self.vec_db.count_documents(metadata_filter=filters)

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
    ) -> tuple[list[dict[str, Any]], int]:
        """列出用户的记忆（分页）

        Args:
            event: 消息事件
            domain: 记忆域过滤（可选）
            page: 页码（从 1 开始）
            page_size: 每页数量
            all_users: 为 True 时跳过用户过滤

        Returns:
            (记忆列表, 总数)
        """
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
            docs = await self._list_visible_user_documents(
                event, domain, page=page, page_size=page_size
            )
            total = len(docs)
            offset = (page - 1) * page_size
            docs = docs[offset : offset + page_size]

        memories = []
        for doc in docs:
            metadata = _safe_parse_metadata(doc.get("metadata", {}))

            memories.append(
                {
                    "text": doc.get("text", ""),
                    "metadata": metadata,
                }
            )

        return memories, total

    async def _list_visible_user_documents(
        self,
        event: AstrMessageEvent,
        domain: str | None = None,
        *,
        page: int = 1,
        page_size: int = 10,
    ) -> list[dict[str, Any]]:
        parsed = UMOInfo.parse(event.unified_msg_origin)
        current_user_id = build_user_id(parsed.platform_id, event.get_sender_id())
        scan_limit = self._memory_list_scan_limit(page, page_size)
        filters = self._build_user_filter(event)
        filters["deprecated"] = False
        if domain:
            filters["domain"] = domain

        docs = await self.vec_db.document_storage.get_documents(
            metadata_filters=filters,
            limit=scan_limit,
        )
        if parsed.session_type != "group":
            return docs

        group_filters = {
            "memory_scope": MemoryScope.PERSONAL,
            "owner_session_id": build_session_id(parsed.platform_id, parsed.session_id),
            "visibility": MemoryVisibility.GROUP,
            "is_memory_record": True,
            "deprecated": False,
        }
        if domain:
            group_filters["domain"] = domain
        group_docs = await self.vec_db.document_storage.get_documents(
            metadata_filters=group_filters,
            limit=scan_limit,
        )

        visible = []
        seen = set()
        for doc in [*docs, *group_docs]:
            metadata = _safe_parse_metadata(doc.get("metadata", {}))
            uri = metadata.get("uri") or doc.get("text", "")
            if uri in seen:
                continue
            if self._is_visible_personal_memory(
                metadata,
                current_user_id,
                require_owner_list=doc in group_docs,
            ):
                visible.append(doc)
                seen.add(uri)
        return visible

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
                memory_type=domain,
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
        if all_users:
            filters = self._build_query_filter(None, all_users=True)
        else:
            # 原行为：非 all_users 模式下不加 deprecated 过滤
            filters = self._build_user_filter(event)

        # 总数
        total = await self.vec_db.count_documents(metadata_filter=filters)

        # 永久记忆数
        permanent_filters = {**filters, "memory_type": MemoryType.PERMANENT}
        permanent = await self.vec_db.count_documents(metadata_filter=permanent_filters)

        # 普通记忆数
        normal_filters = {**filters, "memory_type": MemoryType.NORMAL}
        normal = await self.vec_db.count_documents(metadata_filter=normal_filters)

        # 已压缩数
        compressed_filters = {**filters, "compressed": True}
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

    async def _resume_rebuild_from_snapshot(
        self, memory_records: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """从 KV 快照恢复中断的重建

        将快照中的记忆重新写入当前 KB（原地重建清空后或迁移目标已写入的场景）。
        写入失败的记录会保留，供下次启动继续恢复。

        Args:
            memory_records: 从 KV 恢复的记忆记录列表

        Returns:
            {"success": int, "failed": int, "remaining_records": list}
        """
        if not self._kb_helper or not memory_records:
            return {
                "success": 0,
                "failed": 0,
                "remaining_records": list(memory_records or []),
            }

        target_kb = self._kb_helper
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
                new_doc_id = str(uuid.uuid4())
                updated_metadata = {
                    **metadata,
                    "kb_doc_id": new_doc_id,
                    "kb_id": target_kb.kb.kb_id,
                    "chunk_index": 0,
                    "is_memory_record": True,
                }

                await target_kb.vec_db.insert(
                    content=text,
                    metadata=updated_metadata,
                )
                await self._register_kb_document(
                    new_doc_id, uri, len(text), kb_helper=target_kb
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
            if self._kv_put:
                await self._kv_put("rebuild_status", "in_progress")

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
            else:
                target_kb = source_kb
                target_kb_name = source_kb_name

            # ── 阶段 1: 拉取所有记忆到本地 + 持久化到 KV ──
            source_doc_ids: list[str] = []
            memory_records: list[dict[str, Any]] = []

            # 分页拉取，兼容新旧格式记忆
            # 新格式: metadata 含 is_memory_record=True
            # 旧格式: 无 is_memory_record 字段，但有 uri/domain 等记忆字段
            page_size = 5000
            offset = 0
            while True:
                try:
                    # 优先按 is_memory_record 拉取
                    page_docs = await source_kb.vec_db.document_storage.get_documents(
                        offset=offset,
                        limit=page_size,
                        metadata_filters={"is_memory_record": True},
                    )
                    if not page_docs:
                        # 回退：按 deprecated=False 拉取（兼容旧格式）
                        page_docs = (
                            await source_kb.vec_db.document_storage.get_documents(
                                offset=offset,
                                limit=page_size,
                                metadata_filters={"deprecated": False},
                            )
                        )
                except Exception as e:
                    logger.error(
                        f"[简单长期记忆] 读取源知识库文档失败 (offset={offset}): {e}"
                    )
                    return await self._finalize_rebuild(
                        total=0,
                        success=0,
                        failed=0,
                        target_kb_name=target_kb_name,
                        is_migration=is_migration,
                        error=f"读取源知识库失败: {e}",
                    )
                if not page_docs:
                    break
                offset += len(page_docs)
                for doc in page_docs:
                    metadata = _safe_parse_metadata(doc.get("metadata", {}))
                    # 跳过非记忆文档：必须有 uri 字段才视为记忆
                    if not metadata.get("uri"):
                        continue
                    old_doc_id = metadata.get("kb_doc_id", "")
                    if old_doc_id:
                        source_doc_ids.append(old_doc_id)
                    memory_records.append(
                        {
                            "text": doc.get("text", ""),
                            "metadata": metadata,
                        }
                    )

            total = len(memory_records)
            logger.info(
                f"[简单长期记忆] 已拉取 {total} 条记忆到本地, "
                f"模式: {'迁移' if is_migration else '重建'}"
            )

            # 安全检查：拉取 0 条但源 KB 有记忆记录时中止，防止误删
            if total == 0:
                source_count = await source_kb.vec_db.count_documents(
                    metadata_filter={"is_memory_record": True}
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

            # ── 阶段 2: 清空源 KB（原地重建时）或 留待后续清理（迁移时） ──
            if not is_migration:
                try:
                    await source_kb.vec_db.delete_documents(
                        metadata_filters={"is_memory_record": True}
                    )
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
                    updated_metadata = {
                        **metadata,
                        "kb_doc_id": new_doc_id,
                        "kb_id": target_kb.kb.kb_id,
                        "chunk_index": 0,
                        "is_memory_record": True,
                    }

                    await target_kb.vec_db.insert(
                        content=text,
                        metadata=updated_metadata,
                    )
                    await self._register_kb_document(
                        new_doc_id, uri, len(text), kb_helper=target_kb
                    )

                    success += 1
                except Exception as e:
                    logger.warning(f"[简单长期记忆] 写入记忆失败 (URI: {uri}): {e}")
                    failed += 1

            # ── 阶段 4: 迁移模式 — 仅当全部成功时清空源 KB 并切换 ──
            migration_committed = False
            if is_migration:
                if failed == 0 and success > 0:
                    try:
                        await source_kb.vec_db.delete_documents(
                            metadata_filters={"is_memory_record": True}
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
                        logger.error(f"[简单长期记忆] 清理源知识库失败: {e}")
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
                metadata_filter={"is_memory_record": True}
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
        pending = list(self._pending_writes)
        self._pending_writes.clear()
        flushed = 0

        for item in pending:
            content = item["content"]
            try:
                # 语义去重：召回相似记忆，高相似度则跳过
                filters: dict[str, Any] = {
                    "user_id": item["user_id"],
                    "memory_scope": item.get("memory_scope", MemoryScope.PERSONAL),
                    "is_memory_record": True,
                    "deprecated": False,
                }
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
                )

                try:
                    await self._register_kb_document(
                        doc_id,
                        item["uri"],
                        len(formatted_content),
                        kb_helper=write_kb,
                    )
                except Exception as e:
                    logger.warning(f"[简单长期记忆] 缓冲写入文档注册失败: {e}")

                flushed += 1
                logger.debug(f"[简单长期记忆] 缓冲写入已处理: {item['uri']}")
            except Exception as e:
                logger.warning(f"[简单长期记忆] 缓冲写入处理失败: {item['uri']}, {e}")

        if flushed:
            try:
                await self._sync_kb_stats(kb_helper=write_kb)
            except Exception as e:
                logger.warning(f"[简单长期记忆] 缓冲写入后统计同步失败: {e}")

        # flush 完成后清理 KV 中的 pending_writes，防止重启重复回放
        if self._kv_delete:
            try:
                await self._kv_delete("rebuild_pending_writes")
            except Exception:
                pass

        logger.info(
            f"[简单长期记忆] 缓冲写入处理完成: "
            f"共 {len(pending)} 条, 写入 {flushed} 条, 去重跳过 {len(pending) - flushed} 条"
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
            status["memory_records"] = len(records) if records else 0
            status["pending_writes"] = len(pending) if pending else 0
            status["status"] = rebuild_status
        return status
