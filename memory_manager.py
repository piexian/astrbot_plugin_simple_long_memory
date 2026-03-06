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
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .memory_protocol import (
    MemoryType,
    MemoryURI,
    UMOInfo,
    build_user_id,
    format_memory_content,
)

if TYPE_CHECKING:
    from astrbot.core.knowledge_base.kb_helper import KBHelper
    from astrbot.core.knowledge_base.kb_mgr import KnowledgeBaseManager
    from astrbot.core.platform import AstrMessageEvent

logger = logging.getLogger("astrbot")

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
    ):
        self.kb_mgr = kb_mgr
        self.config = config
        self._kb_helper: KBHelper | None = None

    async def initialize(self) -> None:
        """初始化记忆管理器

        Raises:
            ValueError: 知识库未配置或不存在
        """
        kb_name_raw = self.config.get("kb_name", [])
        kb_name = kb_name_raw[0] if isinstance(kb_name_raw, list) and kb_name_raw else kb_name_raw
        if not kb_name:
            raise ValueError("记忆知识库未配置，请在插件设置中选择一个知识库")

        kb = await self.kb_mgr.get_kb_by_name(kb_name)
        if not kb:
            raise ValueError(f"知识库 '{kb_name}' 不存在，请先在知识库管理中创建")

        self._kb_helper = kb
        logger.info(f"[长期记忆] 已连接知识库: {kb_name}")

    @property
    def vec_db(self):
        """获取向量数据库实例"""
        if not self._kb_helper:
            raise RuntimeError("记忆管理器未初始化")
        return self._kb_helper.vec_db

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
        parsed = UMOInfo.parse(umo)
        user_id = build_user_id(parsed.platform_id, event.get_sender_id())

        return {
            "user_id": user_id,
            "platform_id": parsed.platform_id,
            "sender_id": event.get_sender_id(),
            "umo": umo,
            "session_type": parsed.session_type,
            "session_id": parsed.session_id,
            "created_at": datetime.utcnow().isoformat(),
            "last_recalled_at": datetime.utcnow().isoformat(),
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

        if uri is None:
            uri = str(MemoryURI.generate(domain))

        metadata = self._build_memory_metadata(
            event,
            domain=domain,
            uri=uri,
            version=1,
            deprecated=False,
            memory_type=memory_type,
            disclosure=disclosure,
            importance=importance,
        )

        # 格式化内容
        formatted_content = format_memory_content(content, metadata)

        # 存储到向量数据库
        await self.vec_db.insert(
            content=formatted_content,
            metadata=metadata,
        )

        logger.debug(f"[长期记忆] 存储记忆: {uri}, 用户: {metadata['user_id']}")
        return uri

    async def recall_memories(
        self,
        event: AstrMessageEvent,
        query: str,
        domain: str | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """召回相关记忆（自动按用户隔离）

        Args:
            event: 消息事件
            query: 查询文本
            domain: 记忆域过滤（可选）
            top_k: 返回数量（可选，使用配置值）

        Returns:
            记忆列表，每项包含 'text' 和 'metadata'
        """
        if top_k is None:
            top_k = self.config.get("max_memories_per_inject", 5)

        global_memory = self.config.get("global_memory", True)

        # 构建过滤器
        filters = self._build_memory_filter(event, global_memory)
        filters["deprecated"] = False  # 排除废弃的记忆

        if domain:
            filters["domain"] = domain

        # 调用向量检索
        results = await self.vec_db.retrieve(
            query=query,
            k=top_k,
            metadata_filters=filters,
        )

        # 解析结果
        memories = []
        for result in results:
            data = result.data
            metadata = _safe_parse_metadata(data.get("metadata", {}))

            memories.append(
                {
                    "text": data.get("text", ""),
                    "metadata": metadata,
                    "similarity": result.similarity,
                }
            )

        logger.debug(f"[长期记忆] 召回 {len(memories)} 条记忆")
        return memories

    async def forget_memory(
        self,
        event: AstrMessageEvent,
        uri: str,
    ) -> bool:
        """删除指定 URI 的记忆

        Args:
            event: 消息事件
            uri: 记忆 URI

        Returns:
            是否删除成功
        """
        filters = self._build_user_filter(event)
        filters["uri"] = uri

        await self.vec_db.delete_documents(metadata_filters=filters)
        logger.info(f"[长期记忆] 删除记忆: {uri}")
        return True

    async def clear_memories(
        self,
        event: AstrMessageEvent,
        domain: str | None = None,
    ) -> int:
        """清空用户的所有记忆

        Args:
            event: 消息事件
            domain: 仅清空指定域的记忆（可选）

        Returns:
            删除的记忆数量
        """
        filters = self._build_user_filter(event)
        if domain:
            filters["domain"] = domain

        # 先统计数量
        count = await self.vec_db.count_documents(metadata_filter=filters)

        # 执行删除
        await self.vec_db.delete_documents(metadata_filters=filters)

        logger.info(f"[长期记忆] 清空 {count} 条记忆, 用户: {filters['user_id']}")
        return count

    async def list_memories(
        self,
        event: AstrMessageEvent,
        domain: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """列出用户的所有记忆

        Args:
            event: 消息事件
            domain: 记忆域过滤（可选）
            limit: 最大返回数量

        Returns:
            记忆列表
        """
        filters = self._build_user_filter(event)
        filters["deprecated"] = False

        if domain:
            filters["domain"] = domain

        # 使用 document_storage 直接查询
        docs = await self.vec_db.document_storage.get_documents(
            metadata_filters=filters,
            limit=limit,
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

        return memories

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
            logger.info(f"[长期记忆] 发现相似记忆: {old_uri}, 相似度: {best_score:.2f}")
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
    ) -> dict[str, int]:
        """获取用户记忆统计

        Args:
            event: 消息事件

        Returns:
            统计信息字典
        """
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
