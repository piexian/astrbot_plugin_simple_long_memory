"""物理清理：删除已废弃超期的记忆。

按 deprecated_at + N 天宽限期判断，物理删除 FAISS 向量、
SQLite 文档记录和 KB 文档记录，并级联清理关联边。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from astrbot.api import logger

if TYPE_CHECKING:
    from .links import MemoryLinkManager


async def purge_deprecated_memories(
    vec_db: Any,
    kb_helper: Any,
    link_manager: MemoryLinkManager | None,
    after_days: int = 7,
) -> dict[str, int]:
    """物理删除 deprecated 超过 after_days 天的记忆。

    按精确的 kb_doc_id 逐条删除，不会误删宽限期内的记录。
    只用成功删除的子集清理 KB 文档和关联边，保证跨存储一致性。

    Args:
        vec_db: 向量数据库实例
        kb_helper: KBHelper 实例
        link_manager: 关联表管理器（可选，用于级联清理）
        after_days: 废弃超期天数

    Returns:
        {"purged": int, "links_cleaned": int}
    """
    if after_days <= 0 or not kb_helper:
        return {"purged": 0, "links_cleaned": 0}

    kb_id = kb_helper.kb.kb_id
    cutoff_iso = (datetime.now(timezone.utc) - timedelta(days=after_days)).isoformat()

    # 1. 精确查询待清理的记忆，收集 (doc_id, uri) 配对
    candidates: list[tuple[str, str]] = []  # (kb_doc_id, uri)
    try:
        doc_storage = vec_db.document_storage
        async with doc_storage.get_session() as session:
            from sqlalchemy import text as sa_text

            rows = (
                await session.execute(
                    sa_text(
                        "SELECT metadata FROM documents "
                        "WHERE json_extract(metadata,'$.is_memory_record') = 1 "
                        "  AND json_extract(metadata,'$.deprecated') = 1 "
                        "  AND json_extract(metadata,'$.kb_id') = :kb_id "
                        "  AND json_extract(metadata,'$.deprecated_at') IS NOT NULL "
                        "  AND json_extract(metadata,'$.deprecated_at') < :cutoff"
                    ),
                    {"kb_id": kb_id, "cutoff": cutoff_iso},
                )
            ).all()

            for row in rows:
                meta_raw = getattr(row, "metadata", "{}")
                try:
                    meta = (
                        json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
                    )
                except (json.JSONDecodeError, TypeError):
                    meta = {}
                doc_id = meta.get("kb_doc_id", "")
                uri = meta.get("uri", "")
                if doc_id:
                    candidates.append((doc_id, uri))
    except Exception as e:
        logger.warning(f"[简单长期记忆] purge 查询失败: {e}")
        return {"purged": 0, "links_cleaned": 0}

    if not candidates:
        return {"purged": 0, "links_cleaned": 0}

    # 2. 按精确 kb_doc_id 逐条删除向量，跟踪成功集合
    succeeded_doc_ids: list[str] = []
    succeeded_uris: list[str] = []
    failed = 0
    for doc_id, uri in candidates:
        try:
            await vec_db.delete_documents(
                metadata_filters={"kb_doc_id": doc_id, "kb_id": kb_id}
            )
            succeeded_doc_ids.append(doc_id)
            if uri:
                succeeded_uris.append(uri)
        except Exception as e:
            failed += 1
            logger.debug(f"[简单长期记忆] purge 删除 {doc_id} 失败: {e}")

    # 向量删除全部失败时不继续下游清理
    if not succeeded_doc_ids:
        logger.warning("[简单长期记忆] purge 向量删除全部失败，跳过下游清理")
        return {"purged": 0, "links_cleaned": 0}

    # 3. 仅对成功删除向量的 doc_id 清理 KB 文档记录
    try:
        from astrbot.core.knowledge_base.models import KBDocument
        from sqlmodel import col, delete

        async with kb_helper.kb_db.get_db() as session:
            async with session.begin():
                stmt = delete(KBDocument).where(
                    col(KBDocument.doc_id).in_(succeeded_doc_ids)
                )
                await session.execute(stmt)
                await session.commit()
    except Exception as e:
        logger.warning(f"[简单长期记忆] purge KB 文档删除失败: {e}")

    # 4. 仅对成功删除的 URI 级联清理关联边
    links_cleaned = 0
    if link_manager and succeeded_uris:
        links_cleaned = await link_manager.delete_links_for_uris(succeeded_uris)

    # 5. 同步 KB 统计
    try:
        await kb_helper.kb_db.update_kb_stats(
            kb_id=kb_id,
            vec_db=vec_db,
        )
        await kb_helper.refresh_kb()
    except Exception as e:
        logger.debug(f"[简单长期记忆] purge 统计同步失败: {e}")

    purged = len(succeeded_doc_ids)
    if purged:
        logger.info(
            f"[简单长期记忆] 物理清理完成: {purged} 条记忆, "
            f"{links_cleaned} 条关联边" + (f", {failed} 条失败" if failed else "")
        )
    return {"purged": purged, "links_cleaned": links_cleaned}
