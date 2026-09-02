"""整理师（提取向）：对话块 → 记忆提取操作。

核心职责：
1. 对每个对话块检索同会话旧记忆做新老对照，避免重复创建
2. LLM 提取新记忆（create）或旧记忆更新建议（update）
3. 只输出结构化操作，游标提交与记忆写入均由宿主 runner 统一执行
"""

from __future__ import annotations

import json
from typing import Any

from astrbot.api import logger

from ..prompts import CURATOR_EXTRACTION_PROMPT, build_prompt

try:  # 正常插件加载（包内相对导入）
    from ...extraction_utils import validate_extracted_memories
    from ...memory_protocol import UMOInfo, build_session_id
    from ...prompts import sanitize_memory_content
except ImportError:  # 测试环境：仓库根作为顶层目录直接在 sys.path
    from extraction_utils import validate_extracted_memories
    from memory_protocol import UMOInfo, build_session_id
    from prompts import sanitize_memory_content


class CuratorAgent:
    """整理师（提取向）Agent。"""

    def __init__(
        self,
        context: Any,
        memory_mgr: Any,
        llm: Any,
        config: dict[str, Any],
    ) -> None:
        self._context = context
        self._memory_mgr = memory_mgr
        self._llm = llm
        self._config = config
        self._model_id = config.get("maintenance_curator_model_id", "")

    async def run(self, blocks: list[Any], max_llm_calls: int) -> dict[str, Any]:
        """逐个对话块提取记忆操作。

        Returns:
            {
                "create": [create_op, ...],
                "update": [update_op, ...],
                "notes": str,
                "blocks_processed": int,
                "blocks_nothing": int,
                "llm_calls": int,
                "outcomes": [(block, "created"|"updated"|"nothing"|"skipped_budget")],
            }
        """
        result: dict[str, Any] = {
            "create": [],
            "update": [],
            "notes": "",
            "blocks_processed": 0,
            "blocks_nothing": 0,
            "llm_calls": 0,
            "outcomes": [],
        }
        notes: list[str] = []
        for block in blocks:
            if (
                result["llm_calls"] >= max_llm_calls
                or getattr(self._llm, "remaining_calls", 1) <= 0
            ):
                result["outcomes"].append((block, "skipped_budget"))
                continue
            outcome = await self._process_block(block, result, notes)
            result["outcomes"].append((block, outcome))
        result["notes"] = "; ".join(notes)[:300]
        logger.debug(
            "[简单长期记忆] 整理师完成: blocks=%s, processed=%s, nothing=%s, "
            "create=%s, update=%s, llm_calls=%s",
            len(blocks),
            result["blocks_processed"],
            result["blocks_nothing"],
            len(result["create"]),
            len(result["update"]),
            result["llm_calls"],
        )
        return result

    async def _process_block(
        self, block: Any, result: dict[str, Any], notes: list[str]
    ) -> str:
        """处理单个对话块，返回 outcome。"""
        parsed_umo = UMOInfo.parse(block.umo)
        old_memories = await self._retrieve_old_memories(block, parsed_umo)
        old_uris = {m["uri"] for m in old_memories}
        existing_text = (
            "\n".join(f"[{m['uri']}] {m['excerpt']}" for m in old_memories) or "（无）"
        )
        system = build_prompt(
            CURATOR_EXTRACTION_PROMPT,
            {
                "platform_id": parsed_umo.platform_id,
                "session_type": parsed_umo.session_type,
                "session_id": parsed_umo.session_id,
                "existing_memories": existing_text,
                "conversation": block.text,
            },
        )
        result["llm_calls"] += 1
        parsed_json = await self._llm.chat_json(
            system, "只回 JSON。", model_id=self._model_id
        )
        result["blocks_processed"] += 1
        if parsed_json is None:
            # LLM 失败按无产出处理，游标照常推进（与自动提取路径的保守策略一致）
            logger.warning(
                "[简单长期记忆] 整理师 LLM 失败: conv=%s, ids=%s-%s",
                block.conv_key,
                block.first_id,
                block.last_id,
            )
            result["blocks_nothing"] += 1
            return "nothing"

        memories = validate_extracted_memories(
            parsed_json.get("memories"), parsed_umo.session_type
        )
        source_excerpt = " ".join(block.text.split())[:200]
        for mem in memories:
            result["create"].append(
                {
                    "type": "create",
                    "content": mem["content"],
                    "domain": mem["type"],
                    "scope": mem["scope"],
                    "subject": mem["subject"],
                    "subjects": mem["subjects"],
                    "entities": mem["entities"],
                    "topics": mem["topics"],
                    "disclosure": mem["disclosure"],
                    "importance": mem["importance"],
                    "umo": block.umo,
                    "source_excerpt": source_excerpt,
                    "truncated_block": block.truncated,
                }
            )
        updates = self._parse_updates(parsed_json.get("updates"), old_uris)
        result["update"].extend(updates)
        note = str(parsed_json.get("notes") or "").strip()
        if note:
            notes.append(note)

        if memories:
            outcome = "created"
        elif updates:
            outcome = "updated"
        else:
            outcome = "nothing"
            result["blocks_nothing"] += 1
        logger.debug(
            "[简单长期记忆] 整理师对话块完成: conv=%s, ids=%s-%s, outcome=%s, "
            "create=%s, update=%s",
            block.conv_key,
            block.first_id,
            block.last_id,
            outcome,
            len(memories),
            len(updates),
        )
        return outcome

    async def _retrieve_old_memories(
        self, block: Any, parsed_umo: Any
    ) -> list[dict[str, str]]:
        """检索同会话旧记忆做新老对照；检索失败返回空列表，不阻断提取。"""
        owner_session_id = build_session_id(
            parsed_umo.platform_id, parsed_umo.session_id
        )
        try:
            results = await self._memory_mgr.vec_db.retrieve(
                query=block.text[:500],
                k=5,
                rerank=False,
                metadata_filters={
                    "is_memory_record": True,
                    "deprecated": False,
                    "owner_session_id": owner_session_id,
                },
            )
        except Exception as e:
            logger.debug(
                "[简单长期记忆] 整理师旧记忆检索失败（按无旧记忆继续）: conv=%s, err=%s",
                block.conv_key,
                e,
            )
            return []
        old: list[dict[str, str]] = []
        for item in results or []:
            data = getattr(item, "data", None) or {}
            meta = data.get("metadata") or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            uri = str(meta.get("uri") or "")
            if not uri:
                continue
            content = str(meta.get("memory_content") or data.get("text") or "")
            old.append({"uri": uri, "excerpt": " ".join(content.split())[:80]})
        return old

    @staticmethod
    def _parse_updates(raw: Any, old_uris: set[str]) -> list[dict[str, Any]]:
        """校验 update 建议：uri 必须 ∈ 本块提供的旧记忆列表，防幻觉引用。"""
        if not isinstance(raw, list) or not old_uris:
            return []
        updates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            uri = str(item.get("uri") or "").strip()
            if uri not in old_uris or uri in seen:
                continue
            new_content = sanitize_memory_content(str(item.get("new_content") or ""))
            if not new_content:
                continue
            seen.add(uri)
            updates.append(
                {
                    "type": "update",
                    "uri": uri,
                    "new_content": new_content,
                    "reason": sanitize_memory_content(str(item.get("reason") or ""))[
                        :200
                    ],
                }
            )
        return updates
