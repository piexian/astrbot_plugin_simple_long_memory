"""持久保存提取提案，直到执行完成或可靠转入人工待审。"""

from __future__ import annotations

import copy
import uuid
from typing import Any

JOURNAL_KV_KEY = "maintenance_extract_journal"


class ExtractionJournal:
    """跨周期保存提取操作与块到缓冲行的映射。"""

    def __init__(self, kv_get: Any, kv_put: Any) -> None:
        self._get = kv_get
        self._put = kv_put
        self.state: dict[str, Any] = {}

    async def load(self) -> bool:
        raw = await self._get(JOURNAL_KV_KEY, None) if self._get else None
        if raw and not isinstance(raw, dict):
            raise ValueError("invalid extraction journal")
        self.state = copy.deepcopy(raw or {})
        return bool(self.state)

    async def prepare(
        self,
        operations: list[dict[str, Any]],
        block_rows: dict[str, list[int]],
        metrics: dict[str, Any],
        notes: str,
    ) -> None:
        """保存本周期提取提案；操作的最终完成由 acknowledge 逐条登记。"""
        stored = copy.deepcopy(operations)
        for op in stored:
            op["_extract_id"] = uuid.uuid4().hex
        state = {
            "operations": stored,
            "completed": [],
            "block_rows": {str(key): list(rows) for key, rows in block_rows.items()},
            "metrics": metrics,
            "notes": notes,
        }
        if stored and (not self._get or not self._put):
            raise RuntimeError("extraction requires persistent KV storage")
        if self._put:
            await self._put(JOURNAL_KV_KEY, copy.deepcopy(state))
        self.state = state

    @property
    def pending(self) -> list[dict[str, Any]]:
        done = set(self.state.get("completed", []))
        return [
            copy.deepcopy(op)
            for op in self.state.get("operations", [])
            if op["_extract_id"] not in done
        ]

    async def acknowledge(self, op: dict[str, Any]) -> None:
        identity = op.get("_extract_id")
        if not identity or not any(
            item["_extract_id"] == identity for item in self.state.get("operations", [])
        ):
            return
        state = copy.deepcopy(self.state)
        if identity not in state["completed"]:
            state["completed"].append(identity)
            await self._put(JOURNAL_KV_KEY, copy.deepcopy(state))
            self.state = state

    def settled_block_rows(self) -> dict[str, list[int]]:
        """返回操作已全部完成的块及其缓冲行 id，供宿主标记 processed。"""
        pending_blocks = {op.get("_extract_block") for op in self.pending}
        return {
            str(key): list(rows)
            for key, rows in self.state.get("block_rows", {}).items()
            if key not in pending_blocks
        }

    async def clear_if_finished(self) -> None:
        if self.state and not self.pending:
            if self._put:
                await self._put(JOURNAL_KV_KEY, {})
            self.state = {}
