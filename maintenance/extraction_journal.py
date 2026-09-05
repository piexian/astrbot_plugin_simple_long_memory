"""持久保存提取提案，直到执行完成或可靠转入人工待审。"""

from __future__ import annotations

import copy
import uuid
from dataclasses import asdict
from typing import Any

from .agents.segmenter import ConversationBlock

JOURNAL_KV_KEY = "maintenance_extract_journal"


class ExtractionJournal:
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
        self, segment: dict[str, Any], curator: dict[str, Any], metrics: dict[str, Any]
    ) -> None:
        operations = copy.deepcopy(
            curator.get("create", []) + curator.get("update", [])
        )
        for op in operations:
            op["_extract_id"] = uuid.uuid4().hex
        outcomes = []
        for block, outcome in curator.get("outcomes", []):
            data = asdict(block)
            data["text"] = ""
            outcomes.append({"block": data, "key": block.key, "outcome": outcome})
        state = {
            "operations": operations,
            "completed": [],
            "outcomes": outcomes,
            "segment": {
                k: copy.deepcopy(v) for k, v in segment.items() if k != "blocks"
            },
            "metrics": metrics,
            "notes": curator.get("notes", ""),
        }
        if operations and (not self._get or not self._put):
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

    def outcomes(self) -> list[tuple[ConversationBlock, str]]:
        pending = self.pending
        outcomes = []
        for item in self.state.get("outcomes", []):
            outcome = item["outcome"]
            if any(
                op.get("_extract_block", item["key"]) == item["key"] for op in pending
            ):
                outcome = "skipped_budget"
            outcomes.append((ConversationBlock(**item["block"]), outcome))
        return outcomes

    async def clear_if_finished(self) -> None:
        if self.state and not self.pending:
            if self._put:
                await self._put(JOURNAL_KV_KEY, {})
            self.state = {}
