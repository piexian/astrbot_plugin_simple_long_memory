"""分段员测试：间隙预切、滚动闭环、carry 跨周期、毒块跳过、预算与上限。"""

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from maintenance.agents.segmenter import SegmenterAgent
from maintenance.llm import MaintenanceLLM


def _row(
    rid,
    text,
    ts,
    sender_id="u1",
    sender_name="甲",
    platform="qq",
    umo="g1",
    content=None,
):
    if content is None:
        content = {"type": "user", "message": [{"type": "plain", "text": text}]}
    return SimpleNamespace(
        id=rid,
        platform_id=platform,
        user_id=umo,
        sender_id=sender_id,
        sender_name=sender_name,
        content=content,
        created_at=ts,
    )


def _config(**over):
    cfg = {
        "maintenance_segment_time_gap_minutes": 30,
        "extraction_min_content_length": 100,
        "maintenance_segment_max_chars": 8000,
        "maintenance_segment_max_extensions": 3,
        "maintenance_extract_max_blocks_per_cycle": 20,
    }
    cfg.update(over)
    return cfg


class _FakeLLM:
    """预设 chat_json 返回序列的 fake llm。"""

    def __init__(self, responses=None):
        self._responses = list(responses or [])
        self.calls = 0

    @property
    def remaining_calls(self):
        return 1_000_000

    async def chat_json(self, system_prompt, user_prompt, model_id=""):
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        return {"complete": True, "reason": "闭环"}


class _KV:
    def __init__(self, initial=None):
        self.store = dict(initial or {})

    async def get_kv_data(self, key, default=None):
        return self.store.get(key, default)


class _StubSegmenter(SegmenterAgent):
    """覆写数据访问，喂内存数据。"""

    def __init__(self, rows, conversations, llm, config, kv):
        super().__init__(
            context=SimpleNamespace(
                message_history_manager=SimpleNamespace(db=object())
            ),
            llm=llm,
            config=config,
            kv_get=kv.get_kv_data,
            kv_put=None,
        )
        self._rows = rows
        self._conversations = conversations
        self.seen_since = None

    async def _list_active_conversations(self, since):
        self.seen_since = since
        return list(self._conversations)

    async def _fetch_rows(self, platform_id, user_id, after_id, limit):
        page = [
            r
            for r in self._rows
            if r.platform_id == platform_id and r.user_id == user_id and r.id > after_id
        ]
        page.sort(key=lambda r: r.id)
        return page[:limit]


INCOMPLETE = {"complete": False, "reason": "还有悬而未决"}


class SegmenterTests(unittest.IsolatedAsyncioTestCase):
    def _agent(self, rows, conversations, llm, config=None, kv=None):
        return _StubSegmenter(
            rows, conversations, llm, config or _config(), kv or _KV()
        )

    async def test_gap_pre_split(self):
        now = datetime.now(timezone.utc)
        rows = [
            _row(1, "聊" * 120, now - timedelta(hours=3)),
            _row(2, "聊" * 120, now - timedelta(hours=3) + timedelta(minutes=5)),
            # 间隙 55 分钟 > 30 → 切成第二块
            _row(3, "聊" * 120, now - timedelta(hours=2)),
        ]
        llm = _FakeLLM()
        agent = self._agent(rows, [("qq", "g1")], llm)
        result = await agent.collect_blocks(max_llm_calls=10)

        self.assertEqual(len(result["blocks"]), 2)
        b1, b2 = result["blocks"]
        self.assertEqual((b1.first_id, b1.last_id), (1, 2))
        self.assertEqual((b2.first_id, b2.last_id), (3, 3))
        self.assertEqual(b1.conv_key, "qq:g1")
        self.assertEqual(b1.umo, "g1")
        self.assertFalse(b1.truncated)
        self.assertEqual(result["cursor_updates"], {"qq:g1": 3})
        self.assertEqual(result["carry_updates"], {})
        self.assertEqual(llm.calls, 2)
        self.assertEqual(result["stats"]["blocks_emitted"], 2)
        self.assertFalse(result["stats"]["budget_exhausted"])

    async def test_incomplete_rolls_into_next_segment_then_complete(self):
        now = datetime.now(timezone.utc)
        rows = [
            _row(1, "聊" * 120, now - timedelta(hours=3)),
            _row(2, "聊" * 120, now - timedelta(hours=2)),
        ]
        llm = _FakeLLM(responses=[INCOMPLETE, {"complete": True, "reason": "闭环"}])
        agent = self._agent(rows, [("qq", "g1")], llm)
        result = await agent.collect_blocks(max_llm_calls=10)

        self.assertEqual(len(result["blocks"]), 1)
        block = result["blocks"][0]
        self.assertEqual((block.first_id, block.last_id), (1, 2))
        self.assertEqual(block.message_count, 2)
        self.assertIn("聊", block.text)
        self.assertEqual(result["cursor_updates"], {"qq:g1": 2})
        self.assertEqual(llm.calls, 2)

    async def test_extension_limit_carries(self):
        now = datetime.now(timezone.utc)
        rows = [
            _row(1, "聊" * 120, now - timedelta(hours=4)),
            _row(2, "聊" * 120, now - timedelta(hours=3)),
            _row(3, "聊" * 120, now - timedelta(hours=2)),
        ]
        llm = _FakeLLM(responses=[INCOMPLETE, INCOMPLETE])
        agent = self._agent(
            rows,
            [("qq", "g1")],
            llm,
            config=_config(maintenance_segment_max_extensions=1),
        )
        result = await agent.collect_blocks(max_llm_calls=10)

        # 并入一次后撞滚动上限 → 挂起，游标不动
        self.assertEqual(result["blocks"], [])
        self.assertEqual(result["cursor_updates"], {})
        self.assertEqual(
            result["carry_updates"],
            {"qq:g1": {"start_id": 1, "cycles": 1, "fail_count": 0}},
        )
        self.assertEqual(llm.calls, 2)
        self.assertEqual(result["stats"]["blocks_carried"], 1)

    async def test_char_limit_carry_then_force_truncate_across_cycles(self):
        now = datetime.now(timezone.utc)
        rows = [
            _row(1, "聊" * 80, now - timedelta(hours=5)),
            _row(2, "聊" * 80, now - timedelta(hours=4)),
        ]
        # 每段约 99 字符（含装饰），并入后超 max_chars=120
        cfg = _config(
            extraction_min_content_length=10, maintenance_segment_max_chars=120
        )
        convs = [("qq", "g1")]

        # 周期 1：撞字符上限 → carry cycles=1，游标不动
        llm = _FakeLLM(responses=[INCOMPLETE])
        kv = _KV()
        r1 = await self._agent(rows, convs, llm, cfg, kv).collect_blocks(10)
        self.assertEqual(r1["blocks"], [])
        self.assertEqual(r1["cursor_updates"], {})
        self.assertEqual(
            r1["carry_updates"],
            {"qq:g1": {"start_id": 1, "cycles": 1, "fail_count": 0}},
        )

        # 周期 2：cycles=1 → 仍挂起，cycles=2
        kv = _KV({"maintenance_extract_carry": r1["carry_updates"]})
        r2 = await self._agent(
            rows, convs, _FakeLLM(responses=[INCOMPLETE]), cfg, kv
        ).collect_blocks(10)
        self.assertEqual(r2["blocks"], [])
        self.assertEqual(
            r2["carry_updates"],
            {"qq:g1": {"start_id": 1, "cycles": 2, "fail_count": 0}},
        )

        # 周期 3：cycles 已 >= 2 → 强制截断产出第一块，随后第二块自然闭环
        kv = _KV({"maintenance_extract_carry": r2["carry_updates"]})
        llm3 = _FakeLLM(responses=[INCOMPLETE, INCOMPLETE])
        r3 = await self._agent(rows, convs, llm3, cfg, kv).collect_blocks(10)
        self.assertEqual(len(r3["blocks"]), 2)
        forced = r3["blocks"][0]
        self.assertTrue(forced.truncated)
        self.assertEqual((forced.first_id, forced.last_id), (1, 1))
        self.assertEqual(forced.message_count, 1)
        self.assertFalse(r3["blocks"][1].truncated)
        self.assertEqual(r3["cursor_updates"], {"qq:g1": 2})
        self.assertEqual(r3["carry_updates"], {"qq:g1": None})
        self.assertEqual(r3["stats"]["blocks_truncated"], 1)
        self.assertEqual(r3["stats"]["blocks_emitted"], 1)

    async def test_no_more_messages_beyond_gap_natural_close(self):
        now = datetime.now(timezone.utc)
        rows = [_row(1, "聊" * 120, now - timedelta(hours=3))]
        llm = _FakeLLM(responses=[INCOMPLETE])
        agent = self._agent(rows, [("qq", "g1")], llm)
        result = await agent.collect_blocks(max_llm_calls=10)

        # 块尾距今 3h 已超 gap → 视为自然结束按 complete 处理
        self.assertEqual(len(result["blocks"]), 1)
        self.assertFalse(result["blocks"][0].truncated)
        self.assertEqual(result["cursor_updates"], {"qq:g1": 1})
        self.assertEqual(result["carry_updates"], {})
        self.assertEqual(result["stats"]["blocks_emitted"], 1)

    async def test_no_more_messages_within_gap_carries(self):
        now = datetime.now(timezone.utc)
        rows = [_row(1, "聊" * 120, now - timedelta(minutes=5))]
        llm = _FakeLLM(responses=[INCOMPLETE])
        agent = self._agent(rows, [("qq", "g1")], llm)
        result = await agent.collect_blocks(max_llm_calls=10)

        # 块尾距今 5 分钟未超 gap → 游标停块首 + carry
        self.assertEqual(result["blocks"], [])
        self.assertEqual(result["cursor_updates"], {})
        self.assertEqual(
            result["carry_updates"],
            {"qq:g1": {"start_id": 1, "cycles": 1, "fail_count": 0}},
        )
        self.assertEqual(result["stats"]["blocks_carried"], 1)

    async def test_llm_failure_twice_skips_block(self):
        now = datetime.now(timezone.utc)
        rows = [_row(1, "聊" * 120, now - timedelta(hours=3))]
        convs = [("qq", "g1")]

        # 周期 1：LLM 失败 → fail_count=1，游标停块首
        kv = _KV()
        r1 = await self._agent(
            rows, convs, _FakeLLM(responses=[None]), kv=kv
        ).collect_blocks(10)
        self.assertEqual(r1["blocks"], [])
        self.assertEqual(r1["cursor_updates"], {})
        self.assertEqual(
            r1["carry_updates"],
            {"qq:g1": {"start_id": 1, "cycles": 0, "fail_count": 1}},
        )

        # 周期 2：再失败 → 跳过该块，游标推进，carry 清除
        kv = _KV({"maintenance_extract_carry": r1["carry_updates"]})
        r2 = await self._agent(
            rows, convs, _FakeLLM(responses=[None]), kv=kv
        ).collect_blocks(10)
        self.assertEqual(r2["blocks"], [])
        self.assertEqual(r2["cursor_updates"], {"qq:g1": 1})
        self.assertEqual(r2["carry_updates"], {"qq:g1": None})
        self.assertEqual(r2["stats"]["poison_skipped"], 1)

    async def test_budget_exhausted_stops_and_keeps_cursor(self):
        now = datetime.now(timezone.utc)
        rows = [
            _row(1, "聊" * 120, now - timedelta(hours=3), umo="g1"),
            _row(2, "聊" * 120, now - timedelta(hours=2), umo="g1"),
            _row(3, "聊" * 120, now - timedelta(hours=3), umo="g2"),
        ]
        llm = _FakeLLM()  # 默认 complete
        agent = self._agent(rows, [("qq", "g1"), ("qq", "g2")], llm)
        result = await agent.collect_blocks(max_llm_calls=1)

        # g1 第一块判定完成后预算耗尽：第二块与 g2 不再处理，游标不动
        self.assertTrue(result["stats"]["budget_exhausted"])
        self.assertEqual(len(result["blocks"]), 1)
        self.assertEqual(result["blocks"][0].last_id, 1)
        self.assertEqual(result["cursor_updates"], {"qq:g1": 1})
        self.assertEqual(result["carry_updates"], {})
        self.assertEqual(result["stats"]["llm_calls"], 1)
        self.assertEqual(result["stats"]["conversations_scanned"], 1)

    async def test_short_blocks_skipped_without_llm(self):
        now = datetime.now(timezone.utc)
        rows = [
            _row(1, "嗯", now - timedelta(hours=3)),
            _row(2, "好的", now - timedelta(hours=3) + timedelta(minutes=1)),
            # 间隙 59 分钟 → 第二块为长对话
            _row(3, "聊" * 60, now - timedelta(hours=2)),
            _row(4, "聊" * 60, now - timedelta(hours=2) + timedelta(minutes=2)),
        ]
        llm = _FakeLLM()
        agent = self._agent(rows, [("qq", "g1")], llm)
        result = await agent.collect_blocks(max_llm_calls=10)

        # 短闲聊块被预过滤，不消耗 LLM，游标照常推进
        self.assertEqual(llm.calls, 1)
        self.assertEqual(len(result["blocks"]), 1)
        self.assertEqual(
            (result["blocks"][0].first_id, result["blocks"][0].last_id), (3, 4)
        )
        self.assertEqual(result["cursor_updates"], {"qq:g1": 4})
        self.assertEqual(result["stats"]["blocks_skipped_short"], 1)

    async def test_block_cap_stops_emission(self):
        now = datetime.now(timezone.utc)
        rows = [
            _row(1, "聊" * 120, now - timedelta(hours=5), umo="g1"),
            _row(2, "聊" * 120, now - timedelta(hours=4), umo="g1"),
            _row(3, "聊" * 120, now - timedelta(hours=3), umo="g2"),
        ]
        agent = self._agent(
            rows,
            [("qq", "g1"), ("qq", "g2")],
            _FakeLLM(),
            config=_config(maintenance_extract_max_blocks_per_cycle=1),
        )
        result = await agent.collect_blocks(max_llm_calls=10)

        # 产出达上限：后续块不产出、游标不推进
        self.assertEqual(len(result["blocks"]), 1)
        self.assertEqual(result["cursor_updates"], {"qq:g1": 1})
        self.assertTrue(result["stats"]["block_cap_reached"])

    async def test_legacy_int_cursor_and_since_from_earliest_cursor(self):
        now = datetime.now(timezone.utc)
        rows = [_row(i, "聊" * 60, now - timedelta(hours=50)) for i in range(4, 8)]
        old_ts = now - timedelta(hours=48)
        kv = _KV(
            {
                "maintenance_extract_cursors": {
                    "qq:g1": 5,  # 旧格式纯 int
                    "qq:g9": {"id": 10, "ts": old_ts.isoformat()},
                }
            }
        )
        agent = self._agent(rows, [("qq", "g1")], _FakeLLM(), kv=kv)
        result = await agent.collect_blocks(max_llm_calls=10)

        # 旧格式游标=5 → 只处理 id>5
        self.assertEqual(len(result["blocks"]), 1)
        block = result["blocks"][0]
        self.assertEqual((block.first_id, block.last_id), (6, 7))
        self.assertEqual(result["cursor_updates"], {"qq:g1": 7})
        # since 取最早游标 ts（48h 前）而非默认 24h
        self.assertIsNotNone(agent.seen_since)
        self.assertLess(
            abs((agent.seen_since - old_ts).total_seconds()),
            1,
        )

    async def test_flatten_format_at_bot_empty_and_chain_key(self):
        now = datetime.now(timezone.utc)
        rows = [
            _row(
                1,
                "",
                now - timedelta(hours=3),
                content={
                    "type": "user",
                    "message": [
                        {"type": "at", "user_id": "u2", "name": "乙"},
                        {"type": "plain", "text": " 在吗"},
                        {"type": "image"},  # 未知类型忽略
                    ],
                },
            ),
            # chain 键兼容 + bot 发送者
            _row(
                2,
                "",
                now - timedelta(hours=3) + timedelta(minutes=1),
                sender_id="bot",
                sender_name=None,
                content={"chain": [{"type": "plain", "text": "在的"}]},
            ),
            # 空文本行：不入文本但游标越过
            _row(3, "", now - timedelta(hours=3) + timedelta(minutes=2), content={}),
            _row(4, "聊" * 40, now - timedelta(hours=3) + timedelta(minutes=3)),
        ]
        cfg = _config(extraction_min_content_length=10)
        agent = self._agent(rows, [("qq", "g1")], _FakeLLM(), config=cfg)
        result = await agent.collect_blocks(max_llm_calls=10)

        self.assertEqual(len(result["blocks"]), 1)
        block = result["blocks"][0]
        self.assertEqual(block.last_id, 4)  # 空行 id=3 被游标越过
        self.assertEqual(block.message_count, 3)
        ts = (now - timedelta(hours=3)).strftime("%m-%d %H:%M")
        self.assertEqual(block.text.splitlines()[0], f"[{ts}] 甲: @乙 在吗")
        self.assertIn("] bot: 在的", block.text)
        self.assertEqual(block.char_count, len(block.text))
        self.assertEqual(result["cursor_updates"], {"qq:g1": 4})

    async def test_no_message_history_manager_returns_empty(self):
        llm = _FakeLLM()
        agent = SegmenterAgent(
            context=SimpleNamespace(),  # 无 message_history_manager
            llm=llm,
            config=_config(),
            kv_get=_KV().get_kv_data,
            kv_put=None,
        )
        result = await agent.collect_blocks(max_llm_calls=10)
        self.assertEqual(result["blocks"], [])
        self.assertEqual(result["cursor_updates"], {})
        self.assertEqual(result["carry_updates"], {})
        self.assertEqual(result["stats"]["conversations_scanned"], 0)
        self.assertEqual(llm.calls, 0)


class ChatJsonTests(unittest.IsolatedAsyncioTestCase):
    """MaintenanceLLM.chat_json：解析成功返回 dict，失败/None 均返回 None。"""

    def _llm(self):
        return MaintenanceLLM(None, cache_enabled=False)

    async def test_parses_wrapped_json(self):
        llm = self._llm()

        async def ok(system, user, model_id=""):
            return '前缀\n```json\n{"complete": true, "reason": "ok"}\n```\n后缀'

        llm._chat = ok
        self.assertEqual(
            await llm.chat_json("s", "u"), {"complete": True, "reason": "ok"}
        )

    async def test_returns_none_on_unparseable_and_on_chat_failure(self):
        llm = self._llm()

        async def bad(system, user, model_id=""):
            return "not json at all"

        llm._chat = bad
        self.assertIsNone(await llm.chat_json("s", "u"))

        async def none(system, user, model_id=""):
            return None

        llm._chat = none
        self.assertIsNone(await llm.chat_json("s", "u"))


if __name__ == "__main__":
    unittest.main()
