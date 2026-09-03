"""分段员 ConversationV2 兜底测试：源选择、切段闭环、锚点游标、carry、runner 提交。"""

import hashlib
import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from test_maintenance_extract import (
    _FakeCurator,
    _FakeSegmenter,
    _KVLog,
    _RunnerFakeLLM,
    _seg_result,
)
from test_maintenance_segmenter import (
    _KV,
    INCOMPLETE,
    _config,
    _FakeLLM,
    _row,
    _StubSegmenter,
)

import maintenance.runner as runner_module
from maintenance.agents.segmenter import (
    CURSOR_KV_KEY,
    ConversationBlock,
    SegmenterAgent,
)
from maintenance.runner import MaintenanceRunner

_MSG_BODY = "聊" * 60


def _msgs(n):
    """构造 n 条 OpenAI 格式消息，role 交替 user/assistant。"""
    out = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        out.append({"role": role, "content": f"{_MSG_BODY}{i}"})
    return out


def _msg_anchor(i):
    """与 SegmenterAgent._anchor 同算法，定位 _msgs 第 i 条的锚点。"""
    role = "user" if i % 2 == 0 else "assistant"
    return hashlib.sha256(f"{role}:{_MSG_BODY}{i}".encode()).hexdigest()[:16]


def _conv2_conv(platform_id, user_id, messages, updated_at, cid="cid-1"):
    """构造 v1 Conversation 形状（history 为 JSON 字符串，时间戳为 epoch 秒）。"""
    return SimpleNamespace(
        platform_id=platform_id,
        user_id=user_id,
        cid=cid,
        history=json.dumps(messages, ensure_ascii=False),
        created_at=max(0, updated_at - 3600),
        updated_at=updated_at,
    )


def _c2cfg(**over):
    return _config(
        extraction_min_content_length=10,
        maintenance_extract_conv2_chunk_messages=4,
        **over,
    )


def _conv2_cursor(anchor, idx):
    return {"source": "conv2", "anchor": anchor, "idx": idx, "ts": ""}


class _FakeConversationManager:
    def __init__(self, convs):
        self._convs = convs
        self.calls = 0

    async def get_conversations(self):
        self.calls += 1
        return list(self._convs)


class _Conv2Segmenter(_StubSegmenter):
    """_StubSegmenter 基础上挂内存 conversation_manager。"""

    def __init__(self, rows, conversations, conv2_convs, llm, config, kv):
        super().__init__(rows, conversations, llm, config, kv)
        self._context.conversation_manager = _FakeConversationManager(conv2_convs)


def _agent(rows, pmh_convs, conv2_convs, llm, config=None, kv=None):
    return _Conv2Segmenter(
        rows, pmh_convs, conv2_convs, llm, config or _c2cfg(), kv or _KV()
    )


def _now_epoch():
    return int(datetime.now(timezone.utc).timestamp())


class Conv2SourceSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_pmh_covered_conversation_skips_conv2(self):
        now = datetime.now(timezone.utc)
        rows = [_row(1, "聊" * 120, now)]
        conv2 = [_conv2_conv("qq", "g1", _msgs(4), _now_epoch())]
        llm = _FakeLLM()
        agent = _agent(rows, [("qq", "g1")], conv2, llm)
        result = await agent.collect_blocks(10)

        # pmh 窗口内已有该会话 → conv2 不重复提取
        self.assertEqual(len(result["blocks"]), 1)
        self.assertEqual(result["blocks"][0].source, "pmh")
        self.assertEqual(result["blocks"][0].end_anchor, "")
        self.assertEqual(result["stats"]["conv2_conversations"], 0)
        self.assertEqual(result["stats"]["conv2_blocks_emitted"], 0)
        self.assertEqual(result["cursor_updates"], {"qq:g1": 1})

    async def test_conv2_disabled_closes_path(self):
        conv2 = [_conv2_conv("qq", "u1", _msgs(4), _now_epoch())]
        llm = _FakeLLM()
        agent = _agent(
            [], [], conv2, llm, config=_c2cfg(maintenance_extract_conv2_enabled=False)
        )
        result = await agent.collect_blocks(10)

        self.assertEqual(result["blocks"], [])
        self.assertEqual(result["stats"]["conv2_conversations"], 0)
        self.assertEqual(agent._context.conversation_manager.calls, 0)
        self.assertEqual(llm.calls, 0)

    async def test_conv2_filters_stale_empty_user_and_zero_ts(self):
        now_epoch = _now_epoch()
        conv2 = [
            _conv2_conv("qq", "old", _msgs(4), now_epoch - 3 * 86400),  # 超出窗口
            _conv2_conv("qq", "", _msgs(4), now_epoch),  # user_id 为空
            _conv2_conv("qq", "zero", _msgs(4), 0),  # updated_at 为 0 无法判断新鲜度
            _conv2_conv("qq", "u1", _msgs(4), now_epoch),
        ]
        agent = _agent([], [], conv2, _FakeLLM())
        result = await agent.collect_blocks(10)

        self.assertEqual(result["stats"]["conv2_conversations"], 1)
        self.assertEqual(len(result["blocks"]), 1)
        self.assertEqual(result["blocks"][0].conv_key, "qq:u1")

    async def test_conv2_works_without_message_history_db(self):
        conv2 = [_conv2_conv("qq", "u1", _msgs(3), _now_epoch())]
        agent = SegmenterAgent(
            context=SimpleNamespace(  # 无 message_history_manager
                conversation_manager=_FakeConversationManager(conv2)
            ),
            llm=_FakeLLM(),
            config=_c2cfg(),
            kv_get=_KV().get_kv_data,
            kv_put=None,
        )
        result = await agent.collect_blocks(10)

        self.assertEqual(result["stats"]["conversations_scanned"], 0)
        self.assertEqual(result["stats"]["conv2_conversations"], 1)
        self.assertEqual(len(result["blocks"]), 1)
        self.assertEqual(result["blocks"][0].source, "conv2")


class Conv2SegmentationTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_seen_only_latest_chunk(self):
        conv2 = [_conv2_conv("qq", "u1", _msgs(10), _now_epoch())]
        llm = _FakeLLM()
        agent = _agent([], [], conv2, llm)
        result = await agent.collect_blocks(10)

        # 首见会话只处理最新一段，防首日积压全量倾倒
        self.assertEqual(len(result["blocks"]), 1)
        self.assertEqual(llm.calls, 1)
        block = result["blocks"][0]
        self.assertEqual(block.source, "conv2")
        self.assertEqual((block.first_id, block.last_id), (6, 9))
        self.assertEqual(block.message_count, 4)
        self.assertEqual(block.end_anchor, _msg_anchor(9))
        # 行格式不带时间戳前缀
        self.assertTrue(block.text.splitlines()[0].startswith("user: "))
        self.assertIn("\nbot: ", block.text)
        self.assertEqual(result["stats"]["conv2_blocks_emitted"], 1)
        self.assertEqual(result["stats"]["blocks_emitted"], 0)
        # 游标为内容锚点 + 下标
        cur = result["cursor_updates"]["qq:u1"]
        self.assertEqual(cur, {"source": "conv2", "anchor": _msg_anchor(9), "idx": 9})

    async def test_anchor_resume_chunks_and_close(self):
        kv = _KV({CURSOR_KV_KEY: {"qq:u1": _conv2_cursor(_msg_anchor(1), 1)}})
        conv2 = [_conv2_conv("qq", "u1", _msgs(7), _now_epoch())]
        llm = _FakeLLM()
        agent = _agent([], [], conv2, llm, kv=kv)
        result = await agent.collect_blocks(10)

        # 锚点命中消息 1 → 处理 2..6，chunk=4 切成 (2-5) 与 (6) 两段，全部闭环
        self.assertEqual(len(result["blocks"]), 2)
        self.assertEqual(
            [(b.first_id, b.last_id) for b in result["blocks"]], [(2, 5), (6, 6)]
        )
        self.assertEqual(result["blocks"][0].end_anchor, _msg_anchor(5))
        self.assertEqual(result["blocks"][1].end_anchor, _msg_anchor(6))
        self.assertEqual(
            result["cursor_updates"]["qq:u1"],
            {"source": "conv2", "anchor": _msg_anchor(6), "idx": 6},
        )
        self.assertEqual(llm.calls, 2)
        self.assertEqual(result["stats"]["conv2_blocks_emitted"], 2)
        self.assertEqual(result["stats"]["anchor_miss"], 0)

    async def test_anchor_resume_incomplete_rolls_into_next_chunk(self):
        kv = _KV({CURSOR_KV_KEY: {"qq:u1": _conv2_cursor(_msg_anchor(0), 0)}})
        conv2 = [_conv2_conv("qq", "u1", _msgs(6), _now_epoch())]
        llm = _FakeLLM(responses=[INCOMPLETE, {"complete": True, "reason": "闭环"}])
        agent = _agent([], [], conv2, llm, kv=kv)
        result = await agent.collect_blocks(10)

        # 第一段 (1-4) 判不完整 → 滚动并入第二段 (5) 后闭环
        self.assertEqual(len(result["blocks"]), 1)
        block = result["blocks"][0]
        self.assertEqual((block.first_id, block.last_id), (1, 5))
        self.assertEqual(block.message_count, 5)
        self.assertEqual(block.end_anchor, _msg_anchor(5))
        self.assertEqual(llm.calls, 2)

    async def test_anchor_miss_only_latest_chunk_and_reanchor(self):
        kv = _KV({CURSOR_KV_KEY: {"qq:u1": _conv2_cursor("f" * 16, 30)}})
        conv2 = [_conv2_conv("qq", "u1", _msgs(10), _now_epoch())]
        agent = _agent([], [], conv2, _FakeLLM(), kv=kv)
        result = await agent.collect_blocks(10)

        # 锚点丢失 → 保守只处理最新一段并重锚
        self.assertEqual(result["stats"]["anchor_miss"], 1)
        self.assertEqual(len(result["blocks"]), 1)
        block = result["blocks"][0]
        self.assertEqual((block.first_id, block.last_id), (6, 9))
        self.assertEqual(
            result["cursor_updates"]["qq:u1"],
            {"source": "conv2", "anchor": _msg_anchor(9), "idx": 9},
        )

    async def test_anchor_resume_no_new_messages_noop(self):
        kv = _KV({CURSOR_KV_KEY: {"qq:u1": _conv2_cursor(_msg_anchor(9), 9)}})
        conv2 = [_conv2_conv("qq", "u1", _msgs(10), _now_epoch())]
        llm = _FakeLLM()
        agent = _agent([], [], conv2, llm, kv=kv)
        result = await agent.collect_blocks(10)

        # 锚点已在末条 → 无新消息，不消耗 LLM，游标不动
        self.assertEqual(result["blocks"], [])
        self.assertEqual(result["cursor_updates"], {})
        self.assertEqual(result["carry_updates"], {})
        self.assertEqual(llm.calls, 0)


class Conv2CarryAndPoisonTests(unittest.IsolatedAsyncioTestCase):
    async def test_conv2_carry_format_and_cycle_inherit(self):
        # 块尾（会话 updated_at）在间隙内 + 判不完整 → carry 挂起
        conv2 = [_conv2_conv("qq", "u1", _msgs(3), _now_epoch())]
        llm = _FakeLLM(responses=[INCOMPLETE])
        agent = _agent([], [], conv2, llm)
        r1 = await agent.collect_blocks(10)

        self.assertEqual(r1["blocks"], [])
        self.assertEqual(r1["cursor_updates"], {})
        self.assertEqual(
            r1["carry_updates"],
            {
                "qq:u1": {
                    "start_id": 0,
                    "cycles": 1,
                    "fail_count": 0,
                    "start_anchor": _msg_anchor(0),
                }
            },
        )

        # 下周期内容未变：首块锚点匹配 start_anchor → 继承 cycles
        kv2 = _KV({"maintenance_extract_carry": r1["carry_updates"]})
        r2 = await _agent(
            [], [], conv2, _FakeLLM(responses=[INCOMPLETE]), kv=kv2
        ).collect_blocks(10)
        self.assertEqual(
            r2["carry_updates"],
            {
                "qq:u1": {
                    "start_id": 0,
                    "cycles": 2,
                    "fail_count": 0,
                    "start_anchor": _msg_anchor(0),
                }
            },
        )

    async def test_conv2_stale_carry_dropped_and_cleared(self):
        kv = _KV(
            {
                "maintenance_extract_carry": {
                    "qq:u1": {
                        "start_id": 0,
                        "cycles": 2,
                        "fail_count": 0,
                        "start_anchor": "d" * 16,  # 与候选块首条锚点错配
                    }
                }
            }
        )
        conv2 = [_conv2_conv("qq", "u1", _msgs(3), _now_epoch())]
        agent = _agent([], [], conv2, _FakeLLM(), kv=kv)
        result = await agent.collect_blocks(10)

        # 过期 carry 丢弃清除，cycles 不继承（否则立刻触发强制截断）
        self.assertEqual(result["carry_updates"], {"qq:u1": None})
        self.assertEqual(len(result["blocks"]), 1)
        self.assertFalse(result["blocks"][0].truncated)

    async def test_conv2_poison_skip_advances_anchor_cursor(self):
        conv2 = [_conv2_conv("qq", "u1", _msgs(3), _now_epoch())]

        # 周期 1：LLM 失败 → fail_count=1 挂起，游标不动
        kv = _KV()
        r1 = await _agent(
            [], [], conv2, _FakeLLM(responses=[None]), kv=kv
        ).collect_blocks(10)
        self.assertEqual(r1["blocks"], [])
        self.assertEqual(r1["cursor_updates"], {})
        self.assertEqual(
            r1["carry_updates"]["qq:u1"]["fail_count"],
            1,
        )
        self.assertEqual(r1["carry_updates"]["qq:u1"]["start_anchor"], _msg_anchor(0))

        # 周期 2：再失败 → 毒块跳过，锚点游标推进到块末，carry 清除
        kv = _KV({"maintenance_extract_carry": r1["carry_updates"]})
        r2 = await _agent(
            [], [], conv2, _FakeLLM(responses=[None]), kv=kv
        ).collect_blocks(10)
        self.assertEqual(r2["stats"]["poison_skipped"], 1)
        self.assertEqual(r2["carry_updates"], {"qq:u1": None})
        self.assertEqual(
            r2["cursor_updates"]["qq:u1"],
            {"source": "conv2", "anchor": _msg_anchor(2), "idx": 2},
        )


class Conv2FlattenAndKvTests(unittest.TestCase):
    def _agent(self):
        return SegmenterAgent(
            context=SimpleNamespace(),
            llm=_FakeLLM(),
            config=_c2cfg(),
            kv_get=None,
            kv_put=None,
        )

    def test_flatten_roles_and_content_shapes(self):
        history = [
            {"role": "system", "content": "忽略我"},
            {"role": "user", "content": "  你好  "},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "你"},
                    {"type": "image", "url": "x"},
                    {"type": "text", "text": "好"},
                ],
            },
            {"role": "user", "content": "   "},  # 空文本跳过
            {"role": "tool", "content": "忽略我"},
            {"role": "user"},  # 缺 content 跳过
        ]
        items = self._agent()._flatten_conv2_history(json.dumps(history), "k")
        self.assertEqual(
            [(r, s, t) for r, s, t, _ in items],
            [
                ("user", "user", "你好"),
                ("assistant", "bot", "你好"),
            ],
        )
        self.assertEqual(items[0][3], SegmenterAgent._anchor("user", "你好"))

    def test_flatten_bad_history_returns_empty(self):
        agent = self._agent()
        self.assertEqual(agent._flatten_conv2_history("not json", "k"), [])
        self.assertEqual(agent._flatten_conv2_history('{"a": 1}', "k"), [])
        self.assertEqual(agent._flatten_conv2_history(None, "k"), [])
        self.assertEqual(agent._flatten_conv2_history("", "k"), [])

    def test_normalize_cursors_keeps_conv2_schema(self):
        out = SegmenterAgent._normalize_cursors(
            {
                "a": {"source": "conv2", "anchor": "abc", "idx": 7, "ts": "t1"},
                "b": {"id": 3, "ts": "t2"},
                "c": 5,  # 旧格式纯 int
                "d": {"source": "conv2", "anchor": "abc", "idx": "bad"},
            }
        )
        self.assertEqual(
            out["a"], {"source": "conv2", "anchor": "abc", "idx": 7, "ts": "t1"}
        )
        self.assertEqual(out["b"], {"id": 3, "ts": "t2"})
        self.assertEqual(out["c"], {"id": 5, "ts": ""})
        self.assertNotIn("d", out)

    def test_normalize_carries_keeps_start_anchor(self):
        out = SegmenterAgent._normalize_carries(
            {
                "a": {
                    "start_id": 0,
                    "cycles": 2,
                    "fail_count": 0,
                    "start_anchor": "abc",
                },
                "b": {"start_id": 2, "cycles": 0, "fail_count": 1},
            }
        )
        self.assertEqual(out["a"]["start_anchor"], "abc")
        self.assertNotIn("start_anchor", out["b"])


class RunnerConv2CommitTests(unittest.IsolatedAsyncioTestCase):
    def _conv2_block(self, conv_key, last_id, end_anchor):
        return ConversationBlock(
            conv_key=conv_key,
            platform_id="qq",
            umo=conv_key.split(":", 1)[1],
            text="兜底对话内容",
            message_count=3,
            char_count=6,
            truncated=False,
            first_id=0,
            last_id=last_id,
            source="conv2",
            end_anchor=end_anchor,
        )

    async def test_commit_conv2_cursor_per_block_and_segmenter_updates(self):
        b1 = self._conv2_block("qq:p1", 9, "a" * 16)
        b2 = self._conv2_block("qq:p2", 3, "b" * 16)
        kv = _KVLog()
        seg = _FakeSegmenter(
            _seg_result(
                [b1, b2],
                cursor_updates={
                    "qq:p3": {"source": "conv2", "anchor": "c" * 16, "idx": 5},
                    "qq:p4": 42,  # pmh 会话维持 id 游标
                },
            )
        )
        cur = _FakeCurator(outcomes=[(b1, "created"), (b2, "skipped_budget")])
        runner = MaintenanceRunner(
            context=None,
            memory_mgr=None,
            llm=_RunnerFakeLLM(),
            config={"maintenance_max_llm_calls": 50},
            kv_put=kv.put_kv_data,
            kv_get=kv.get_kv_data,
        )
        patches = [
            patch.object(runner_module, "SegmenterAgent", lambda **kw: seg),
            patch.object(runner_module, "CuratorAgent", lambda **kw: cur),
        ]
        for p in patches:
            p.start()
        try:
            await runner._run_extraction(dry_run=False)
        finally:
            for p in patches:
                p.stop()

        cursors = kv.store[CURSOR_KV_KEY]
        # 已处理块：conv2 游标为 锚点+下标，ts 由 runner 补
        c1 = cursors["qq:p1"]
        self.assertEqual(c1["source"], "conv2")
        self.assertEqual(c1["anchor"], "a" * 16)
        self.assertEqual(c1["idx"], 9)
        self.assertIsInstance(c1["ts"], str)
        # skipped_budget 块不推进
        self.assertNotIn("qq:p2", cursors)
        # 无产出块会话采用分段员回传游标：conv2 dict / pmh int 分流
        c3 = cursors["qq:p3"]
        self.assertEqual(
            (c3["source"], c3["anchor"], c3["idx"]), ("conv2", "c" * 16, 5)
        )
        c4 = cursors["qq:p4"]
        self.assertEqual(c4["id"], 42)
        self.assertNotIn("source", c4)


if __name__ == "__main__":
    unittest.main()
