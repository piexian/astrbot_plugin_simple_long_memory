"""AstrBot 4.29.4 → 4.30.0-beta.2 兼容性与接口修复回归测试。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import yaml
from test_memory_commands import MemoryManager, plugin_main

from maintenance.llm import MaintenanceLLM
from maintenance.relations import extract_plain_text
from maintenance.runner import MaintenanceRunner
from maintenance.scheduler import MaintenanceScheduler


class _Event:
    def __init__(
        self,
        *,
        umo: str = "qq:FriendMessage:u1",
        sender_id: str = "u1",
        sender_name: str = "Alice",
        self_id: str = "bot1",
        event_kind: str = "message",
    ) -> None:
        self.unified_msg_origin = umo
        self._sender_id = sender_id
        self._sender_name = sender_name
        self._self_id = self_id
        self.event_kind = event_kind
        self.message_obj = SimpleNamespace(group=None)

    def get_sender_id(self) -> str:
        return self._sender_id

    def get_sender_name(self) -> str:
        return self._sender_name

    def get_self_id(self) -> str:
        return self._self_id

    def get_platform_id(self) -> str:
        return self.unified_msg_origin.split(":", 1)[0]

    def get_group_id(self) -> str:
        return ""


class _FakePart:
    def __init__(self, part_type: str, text: str = "", *, no_save: bool = False):
        self.type = part_type
        self.text = text
        self._no_save = no_save


class _FakeMessage:
    def __init__(
        self,
        role: str,
        content,
        *,
        no_save: bool = False,
        turn_continuation: bool = False,
    ):
        self.role = role
        self.content = content
        self._no_save = no_save
        self._turn_continuation = turn_continuation


def _make_plugin(*, config: dict | None = None, context=None, memory_mgr=None):
    plugin = plugin_main.MemoryPlugin.__new__(plugin_main.MemoryPlugin)
    plugin.context = context
    plugin.config = {
        "auto_memorize": True,
        "extraction_interval": 1,
        "extraction_min_content_length": 1,
        **(config or {}),
    }
    plugin.memory_mgr = memory_mgr
    plugin._request_snapshots = {}
    plugin._session_counters = {}
    plugin._contact_write_times = {}
    plugin._last_snapshot_cleanup = 0.0
    plugin._last_ttl_expire = 0.0
    plugin._last_consolidation = 0.0
    plugin._background_tasks = set()
    plugin._confirmation_waiters = set()
    plugin._scheduler = None
    return plugin


class DefaultProviderResolutionTests(unittest.TestCase):
    def test_resolves_430_agent_model_provider_id(self):
        ctx = SimpleNamespace(
            get_config=lambda: {
                "agent": {
                    "runtime": "astrbot",
                    "config": {"model": {"provider_id": "prov-430"}},
                }
            },
            provider_manager=SimpleNamespace(provider_insts=[], inst_map={}),
        )
        llm = MaintenanceLLM(context=ctx, cache_enabled=False)
        self.assertEqual(llm._resolve_default_provider(), "prov-430")

    def test_resolves_429_agent_runner_and_legacy_provider_settings(self):
        ctx_429 = SimpleNamespace(
            get_config=lambda: {
                "agent_runner": {
                    "runner_type": "local",
                    "config": {"model": {"provider_id": "prov-429"}},
                }
            },
            provider_manager=SimpleNamespace(provider_insts=[], inst_map={}),
        )
        self.assertEqual(
            MaintenanceLLM(
                context=ctx_429, cache_enabled=False
            )._resolve_default_provider(),
            "prov-429",
        )

        ctx_legacy = SimpleNamespace(
            get_config=lambda: {
                "provider_settings": {"default_provider_id": "prov-legacy"}
            },
            provider_manager=SimpleNamespace(provider_insts=[], inst_map={}),
        )
        self.assertEqual(
            MaintenanceLLM(
                context=ctx_legacy, cache_enabled=False
            )._resolve_default_provider(),
            "prov-legacy",
        )

    def test_falls_back_to_provider_meta_and_config_when_default_missing_or_stale(self):
        prov = SimpleNamespace(
            meta=lambda: SimpleNamespace(id="prov-from-meta"),
            provider_config={"id": "prov-from-config"},
        )
        ctx = SimpleNamespace(
            get_config=lambda: {
                "agent": {
                    "runtime": "astrbot",
                    "config": {"model": {"provider_id": "deleted-provider"}},
                }
            },
            provider_manager=SimpleNamespace(
                default_chat_provider_id="deleted-provider",
                inst_map={"prov-from-meta": prov},
                curr_provider_inst=None,
                provider_insts=[prov],
            ),
        )
        self.assertEqual(
            MaintenanceLLM(
                context=ctx, cache_enabled=False
            )._resolve_default_provider(),
            "prov-from-meta",
        )

        prov_cfg_only = SimpleNamespace(provider_config={"id": "prov-cfg-only"})
        ctx_cfg = SimpleNamespace(
            get_config=lambda: {},
            provider_manager=SimpleNamespace(
                default_chat_provider_id="",
                inst_map={},
                curr_provider_inst=None,
                provider_insts=[prov_cfg_only],
            ),
        )
        self.assertEqual(
            MaintenanceLLM(
                context=ctx_cfg, cache_enabled=False
            )._resolve_default_provider(),
            "prov-cfg-only",
        )

    def test_live_config_overrides_stale_startup_default_chat_provider_id(self):
        prov_a = SimpleNamespace(meta=lambda: SimpleNamespace(id="model-a"))
        prov_b = SimpleNamespace(meta=lambda: SimpleNamespace(id="model-b"))
        live_config = {
            "agent": {
                "runtime": "astrbot",
                "config": {"model": {"provider_id": "model-b"}},
            }
        }
        ctx = SimpleNamespace(
            get_config=lambda: live_config,
            provider_manager=SimpleNamespace(
                default_chat_provider_id="model-a",
                inst_map={"model-a": prov_a, "model-b": prov_b},
                curr_provider_inst=prov_a,
                provider_insts=[prov_a, prov_b],
            ),
        )
        llm = MaintenanceLLM(context=ctx, cache_enabled=False)
        self.assertEqual(llm._resolve_default_provider(), "model-b")


class WakeupAndAgentLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_synthetic_wakeup_skips_contact_injection_and_extraction(self):
        memory_mgr = SimpleNamespace(
            observe_contact=AsyncMock(return_value=True),
            get_contact_summary=AsyncMock(return_value=""),
            has_any_active_memory=AsyncMock(return_value=True),
            recall_memories=AsyncMock(return_value=[]),
            store_memory=AsyncMock(),
        )
        plugin = _make_plugin(memory_mgr=memory_mgr)
        wakeup = _Event(
            sender_id="astrbot",
            sender_name="System",
            self_id="",
            event_kind="agent_wakeup",
        )
        request = SimpleNamespace(prompt="cron prompt", contexts=[])

        await plugin.inject_memories(wakeup, request)
        await plugin.on_agent_begin(wakeup, SimpleNamespace(messages=[]))
        await plugin.extract_memories(wakeup, SimpleNamespace(completion_text="done"))
        await plugin.on_agent_done(
            wakeup,
            SimpleNamespace(context=SimpleNamespace(is_subagent=False), messages=[]),
            SimpleNamespace(completion_text="done"),
        )

        memory_mgr.observe_contact.assert_not_awaited()
        memory_mgr.recall_memories.assert_not_awaited()
        self.assertEqual(plugin._request_snapshots, {})
        self.assertEqual(plugin._session_counters, {})

    async def test_wakeup_with_real_sender_injects_readonly_without_contact_or_extract(
        self,
    ):
        recalled_item = {
            "content": "喜欢手冲咖啡",
            "metadata": {
                "uri": "preferences://coffee",
                "domain": "preferences",
                "memory_scope": "personal",
            },
        }
        memory_mgr = SimpleNamespace(
            observe_contact=AsyncMock(return_value=True),
            get_contact_summary=AsyncMock(return_value=""),
            has_any_active_memory=AsyncMock(return_value=True),
            recall_memories=AsyncMock(return_value=[recalled_item]),
            _current_owner_user_id=lambda ev: "qq_u1",
        )
        ctx = SimpleNamespace(get_config=lambda umo=None: {"timezone": "Asia/Shanghai"})
        plugin = _make_plugin(context=ctx, memory_mgr=memory_mgr)
        wakeup = _Event(
            sender_id="u1",
            sender_name="Alice",
            self_id="bot1",
            event_kind="agent_wakeup",
        )
        request = SimpleNamespace(prompt="子代理任务完成了", contexts=[])

        await plugin.inject_memories(wakeup, request)

        memory_mgr.observe_contact.assert_not_awaited()
        memory_mgr.recall_memories.assert_awaited_once()
        self.assertFalse(memory_mgr.recall_memories.await_args.kwargs["bump"])
        self.assertEqual(plugin._request_snapshots, {})
        self.assertTrue(request.contexts)
        self.assertIn("喜欢手冲咖啡", request.contexts[0]["content"])

    async def test_observe_contact_rejects_wakeup_events_directly(self):
        mgr = MemoryManager(None, {})
        mgr._contact_index = SimpleNamespace(observe=AsyncMock(return_value=True))
        wakeup = _Event(sender_id="u1", event_kind="agent_wakeup")
        self.assertFalse(await mgr.observe_contact(wakeup))
        mgr._contact_index.observe.assert_not_awaited()

    async def test_subagent_and_background_subagent_cannot_corrupt_main_snapshot(self):
        extracted_conversations: list[str] = []

        async def fake_llm_generate(*, chat_provider_id: str, prompt: str):
            extracted_conversations.append(prompt)
            return SimpleNamespace(completion_text="[]")

        ctx = SimpleNamespace(
            get_config=lambda umo=None: {},
            llm_generate=AsyncMock(side_effect=fake_llm_generate),
        )
        memory_mgr = SimpleNamespace(
            is_kb_connected=False,
            observe_contact=AsyncMock(return_value=True),
            get_contact_summary=AsyncMock(return_value=""),
            has_any_active_memory=AsyncMock(return_value=False),
            store_memory=AsyncMock(),
        )
        plugin = _make_plugin(
            config={"extraction_interval": 1, "extraction_min_content_length": 1},
            context=ctx,
            memory_mgr=memory_mgr,
        )
        plugin._get_llm_provider_id = AsyncMock(return_value="p1")

        event = _Event(umo="qq:FriendMessage:u1", sender_id="u1")
        request = SimpleNamespace(prompt="帮我查一下咖啡豆", contexts=[])
        await plugin.inject_memories(event, request)

        initial_user_msg = _FakeMessage("user", "帮我查一下咖啡豆")
        main_rc = SimpleNamespace(
            context=SimpleNamespace(is_subagent=False, event=event),
            messages=[initial_user_msg],
            run_end_reason="completed",
        )
        sub_rc = SimpleNamespace(
            context=SimpleNamespace(is_subagent=True, event=event),
            messages=[_FakeMessage("user", "子代理内部任务")],
            run_end_reason="completed",
        )

        # 主 Agent 启动后，子代理先执行并触发 on_llm_response + on_agent_done
        await plugin.on_agent_begin(event, main_rc)
        await plugin.on_agent_begin(event, sub_rc)
        await plugin.extract_memories(
            event, SimpleNamespace(completion_text="子代理输出")
        )
        await plugin.on_agent_done(
            event, sub_rc, SimpleNamespace(completion_text="子代理输出")
        )
        self.assertEqual(plugin._session_counters, {})
        self.assertEqual(extracted_conversations, [])

        # 主 Agent 正常回复并结束
        main_rc.messages.append(
            _FakeMessage(
                "assistant",
                [
                    _FakePart("think", "隐藏思考"),
                    _FakePart("text", "主回复：推荐耶加雪菲"),
                ],
            )
        )
        await plugin.extract_memories(
            event, SimpleNamespace(completion_text="主回复：推荐耶加雪菲")
        )
        await plugin.on_agent_done(
            event, main_rc, SimpleNamespace(completion_text="主回复：推荐耶加雪菲")
        )

        # 主 Agent 结束后，后台子代理晚到触发 on_llm_response 和 on_agent_done
        await plugin.extract_memories(
            event, SimpleNamespace(completion_text="迟到的后台子代理输出")
        )
        await plugin.on_agent_done(
            event, sub_rc, SimpleNamespace(completion_text="迟到的后台子代理输出")
        )

        self.assertEqual(plugin._session_counters["qq:FriendMessage:u1"], 1)
        self.assertEqual(len(extracted_conversations), 1)
        self.assertIn("帮我查一下咖啡豆", extracted_conversations[0])
        self.assertIn("主回复：推荐耶加雪菲", extracted_conversations[0])
        self.assertNotIn("子代理输出", extracted_conversations[0])
        self.assertNotIn("隐藏思考", extracted_conversations[0])

    async def test_steering_follow_up_is_captured_once_on_agent_done(self):
        extracted_prompts: list[str] = []

        async def fake_llm_generate(*, chat_provider_id: str, prompt: str):
            extracted_prompts.append(prompt)
            return SimpleNamespace(completion_text="[]")

        ctx = SimpleNamespace(
            get_config=lambda umo=None: {},
            llm_generate=AsyncMock(side_effect=fake_llm_generate),
        )
        memory_mgr = SimpleNamespace(
            is_kb_connected=False,
            observe_contact=AsyncMock(return_value=True),
            get_contact_summary=AsyncMock(return_value=""),
            has_any_active_memory=AsyncMock(return_value=False),
            store_memory=AsyncMock(),
        )
        plugin = _make_plugin(
            config={"extraction_interval": 1, "extraction_min_content_length": 1},
            context=ctx,
            memory_mgr=memory_mgr,
        )
        plugin._get_llm_provider_id = AsyncMock(return_value="p1")

        event = _Event(umo="qq:FriendMessage:u1", sender_id="u1")
        request = SimpleNamespace(prompt="推荐一本书", contexts=[])
        await plugin.inject_memories(event, request)

        anchor = _FakeMessage("user", "推荐一本书")
        main_rc = SimpleNamespace(
            context=SimpleNamespace(is_subagent=False, event=event),
            messages=[anchor],
            run_end_reason="completed",
        )
        await plugin.on_agent_begin(event, main_rc)

        # 第一段回复后用户 Steering 追加输入，再产出第二段最终回复
        main_rc.messages.extend(
            [
                _FakeMessage("assistant", [_FakePart("text", "先推荐科幻类的")]),
                _FakeMessage(
                    "user",
                    [
                        _FakePart("text", "补充一下，我想看历史类"),
                        _FakePart(
                            "text",
                            "<system_reminder>steer</system_reminder>",
                            no_save=True,
                        ),
                    ],
                    turn_continuation=True,
                ),
                _FakeMessage("assistant", [_FakePart("text", "改推荐《万历十五年》")]),
            ]
        )
        await plugin.extract_memories(
            event, SimpleNamespace(completion_text="先推荐科幻类的")
        )
        await plugin.extract_memories(
            event, SimpleNamespace(completion_text="改推荐《万历十五年》")
        )
        await plugin.on_agent_done(
            event, main_rc, SimpleNamespace(completion_text="改推荐《万历十五年》")
        )

        self.assertEqual(plugin._session_counters["qq:FriendMessage:u1"], 1)
        self.assertEqual(len(extracted_prompts), 1)
        self.assertIn("推荐一本书", extracted_prompts[0])
        self.assertIn("补充一下，我想看历史类", extracted_prompts[0])
        self.assertNotIn("<system_reminder>", extracted_prompts[0])
        self.assertIn("改推荐《万历十五年》", extracted_prompts[0])

    async def test_cancelled_empty_run_discards_pending_prompt_without_counting(self):
        plugin = _make_plugin(memory_mgr=SimpleNamespace())
        event = _Event(umo="qq:FriendMessage:u1", sender_id="u1")
        plugin._accumulate_request_snapshot(
            event, SimpleNamespace(prompt="刚发出的消息", contexts=[])
        )
        rc = SimpleNamespace(
            context=SimpleNamespace(is_subagent=False, event=event),
            messages=[],
            run_end_reason="cancelled",
        )
        await plugin.on_agent_begin(event, rc)
        await plugin.on_agent_done(event, rc, None)

        self.assertEqual(plugin._session_counters, {})
        self.assertIsNone(
            plugin._request_snapshots["qq:FriendMessage:u1"]["pending_prompt"]
        )

    def test_collect_turn_messages_recovers_after_in_turn_compaction(self):
        original_anchor = _FakeMessage("user", "初始问题")
        run_state = {"start_len": 10, "anchor": original_anchor}
        # 模拟轮内上下文压缩替换了原始消息列表，并验证字典形态的 _turn_continuation
        compacted_messages = [
            {"role": "system", "content": "历史摘要"},
            {"role": "user", "content": "初始问题"},
            {"role": "assistant", "content": [{"type": "text", "text": "第一段回复"}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": "追加限制条件"}],
                "_turn_continuation": True,
            },
            {"role": "assistant", "content": [{"type": "text", "text": "第二段回复"}]},
        ]
        rc = SimpleNamespace(messages=compacted_messages)

        turn_messages = plugin_main._collect_turn_messages(rc, run_state)
        self.assertEqual(
            turn_messages,
            [
                ("assistant", "第一段回复"),
                ("user", "追加限制条件"),
                ("assistant", "第二段回复"),
            ],
        )

    def test_subagent_results_and_interruption_messages_are_filtered(self):
        anchor = _FakeMessage("user", "帮我调研一下")
        run_state = {"start_len": 1, "anchor": anchor}
        rc = SimpleNamespace(
            run_end_reason="cancelled",
            messages=[
                anchor,
                _FakeMessage("assistant", [_FakePart("text", "正在安排调研")]),
                _FakeMessage(
                    "user",
                    [
                        _FakePart(
                            "text",
                            "<system_reminder>\n[Subagent results]\n"
                            "[subagent-result:a1:t1]\n"
                            '[{"summary": "子代理结论"}]\n'
                            "</system_reminder>",
                        )
                    ],
                    turn_continuation=True,
                ),
                _FakeMessage(
                    "user",
                    [
                        _FakePart(
                            "text",
                            "<system_reminder>\n[Subagent results]\n"
                            "internal\n</system_reminder>",
                        ),
                        _FakePart("text", "顺便重点看性能数据"),
                    ],
                    turn_continuation=True,
                ),
                _FakeMessage("user", [_FakePart("text", "Stop output.")]),
                _FakeMessage("assistant", [_FakePart("text", "Output stopped.")]),
            ],
        )

        turn_messages = plugin_main._collect_turn_messages(rc, run_state)
        self.assertEqual(
            turn_messages,
            [
                ("assistant", "正在安排调研"),
                ("user", "顺便重点看性能数据"),
            ],
        )

    async def test_err_response_with_non_empty_text_is_not_recorded_or_counted(self):
        plugin = _make_plugin(
            config={"extraction_interval": 1, "extraction_min_content_length": 1},
            memory_mgr=SimpleNamespace(),
        )
        event = _Event(umo="qq:FriendMessage:u1", sender_id="u1")
        plugin._accumulate_request_snapshot(
            event, SimpleNamespace(prompt="今天天气怎么样", contexts=[])
        )
        rc = SimpleNamespace(
            context=SimpleNamespace(is_subagent=False, event=event),
            messages=[_FakeMessage("user", "今天天气怎么样")],
            run_end_reason="error",
        )
        await plugin.on_agent_begin(event, rc)
        await plugin.on_agent_done(
            event,
            rc,
            SimpleNamespace(
                role="err",
                completion_text="All available chat models are unavailable.",
            ),
        )

        self.assertEqual(plugin._session_counters, {})
        self.assertEqual(
            plugin._request_snapshots["qq:FriendMessage:u1"]["snapshots"], []
        )
        self.assertIsNone(
            plugin._request_snapshots["qq:FriendMessage:u1"]["pending_prompt"]
        )

    async def test_steering_preserves_order_and_deidentifies_group_followups(self):
        extracted_prompts: list[str] = []

        async def fake_llm_generate(*, chat_provider_id: str, prompt: str):
            extracted_prompts.append(prompt)
            return SimpleNamespace(completion_text="[]")

        ctx = SimpleNamespace(
            get_config=lambda umo=None: {},
            llm_generate=AsyncMock(side_effect=fake_llm_generate),
        )
        memory_mgr = SimpleNamespace(
            is_kb_connected=False,
            observe_contact=AsyncMock(return_value=True),
            get_contact_summary=AsyncMock(return_value=""),
            has_any_active_memory=AsyncMock(return_value=False),
            store_memory=AsyncMock(),
        )
        plugin = _make_plugin(
            config={"extraction_interval": 1, "extraction_min_content_length": 1},
            context=ctx,
            memory_mgr=memory_mgr,
        )
        plugin._get_llm_provider_id = AsyncMock(return_value="p1")

        # 群聊场景：保持轮内先后顺序，且 Steering 追问不沿用发起人 sender_id
        group_event = _Event(umo="qq:GroupMessage:g1", sender_id="u1")
        await plugin.inject_memories(
            group_event, SimpleNamespace(prompt="帮我规划一下方案", contexts=[])
        )
        anchor = _FakeMessage("user", "帮我规划一下方案")
        group_rc = SimpleNamespace(
            context=SimpleNamespace(is_subagent=False, event=group_event),
            messages=[anchor],
            run_end_reason="completed",
        )
        await plugin.on_agent_begin(group_event, group_rc)
        group_rc.messages.extend(
            [
                _FakeMessage("assistant", [_FakePart("text", "我会按方案 A 规划")]),
                _FakeMessage(
                    "user",
                    [_FakePart("text", "不要这个，换另一个")],
                    turn_continuation=True,
                ),
                _FakeMessage("assistant", [_FakePart("text", "那就采用方案 B")]),
            ]
        )
        await plugin.on_agent_done(
            group_event,
            group_rc,
            SimpleNamespace(role="assistant", completion_text="那就采用方案 B"),
        )

        self.assertEqual(len(extracted_prompts), 1)
        group_conv = extracted_prompts[0]
        pos_init = group_conv.index("[用户:u1]: 帮我规划一下方案")
        pos_a = group_conv.index("[助手]: 我会按方案 A 规划")
        pos_steer = group_conv.index("[用户]: 不要这个，换另一个")
        pos_b = group_conv.index("[助手]: 那就采用方案 B")
        self.assertTrue(pos_init < pos_a < pos_steer < pos_b)
        self.assertNotIn("[用户:u1]: 不要这个，换另一个", group_conv)

        # 私聊场景：追问保留同一用户 sender_id 并保持时间顺序
        private_event = _Event(umo="qq:FriendMessage:u1", sender_id="u1")
        await plugin.inject_memories(
            private_event, SimpleNamespace(prompt="推荐一本书", contexts=[])
        )
        p_anchor = _FakeMessage("user", "推荐一本书")
        p_rc = SimpleNamespace(
            context=SimpleNamespace(is_subagent=False, event=private_event),
            messages=[p_anchor],
            run_end_reason="completed",
        )
        await plugin.on_agent_begin(private_event, p_rc)
        p_rc.messages.extend(
            [
                _FakeMessage("assistant", [_FakePart("text", "先推荐科幻类")]),
                _FakeMessage(
                    "user",
                    [_FakePart("text", "换成历史类")],
                    turn_continuation=True,
                ),
                _FakeMessage("assistant", [_FakePart("text", "推荐《万历十五年》")]),
            ]
        )
        await plugin.on_agent_done(
            private_event,
            p_rc,
            SimpleNamespace(role="assistant", completion_text="推荐《万历十五年》"),
        )

        self.assertEqual(len(extracted_prompts), 2)
        priv_conv = extracted_prompts[1]
        self.assertIn("[用户:u1]: 换成历史类", priv_conv)
        self.assertLess(
            priv_conv.index("[助手]: 先推荐科幻类"),
            priv_conv.index("[用户:u1]: 换成历史类"),
        )

    async def test_overlapping_turns_isolate_pending_snapshots_and_clear(self):
        plugin = _make_plugin(
            config={"extraction_interval": 10}, memory_mgr=SimpleNamespace()
        )
        ev_a = _Event(umo="qq:GroupMessage:g1", sender_id="u1")
        ev_b = _Event(umo="qq:GroupMessage:g1", sender_id="u2")

        plugin._accumulate_request_snapshot(
            ev_a, SimpleNamespace(prompt="请求 A", contexts=[])
        )
        plugin._accumulate_request_snapshot(
            ev_b, SimpleNamespace(prompt="请求 B", contexts=[])
        )

        # 取消 A 不应清除仍在进行的 B 的待匹配请求
        plugin._clear_pending_snapshot(ev_a)
        self.assertEqual(
            plugin._request_snapshots["qq:GroupMessage:g1"]["pending_prompt"], "请求 B"
        )

        plugin._complete_snapshot_with_response(ev_b, "回复 B")
        snaps = plugin._request_snapshots["qq:GroupMessage:g1"]["snapshots"]
        self.assertEqual(len(snaps), 1)
        self.assertEqual(snaps[0]["prompt"], "请求 B")
        self.assertEqual(snaps[0]["response"], "回复 B")
        self.assertEqual(snaps[0]["sender_id"], "u2")

        # 两次重叠轮次错序完成时，各自匹配自己的提问
        ev_c = _Event(umo="qq:GroupMessage:g2", sender_id="u1")
        ev_d = _Event(umo="qq:GroupMessage:g2", sender_id="u2")
        plugin._accumulate_request_snapshot(
            ev_c, SimpleNamespace(prompt="请求 C", contexts=[])
        )
        plugin._accumulate_request_snapshot(
            ev_d, SimpleNamespace(prompt="请求 D", contexts=[])
        )
        plugin._complete_snapshot_with_response(ev_c, "回复 C")
        plugin._complete_snapshot_with_response(ev_d, "回复 D")
        snaps_g2 = plugin._request_snapshots["qq:GroupMessage:g2"]["snapshots"]
        self.assertEqual(
            [(s["prompt"], s["response"], s["sender_id"]) for s in snaps_g2],
            [("请求 C", "回复 C", "u1"), ("请求 D", "回复 D", "u2")],
        )

    def test_unweakrefable_event_tracking_uses_object_identity(self):
        class _SlottedEvent:
            __slots__ = ("name",)

            def __init__(self, name: str):
                self.name = name

        plugin = _make_plugin()
        ev1 = _SlottedEvent("first")
        ev2 = _SlottedEvent("second")

        plugin_main._mark_agent_managed_event(plugin, ev1)
        plugin_main._set_agent_run_state(plugin, ev1, {"start_len": 3, "anchor": None})

        self.assertTrue(plugin_main._is_agent_managed_event(plugin, ev1))
        self.assertFalse(plugin_main._is_agent_managed_event(plugin, ev2))
        self.assertEqual(
            plugin_main._pop_agent_run_state(plugin, ev1),
            {"start_len": 3, "anchor": None},
        )
        self.assertIsNone(plugin_main._pop_agent_run_state(plugin, ev1))


class ConfigAndNotificationFixTests(unittest.IsolatedAsyncioTestCase):
    def test_timezone_and_cmd_prefix_use_context_get_config(self):
        seen_umos: list[str | None] = []

        def get_config(umo: str | None = None):
            seen_umos.append(umo)
            if umo == "qq:GroupMessage:g1":
                return {"timezone": "UTC", "wake_prefix": ["!"]}
            return {"timezone": "Asia/Shanghai", "wake_prefix": ["/"]}

        # 刻意不设置 astrbot_config 属性，模拟真实 AstrBot Context
        ctx = SimpleNamespace(get_config=get_config)
        plugin = _make_plugin(context=ctx)

        self.assertEqual(plugin._get_display_timezone("qq:GroupMessage:g1"), "UTC")
        self.assertEqual(plugin._get_cmd_prefix("qq:GroupMessage:g1"), "!")
        self.assertEqual(plugin._get_display_timezone(), "Asia/Shanghai")
        self.assertEqual(plugin._get_cmd_prefix(), "/")
        self.assertIn("qq:GroupMessage:g1", seen_umos)

        scheduler = MaintenanceScheduler(
            context=ctx, memory_mgr=SimpleNamespace(), config={}
        )
        self.assertEqual(scheduler._tz_str, "Asia/Shanghai")

    async def test_review_notification_imports_message_chain_from_api_event(self):
        from astrbot.api import star as astrbot_star
        from astrbot.api.event import MessageChain

        self.assertFalse(hasattr(astrbot_star, "MessageChain"))
        sent: list[tuple[str, object]] = []

        async def send_message(umo: str, msg):
            sent.append((umo, msg))

        runner = MaintenanceRunner(
            context=SimpleNamespace(send_message=send_message),
            memory_mgr=SimpleNamespace(),
            llm=SimpleNamespace(),
            config={"review_notify_umo": "qq:FriendMessage:admin"},
            kv_put=None,
            kv_get=None,
        )
        await runner._send_review_notification(3)

        self.assertEqual(len(sent), 1)
        umo, msg = sent[0]
        self.assertEqual(umo, "qq:FriendMessage:admin")
        self.assertIsInstance(msg, MessageChain)
        if hasattr(msg, "use_markdown_"):
            self.assertIs(msg.use_markdown_, False)
        self.assertIn("3 条争议操作", msg.chain[0].text)

    def test_plugin_registers_real_astrbot_handlers_and_commands(self):
        from astrbot.core.star.filter.command import CommandFilter
        from astrbot.core.star.filter.command_group import CommandGroupFilter
        from astrbot.core.star.star_handler import EventType, star_handlers_registry

        by_name = {h.handler_name: h for h in star_handlers_registry}
        for name, expected_type in (
            ("inject_memories", EventType.OnLLMRequestEvent),
            ("extract_memories", EventType.OnLLMResponseEvent),
            ("on_agent_begin", EventType.OnAgentBeginEvent),
            ("on_agent_done", EventType.OnAgentDoneEvent),
            ("on_loaded", EventType.OnAstrBotLoadedEvent),
            ("memory_group", EventType.AdapterMessageEvent),
            ("cmd_list", EventType.AdapterMessageEvent),
            ("cmd_search", EventType.AdapterMessageEvent),
            ("cmd_rebuild", EventType.AdapterMessageEvent),
        ):
            self.assertIn(name, by_name, f"missing registered handler: {name}")
            self.assertEqual(by_name[name].event_type, expected_type)

        group_filters = [
            f
            for f in by_name["memory_group"].event_filters
            if isinstance(f, CommandGroupFilter)
        ]
        self.assertTrue(any(f.group_name == "memory" for f in group_filters))

        for handler_name, expected_cmd in (
            ("cmd_list", "list"),
            ("cmd_search", "search"),
            ("cmd_rebuild", "rebuild"),
        ):
            cmd_filters = [
                f
                for f in by_name[handler_name].event_filters
                if isinstance(f, CommandFilter)
            ]
            self.assertTrue(
                any(
                    f.command_name == expected_cmd
                    and list(f.parent_command_names) == ["memory"]
                    for f in cmd_filters
                ),
                f"{handler_name} did not register /memory {expected_cmd}",
            )


class AnalystContentTests(unittest.IsolatedAsyncioTestCase):
    def test_extracts_plain_text_from_structured_content_parts(self):
        user_text = extract_plain_text(
            [
                {"type": "text", "text": "我最喜欢拿铁"},
                {"type": "text", "text": "<temp>", "_no_save": True},
                {"type": "image_url", "image_url": {"url": "data:..."}},
            ]
        )
        assistant_text = extract_plain_text(
            [
                {"type": "think", "think": "内部思考", "encrypted": "SECRET_SIG"},
                {"type": "text", "text": "记住了，你喜欢拿铁。"},
            ]
        )

        self.assertEqual(user_text, "我最喜欢拿铁")
        self.assertEqual(assistant_text, "记住了，你喜欢拿铁。")
        self.assertNotIn("内部思考", assistant_text)
        self.assertNotIn("SECRET_SIG", assistant_text)
        self.assertNotIn("<temp>", user_text)
        self.assertNotIn("data:...", user_text)


class MetadataAndI18nSchemaTests(unittest.TestCase):
    def test_metadata_schema_and_i18n_cover_all_config_keys(self):
        root = Path(__file__).resolve().parents[1]
        meta = yaml.safe_load((root / "metadata.yaml").read_text(encoding="utf-8"))
        self.assertEqual(meta["category"], "ai_context")
        self.assertEqual(meta["astrbot_version"], ">=4.29.4")

        schema = json.loads((root / "_conf_schema.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["review_notify_umo"]["_special"], "umo")

        for locale in ("zh-CN", "en-US"):
            i18n = json.loads(
                (root / ".astrbot-plugin" / "i18n" / f"{locale}.json").read_text(
                    encoding="utf-8"
                )
            )
            cfg = i18n["config"]
            self.assertEqual(set(cfg.keys()), set(schema.keys()), locale)
            for key, entry in cfg.items():
                self.assertTrue(entry.get("description"), f"{locale}:{key}")
                self.assertTrue(entry.get("hint"), f"{locale}:{key}")


if __name__ == "__main__":
    unittest.main()
