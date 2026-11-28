"""AstrBot 4.29.4 → 4.30.0-beta.2 兼容性回归：真实 Context、真实事件与真实消息结构。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import yaml
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.core.agent.message import Message, TextPart, ThinkPart
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_runtime.subagents.context import SubAgentContext
from astrbot.core.config_bindings import BINDING_KEY
from real_stack import (
    CHAT_PROVIDER_ID,
    RealStackTestCase,
    make_memory_meta,
    plugin_main,
    provider_configs,
    real_message_event,
)
from sqlalchemy import text as sa_text

from maintenance.llm import MaintenanceLLM
from maintenance.relations import extract_plain_text
from maintenance.runner import MaintenanceRunner
from maintenance.scheduler import MaintenanceScheduler

SECOND_CHAT_PROVIDER_ID = "test-chat-secondary"
PLUGIN_CONFIG = {
    "auto_memorize": True,
    "extraction_interval": 1,
    "extraction_min_content_length": 1,
    "extraction_provider_id": CHAT_PROVIDER_ID,
    "memory_consolidation_enabled": False,
}
COFFEE_MEMORY = (
    "doc-coffee",
    "喜欢手冲咖啡",
    make_memory_meta(
        "preferences://coffee",
        domain="preferences",
        owner="webchat_tester",
        umo="webchat:FriendMessage:tester",
    ),
)


def _user(text: str, *, temp: str = "") -> Message:
    """构造用户消息，可附带本轮临时提醒块。"""
    parts = [TextPart(text=text)]
    if temp:
        parts.append(TextPart(text=temp).mark_as_temp())
    return Message(role="user", content=parts)


def _assistant(text: str) -> Message:
    return Message(role="assistant", content=[TextPart(text=text)])


def _continuation(message: Message) -> Message:
    """标记为 Steering 续轮消息，与真实 Agent 运行上下文一致。"""
    message._turn_continuation = True
    return message


class _RealPluginCase(RealStackTestCase):
    """真实插件实例 + 真实事件的提取与生命周期用例底座。"""

    async def asyncSetUp(self):
        await self.start_stack(
            with_decision=False, config=dict(PLUGIN_CONFIG), with_plugin=True
        )
        self.server.set_default_chat("[]")

    async def _restart(self, **kwargs):
        """换一套真实 KB，隔离同一用例内的多轮断言。"""
        self.server.shutdown()
        self.server.server_close()
        await self.start_stack(
            with_decision=False, config=dict(PLUGIN_CONFIG), with_plugin=True, **kwargs
        )

    def _run_context(
        self, event, messages, *, end_reason="completed", is_subagent=False
    ):
        inner = (
            SubAgentContext(context=self.context, event=event, is_subagent=True)
            if is_subagent
            else None
        )
        return ContextWrapper(
            context=inner, messages=messages, run_end_reason=end_reason
        )

    async def _run_turn(self, event, prompt, follow_ups, *, final_text):
        """按真实 Agent 顺序跑一轮：注入 → 起始锚点 → 续轮消息 → 收口。"""
        await self.plugin.inject_memories(event, ProviderRequest(prompt=prompt))
        messages = [Message(role="user", content=prompt)]
        await self.plugin.on_agent_begin(event, self._run_context(event, messages))
        messages.extend(follow_ups)
        await self.plugin.on_agent_done(
            event,
            self._run_context(event, messages),
            LLMResponse(role="assistant", completion_text=final_text),
        )

    def _chat_prompts(self) -> list[str]:
        """收集真实 provider 收到的 chat 请求体，用于断言提取输入。"""
        return [
            json.dumps(entry.get("messages"), ensure_ascii=False)
            for entry in self.server.request_log
            if entry.get("kind") == "chat"
        ]

    async def _contact_rows(self) -> list[dict]:
        async with self.mgr.vec_db.document_storage.get_session() as session:
            rows = await session.execute(sa_text("SELECT * FROM contact_observations"))
            return [dict(row) for row in rows.mappings()]


class WakeupAndAgentLifecycleTests(_RealPluginCase):
    async def test_synthetic_wakeup_skips_contact_and_extraction(self):
        wakeup = real_message_event(
            sender_id="astrbot", sender_name="System", self_id=""
        )
        wakeup.event_kind = "agent_wakeup"
        messages = [Message(role="user", content="cron prompt")]
        request = ProviderRequest(prompt="cron prompt")

        await self.plugin.inject_memories(wakeup, request)
        await self.plugin.on_agent_begin(wakeup, self._run_context(wakeup, messages))
        await self.plugin.extract_memories(
            wakeup, LLMResponse(role="assistant", completion_text="done")
        )
        await self.plugin.on_agent_done(
            wakeup,
            self._run_context(wakeup, messages),
            LLMResponse(role="assistant", completion_text="done"),
        )

        self.assertFalse(await self.mgr.observe_contact(wakeup))
        self.assertEqual(request.extra_user_content_parts, [])
        self.assertEqual(await self._contact_rows(), [])
        self.assertEqual(self._chat_prompts(), [])
        self.assertEqual(self.plugin._request_snapshots, {})
        self.assertEqual(self.plugin._session_counters, {})

    async def test_wakeup_with_real_sender_injects_readonly(self):
        await self._restart(memories=[COFFEE_MEMORY])
        wakeup = real_message_event(sender_id="tester", message_str="子代理任务完成了")
        wakeup.event_kind = "agent_wakeup"
        request = ProviderRequest(prompt="子代理任务完成了")

        await self.plugin.inject_memories(wakeup, request)

        injected = "\n".join(part.text for part in request.extra_user_content_parts)
        self.assertIn("喜欢手冲咖啡", injected)
        self.assertEqual(await self._contact_rows(), [])
        self.assertEqual(self.plugin._request_snapshots, {})
        self.assertEqual(self._chat_prompts(), [])

    async def test_subagent_runs_do_not_corrupt_main_snapshot(self):
        event = real_message_event(message_str="帮我查一下咖啡豆")
        await self.plugin.inject_memories(
            event, ProviderRequest(prompt="帮我查一下咖啡豆")
        )
        main_messages = [Message(role="user", content="帮我查一下咖啡豆")]
        main_context = self._run_context(event, main_messages)
        sub_context = self._run_context(
            event, [Message(role="user", content="子代理内部任务")], is_subagent=True
        )

        await self.plugin.on_agent_begin(event, main_context)
        await self.plugin.on_agent_begin(event, sub_context)
        await self.plugin.on_agent_done(
            event,
            sub_context,
            LLMResponse(role="assistant", completion_text="子代理输出"),
        )
        self.assertEqual(self.plugin._session_counters, {})
        self.assertEqual(self._chat_prompts(), [])

        main_messages.append(
            Message(
                role="assistant",
                content=[
                    ThinkPart(think="隐藏思考"),
                    TextPart(text="主回复：推荐耶加雪菲"),
                ],
            )
        )
        await self.plugin.on_agent_done(
            event,
            main_context,
            LLMResponse(role="assistant", completion_text="主回复：推荐耶加雪菲"),
        )

        prompts = self._chat_prompts()
        self.assertEqual(self.plugin._session_counters[event.unified_msg_origin], 1)
        self.assertEqual(len(prompts), 1)
        self.assertIn("帮我查一下咖啡豆", prompts[0])
        self.assertIn("主回复：推荐耶加雪菲", prompts[0])
        self.assertNotIn("子代理输出", prompts[0])
        self.assertNotIn("隐藏思考", prompts[0])

    async def test_steering_keeps_order_and_deidentifies_group_followups(self):
        await self._run_turn(
            real_message_event(umo="webchat:GroupMessage:g1", sender_id="u1"),
            "帮我规划一下方案",
            [
                _assistant("我会按方案 A 规划"),
                _continuation(
                    _user(
                        "不要这个，换另一个",
                        temp="<system_reminder>steer</system_reminder>",
                    )
                ),
                _assistant("那就采用方案 B"),
            ],
            final_text="那就采用方案 B",
        )
        await self._run_turn(
            real_message_event(sender_id="u1"),
            "推荐一本书",
            [
                _assistant("先推荐科幻类"),
                _continuation(_user("换成历史类")),
                _assistant("推荐《万历十五年》"),
            ],
            final_text="推荐《万历十五年》",
        )

        prompts = self._chat_prompts()
        self.assertEqual(len(prompts), 2)
        group_conv = prompts[0]
        positions = [
            group_conv.index("[用户:u1]: 帮我规划一下方案"),
            group_conv.index("[助手]: 我会按方案 A 规划"),
            group_conv.index("[用户]: 不要这个，换另一个"),
            group_conv.index("[助手]: 那就采用方案 B"),
        ]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("<system_reminder>", group_conv)

        private_conv = prompts[1]
        self.assertIn("[用户:u1]: 换成历史类", private_conv)
        self.assertLess(
            private_conv.index("[助手]: 先推荐科幻类"),
            private_conv.index("[用户:u1]: 换成历史类"),
        )

    def test_collect_turn_messages_recovers_after_in_turn_compaction(self):
        # 轮内上下文压缩替换了原始消息列表，起始锚点已不在列表中
        compacted = [
            Message(role="system", content="历史摘要"),
            _user("初始问题"),
            _assistant("第一段回复"),
            _continuation(_user("追加限制条件")),
            _assistant("第二段回复"),
        ]
        run_state = {"start_len": 10, "anchor": _user("初始问题")}

        self.assertEqual(
            plugin_main._collect_turn_messages(
                ContextWrapper(context=None, messages=compacted), run_state
            ),
            [
                ("assistant", "第一段回复"),
                ("user", "追加限制条件"),
                ("assistant", "第二段回复"),
            ],
        )

        # 子代理结果提醒与中断控制消息不进入提取对话
        anchor = _user("帮我调研一下")
        interrupted = ContextWrapper(
            context=None,
            run_end_reason="cancelled",
            messages=[
                anchor,
                _assistant("正在安排调研"),
                _continuation(
                    _user(
                        "<system_reminder>\n[Subagent results]\n"
                        '[{"summary": "子代理结论"}]\n</system_reminder>'
                    )
                ),
                _user("顺便重点看性能数据"),
                _user("Stop output."),
                _assistant("Output stopped."),
            ],
        )

        self.assertEqual(
            plugin_main._collect_turn_messages(
                interrupted, {"start_len": 1, "anchor": anchor}
            ),
            [("assistant", "正在安排调研"), ("user", "顺便重点看性能数据")],
        )

    async def test_overlapping_turns_isolate_pending_snapshots(self):
        event_a = real_message_event(umo="webchat:GroupMessage:g1", sender_id="u1")
        event_b = real_message_event(umo="webchat:GroupMessage:g1", sender_id="u2")
        self.plugin._accumulate_request_snapshot(
            event_a, ProviderRequest(prompt="请求 A")
        )
        self.plugin._accumulate_request_snapshot(
            event_b, ProviderRequest(prompt="请求 B")
        )

        # 取消 A 不应清除仍在进行的 B 的待匹配请求
        self.plugin._clear_pending_snapshot(event_a)
        self.assertEqual(
            self.plugin._request_snapshots["webchat:GroupMessage:g1"]["pending_prompt"],
            "请求 B",
        )

        self.plugin._complete_snapshot_with_response(event_b, "回复 B")
        snapshots = self.plugin._request_snapshots["webchat:GroupMessage:g1"][
            "snapshots"
        ]

        self.assertEqual(
            [(s["prompt"], s["response"], s["sender_id"]) for s in snapshots],
            [("请求 B", "回复 B", "u2")],
        )

    def test_unweakrefable_event_tracking_uses_object_identity(self):
        class _SlottedEvent:
            __slots__ = ("name",)

            def __init__(self, name: str):
                self.name = name

        plugin = self.plugin
        first = _SlottedEvent("first")
        plugin_main._mark_agent_managed_event(plugin, first)
        plugin_main._set_agent_run_state(
            plugin, first, {"start_len": 3, "anchor": None}
        )

        self.assertTrue(plugin_main._is_agent_managed_event(plugin, first))
        self.assertFalse(
            plugin_main._is_agent_managed_event(plugin, _SlottedEvent("x"))
        )
        self.assertEqual(
            plugin_main._pop_agent_run_state(plugin, first),
            {"start_len": 3, "anchor": None},
        )
        self.assertIsNone(plugin_main._pop_agent_run_state(plugin, first))


class DefaultProviderResolutionTests(RealStackTestCase):
    async def asyncSetUp(self):
        await self.start_stack(with_decision=False)
        self.llm = MaintenanceLLM(context=self.context, cache_enabled=False)
        self.config = self.context.get_config()
        self.original_config = json.loads(json.dumps(dict(self.config)))

    async def asyncTearDown(self):
        self.config.clear()
        self.config.update(self.original_config)

    async def _load_chat_provider(self, provider_id: str) -> None:
        provider_config = dict(provider_configs(self.server.base_url)["chat"])
        provider_config["id"] = provider_id
        await self.context.provider_manager.load_provider(provider_config)

    def _set_provider_config(self, *, agent="", agent_runner="", legacy=""):
        """只保留一种默认模型配置来源，模拟对应的 AstrBot 版本。"""
        for key in ("agent", "agent_runner", "provider_settings"):
            self.config.pop(key, None)
        if agent:
            self.config["agent"] = {
                "runtime": "astrbot",
                "config": {"model": {"provider_id": agent}},
            }
        if agent_runner:
            self.config["agent_runner"] = {
                "runner_type": "local",
                "config": {"model": {"provider_id": agent_runner}},
            }
        if legacy:
            self.config["provider_settings"] = {"default_provider_id": legacy}

    async def test_resolves_430_agent_model_provider_id(self):
        self._set_provider_config(agent=CHAT_PROVIDER_ID)

        self.assertEqual(self.llm._resolve_default_provider(), CHAT_PROVIDER_ID)
        self.assertEqual(
            self.context.get_config()["agent"]["config"]["model"]["provider_id"],
            CHAT_PROVIDER_ID,
        )

    async def test_resolves_429_agent_runner_and_legacy_provider_settings(self):
        await self._load_chat_provider(SECOND_CHAT_PROVIDER_ID)

        self._set_provider_config(agent_runner=SECOND_CHAT_PROVIDER_ID)
        self.assertEqual(self.llm._resolve_default_provider(), SECOND_CHAT_PROVIDER_ID)

        self._set_provider_config(legacy=CHAT_PROVIDER_ID)
        self.assertEqual(self.llm._resolve_default_provider(), CHAT_PROVIDER_ID)

    async def test_falls_back_to_provider_meta_when_default_is_stale(self):
        manager = self.context.provider_manager
        provider = manager.inst_map[CHAT_PROVIDER_ID]
        self._set_provider_config(agent="deleted-provider")
        manager.default_chat_provider_id = "deleted-provider"
        manager.curr_provider_inst = None

        resolved = self.llm._resolve_default_provider()

        self.assertEqual(resolved, provider.meta().id)
        self.assertIn(resolved, manager.inst_map)

    async def test_live_config_overrides_stale_default_chat_provider_id(self):
        await self._load_chat_provider(SECOND_CHAT_PROVIDER_ID)
        manager = self.context.provider_manager
        manager.default_chat_provider_id = CHAT_PROVIDER_ID
        self._set_provider_config(agent=SECOND_CHAT_PROVIDER_ID)

        self.assertEqual(self.llm._resolve_default_provider(), SECOND_CHAT_PROVIDER_ID)


class ConfigAndNotificationFixTests(RealStackTestCase):
    async def asyncSetUp(self):
        await self.start_stack(with_decision=False, with_plugin=True)

    async def test_timezone_and_cmd_prefix_use_real_context_config(self):
        config = self.context.get_config()
        session_umo = "webchat:FriendMessage:tester"
        config_mgr = self.context.astrbot_config_mgr
        # 核心启动时会初始化配置文件管理器，之后才能解析会话级覆盖
        await config_mgr.initialize()

        await config_mgr.sp.put_async(
            "umo", session_umo, BINDING_KEY, {"overrides": {"wake_prefix": ["!"]}}
        )
        try:
            self.assertEqual(self.plugin._get_cmd_prefix(session_umo), "!")
            self.assertEqual(
                self.plugin._get_cmd_prefix("webchat:FriendMessage:x"), "/"
            )
            self.assertEqual(
                self.plugin._get_display_timezone(session_umo), config.get("timezone")
            )
        finally:
            await config_mgr.sp.remove_async("umo", session_umo, BINDING_KEY)

        self.assertEqual(self.plugin._get_cmd_prefix(), config["wake_prefix"][0])
        self.assertEqual(self.plugin._get_display_timezone(), config.get("timezone"))
        scheduler = MaintenanceScheduler(
            context=self.context, memory_mgr=self.mgr, config={}
        )
        self.assertEqual(scheduler._tz_str, config.get("timezone"))

    async def test_review_notification_imports_message_chain_from_api_event(self):
        from astrbot.api import star as astrbot_star
        from astrbot.api.event import MessageChain

        self.assertFalse(hasattr(astrbot_star, "MessageChain"))
        runner = MaintenanceRunner(
            context=self.context,
            memory_mgr=self.mgr,
            llm=self.make_llm(),
            config={"review_notify_umo": "webchat:FriendMessage:admin"},
        )

        with patch.object(self.context, "send_message", new=AsyncMock()) as send:
            await runner._send_review_notification(3)

        self.assertEqual(send.await_count, 1)
        umo, message = send.await_args.args
        self.assertEqual(umo, "webchat:FriendMessage:admin")
        self.assertIsInstance(message, MessageChain)
        self.assertFalse(getattr(message, "use_markdown_", False))
        self.assertIn("3 条争议操作", message.chain[0].text)

    def test_plugin_registers_real_astrbot_handlers_and_commands(self):
        from astrbot.core.star.filter.command import CommandFilter
        from astrbot.core.star.filter.command_group import CommandGroupFilter
        from astrbot.core.star.star_handler import EventType, star_handlers_registry

        by_name = {h.handler_name: h for h in star_handlers_registry}
        expected = {
            "inject_memories": EventType.OnLLMRequestEvent,
            "extract_memories": EventType.OnLLMResponseEvent,
            "on_agent_begin": EventType.OnAgentBeginEvent,
            "on_agent_done": EventType.OnAgentDoneEvent,
            "on_loaded": EventType.OnAstrBotLoadedEvent,
            "memory_group": EventType.AdapterMessageEvent,
            "cmd_list": EventType.AdapterMessageEvent,
            "cmd_search": EventType.AdapterMessageEvent,
            "cmd_rebuild": EventType.AdapterMessageEvent,
        }
        self.assertEqual(
            {name: by_name[name].event_type for name in expected}, expected
        )

        def filters(handler: str, kind: type) -> list:
            return [f for f in by_name[handler].event_filters if isinstance(f, kind)]

        self.assertTrue(
            any(
                f.group_name == "memory"
                for f in filters("memory_group", CommandGroupFilter)
            )
        )
        for handler, command in (
            ("cmd_list", "list"),
            ("cmd_search", "search"),
            ("cmd_rebuild", "rebuild"),
        ):
            self.assertTrue(
                any(
                    f.command_name == command
                    and list(f.parent_command_names) == ["memory"]
                    for f in filters(handler, CommandFilter)
                ),
                f"{handler} did not register /memory {command}",
            )


class AnalystContentTests(unittest.TestCase):
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
        self.assertNotIn("思考", assistant_text)
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
        maintenance = schema["maintenance"]
        self.assertEqual(maintenance["type"], "object")
        self.assertTrue(maintenance["collapsed"])
        self.assertEqual(maintenance["items"]["review_notify_umo"]["_special"], "umo")
        visible = {
            key
            for key, item in schema.items()
            if not item.get("invisible") and key != "maintenance"
        }

        for locale in ("zh-CN", "en-US"):
            cfg = json.loads(
                (root / ".astrbot-plugin" / "i18n" / f"{locale}.json").read_text(
                    encoding="utf-8"
                )
            )["config"]
            group = cfg["maintenance"]
            self.assertEqual(set(cfg.keys()), visible | {"maintenance"}, locale)
            self.assertTrue(group.get("description") and group.get("hint"), locale)
            self.assertEqual(
                set(group) - {"description", "hint"}, set(maintenance["items"]), locale
            )
            for key in visible:
                self.assertTrue(cfg[key].get("description"), f"{locale}:{key}")
                self.assertTrue(cfg[key].get("hint"), f"{locale}:{key}")
            for key, entry in maintenance["items"].items():
                self.assertTrue(group[key].get("description"), f"{locale}:{key}")
                self.assertTrue(group[key].get("hint"), f"{locale}:{key}")
                if "options" in entry:
                    self.assertEqual(
                        len(entry["options"]), len(group[key]["labels"]), locale
                    )


if __name__ == "__main__":
    unittest.main()
