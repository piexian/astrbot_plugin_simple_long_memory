"""后台采集与准入闸门真实链路测试：真实 AstrBot 4.30 栈端到端。

不模拟任何 AstrBot 接口：Context、ConversationManager、AstrMessageEvent、
Message、LLMResponse、ProviderRequest、MemoryPlugin、MaintenanceInbox
全部为真实实例，LLM 响应由标准库本地 HTTP 服务脚本化应答。
"""

from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.core.agent.message import Message
from real_stack import (
    CHAT_PROVIDER_ID,
    build_real_context,
    real_message_event,
    start_server,
)
from test_maintenance_extract import _real_inbox
from test_memory_commands import MemoryManager, plugin_main

from maintenance.version_gate import maintenance_supported, parse_release_tuple

main_module = plugin_main
MemoryPlugin = plugin_main.MemoryPlugin


def _gate_chat_payload(candidate_id, *, durable=0.95, risk=0.05):
    return json.dumps(
        {
            "results": [
                {
                    "id": candidate_id,
                    "has_durable_value": durable,
                    "persona_or_behavior_control": risk,
                    "emotion_or_grudge": risk,
                    "ephemeral_or_system_noise": risk,
                }
            ]
        },
        ensure_ascii=False,
    )


class _RealChainCase(unittest.IsolatedAsyncioTestCase):
    """真实 Context + 真实事件 + 真实缓冲表的公共装配方。"""

    def _config(self):
        return {
            "maintenance_enabled": True,
            "auto_memorize": False,
            "extraction_interval": 1,
            "extraction_min_content_length": 1,
            "extraction_provider_id": CHAT_PROVIDER_ID,
        }

    async def asyncSetUp(self):
        self.server = start_server()
        self.context = await build_real_context(self.server, with_decision=False)
        self.inbox, _storage = await _real_inbox(self)
        self.assertTrue(await self.inbox.ensure_table())
        self.plugin = MemoryPlugin(self.context, self._config())
        self.plugin.memory_mgr = MemoryManager(
            kb_mgr=self.context.kb_manager, config=self.plugin.config
        )
        self.plugin.memory_mgr.inbox = self.inbox

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def _run_context(self, messages, *, end_reason="completed", is_subagent=False):
        return SimpleNamespace(
            context=SimpleNamespace(is_subagent=is_subagent),
            messages=messages,
            run_end_reason=end_reason,
        )

    async def _finish_turn(
        self,
        event,
        messages,
        *,
        assistant_text="好的，记住了",
        end_reason="completed",
        role="assistant",
    ):
        response = LLMResponse(role=role, completion_text=assistant_text)
        await self.plugin.on_agent_done(
            event, self._run_context(messages, end_reason=end_reason), response
        )

    async def _rows(self):
        return await self.inbox._fetch_rows(3)


class InboxCollectionRealChainTests(_RealChainCase):
    async def _single_turn(
        self, message_str="我最近搬到杭州工作了，做后端开发", **kwargs
    ):
        event = real_message_event(message_str=message_str)
        messages = [Message(role="user", content=message_str)]
        await self.plugin.on_agent_begin(event, self._run_context(messages))
        await self._finish_turn(event, messages, **kwargs)
        return event

    async def test_normal_turn_appends_row_with_trusted_fields(self):
        event = await self._single_turn(
            assistant_text="好的，祝你工作顺利，杭州是个好城市"
        )
        rows = await self._rows()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.umo, event.unified_msg_origin)
        self.assertEqual(row.platform_id, "webchat")
        self.assertEqual(row.session_type, "private")
        self.assertEqual(row.session_id, "tester")
        self.assertEqual(row.sender_id, "tester")
        self.assertEqual(row.sender_name, "测试用户")
        self.assertEqual(row.user_text, "我最近搬到杭州工作了，做后端开发")
        self.assertEqual(row.assistant_brief, "好的，祝你工作顺利，杭州是个好城市")
        self.assertFalse(row.realtime_extracted)

    async def test_command_turn_is_not_collected(self):
        await self._single_turn(message_str="/memory status")
        await self._single_turn(message_str="memory review 3")
        self.assertEqual(await self._rows(), [])

    async def test_prefix_consumed_command_turn_is_not_collected(self):
        # 核心剥离 "/" 后 help 等其他命令靠已消费前缀标记识别
        prefixed = real_message_event(message_str="help")
        prefixed.set_extra("_consumed_wake_prefix", "/")
        messages = [Message(role="user", content="help")]
        await self.plugin.on_agent_begin(prefixed, self._run_context(messages))
        await self._finish_turn(prefixed, messages)
        self.assertEqual(await self._rows(), [])

        # 普通文本不带前缀标记：同一词也保留
        plain = real_message_event(message_str="help")
        plain_messages = [Message(role="user", content="help")]
        await self.plugin.on_agent_begin(plain, self._run_context(plain_messages))
        await self._finish_turn(plain, plain_messages)
        rows = await self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].user_text, "help")

    async def test_low_info_turn_is_not_collected(self):
        # 用户文本过短且内容几乎全在助手输出：单向回显不入缓冲
        await self._single_turn(message_str="嗯", assistant_text="好的")
        self.assertEqual(await self._rows(), [])

        await self._single_turn(
            message_str="好的",
            assistant_text="这是一段非常长的助手输出" + "内容" * 120,
        )
        self.assertEqual(await self._rows(), [])

    async def test_excluded_turn_kinds_are_not_collected(self):
        # wakeup 事件
        wakeup = real_message_event(message_str="定时任务输出")
        wakeup.event_kind = "agent_wakeup"
        messages = [Message(role="user", content="定时任务输出")]
        await self.plugin.on_agent_begin(wakeup, self._run_context(messages))
        await self._finish_turn(wakeup, messages)
        # 子代理轮次
        sub_event = real_message_event(message_str="子代理的中间过程")
        sub_messages = [Message(role="user", content="子代理的中间过程")]
        await self.plugin.on_agent_begin(
            sub_event, self._run_context(sub_messages, is_subagent=True)
        )
        await self._finish_turn(sub_event, sub_messages, assistant_text="子代理回复")
        # 后台整理内部调用
        internal = real_message_event(message_str="后台整理内部轮次")
        internal._is_maintenance_internal = True
        internal_messages = [Message(role="user", content="后台整理内部轮次")]
        await self.plugin.on_agent_begin(internal, self._run_context(internal_messages))
        await self._finish_turn(internal, internal_messages)
        self.assertEqual(await self._rows(), [])

    async def test_cancelled_and_error_turns_are_not_collected(self):
        await self._single_turn(end_reason="cancelled")
        await self._single_turn(end_reason="error")
        await self._single_turn(role="err", assistant_text="")
        self.assertEqual(await self._rows(), [])

    async def test_steering_followup_keeps_user_text_order(self):
        event = real_message_event(message_str="帮我看看这个报错怎么解决")
        messages = [Message(role="user", content="帮我看看这个报错怎么解决")]
        await self.plugin.on_agent_begin(event, self._run_context(messages))
        messages.append(Message(role="assistant", content="看起来是配置问题"))
        messages.append(Message(role="user", content="对，我把超时时间改大到 30 秒了"))
        messages.append(Message(role="assistant", content="这样应该能缓解"))
        await self._finish_turn(event, messages, assistant_text="改完再观察一下")
        rows = await self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0].user_text,
            "帮我看看这个报错怎么解决\n对，我把超时时间改大到 30 秒了",
        )
        self.assertEqual(rows[0].assistant_brief, "改完再观察一下")

    async def test_assistant_brief_is_capped(self):
        long_reply = "长回复" * 300
        await self._single_turn(assistant_text=long_reply)
        rows = await self._rows()
        self.assertEqual(len(rows), 1)
        self.assertGreater(len(long_reply), 300)
        self.assertEqual(len(rows[0].assistant_brief), 300)
        self.assertEqual(rows[0].assistant_brief, long_reply[:300])

    async def test_empty_response_falls_back_to_collected_assistant_output(self):
        # response 为空但轮内已有助手回复：brief 回退到已收集内容，不为空
        event = real_message_event(message_str="帮我总结一下今天的安排")
        messages = [Message(role="user", content="帮我总结一下今天的安排")]
        await self.plugin.on_agent_begin(event, self._run_context(messages))
        messages.append(
            Message(role="assistant", content="上午有例会，下午两点和设计对齐")
        )
        await self.plugin.on_agent_done(
            event,
            self._run_context(messages),
            LLMResponse(role="assistant", completion_text=""),
        )
        rows = await self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].assistant_brief, "上午有例会，下午两点和设计对齐")

    async def test_collection_without_auto_memorize_makes_no_llm_calls(self):
        # maintenance_enabled=true、auto_memorize=false：只采集，不触发实时提取
        await self._single_turn()
        rows = await self._rows()
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0].realtime_extracted)
        self.assertEqual(self.server.request_log, [])

    async def test_conversation_manager_stays_read_only(self):
        event = await self._single_turn()
        rows = await self._rows()
        self.assertEqual(rows[0].conversation_id, "")
        conversations = await self.context.conversation_manager.get_conversations(
            event.unified_msg_origin
        )
        self.assertEqual(conversations, [])


class RealtimeExtractionMarkTests(_RealChainCase):
    def _config(self):
        config = super()._config()
        config["auto_memorize"] = True
        return config

    async def _turn_with_request(self, message_str):
        event = real_message_event(message_str=message_str)
        request = ProviderRequest(prompt=message_str)
        self.plugin._accumulate_request_snapshot(event, request)
        messages = [Message(role="user", content=message_str)]
        await self.plugin.on_agent_begin(event, self._run_context(messages))
        await self._finish_turn(event, messages)
        return event

    async def test_completed_extraction_marks_row_including_empty_result(self):
        self.server.queue_chat("[]")  # 模型返回空数组也算完成一次提取
        await self._turn_with_request("我最近搬到杭州工作了，做后端开发")
        rows = await self._rows()
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0].realtime_extracted)
        chat_kinds = [r["kind"] for r in self.server.request_log]
        self.assertEqual(chat_kinds, ["chat"])

    async def test_row_not_marked_when_model_not_called(self):
        # 未达提取间隔：模型未被调用，行保持 realtime_extracted=0
        self.plugin.config["extraction_interval"] = 5
        await self._turn_with_request("我最近搬到杭州工作了，做后端开发")
        rows = await self._rows()
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0].realtime_extracted)
        self.assertEqual(self.server.request_log, [])


class MemoryStoreToolGateRealChainTests(_RealChainCase):
    async def _gate(self, content="用户每周三都去打羽毛球"):
        event = real_message_event()
        return await self.plugin._gate_tool_content(event, content), event

    async def test_approved_content_returns_none(self):
        self.server.queue_chat(_gate_chat_payload("tool"))
        result, _ = await self._gate()
        self.assertIsNone(result)
        self.assertEqual([r["kind"] for r in self.server.request_log], ["chat"])

    async def test_rejected_content_returns_explicit_error(self):
        self.server.queue_chat(_gate_chat_payload("tool", durable=0.2))
        result, _ = await self._gate()
        self.assertEqual(
            result,
            "Error: memory rejected by admission gate (rejected:has_durable_value)",
        )

        self.server.queue_chat(_gate_chat_payload("tool", risk=0.9))
        result2, _ = await self._gate()
        self.assertIn("persona_or_behavior_control", result2)
        self.assertTrue(result2.startswith("Error: memory rejected"))

    async def test_unavailable_gate_fails_closed_with_retry_hint(self):
        self.server.queue_chat("这不是 JSON")
        result, _ = await self._gate()
        self.assertEqual(
            result, "Error: memory gate temporarily unavailable, please retry later"
        )


class MaintenanceVersionGateTests(unittest.TestCase):
    def test_real_beta_version_is_supported(self):
        import astrbot

        # 本地 AstrBot 为真实 4.30.0-beta.2：预发布后缀不参与比较
        self.assertEqual(parse_release_tuple("4.30.0-beta.2"), (4, 30, 0))
        self.assertTrue(maintenance_supported("4.30.0-beta.2"))
        self.assertTrue(maintenance_supported(astrbot.__version__))
        self.assertTrue(maintenance_supported())

    def test_429_is_rejected_with_reason(self):
        self.assertFalse(maintenance_supported("4.29.4"))
        reason = main_module.unsupported_maintenance_reason("4.29.4")
        self.assertIn("4.30.0", reason)
        self.assertIn("4.29.4", reason)

    def test_metadata_keeps_4294_baseline(self):
        metadata = yaml.safe_load(
            (Path(__file__).resolve().parent.parent / "metadata.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(metadata["astrbot_version"], ">=4.29.4")

    def test_maintenance_test_entry_is_blocked_below_430(self):
        plugin = MemoryPlugin(None, {})
        with (
            patch.object(main_module, "maintenance_supported", return_value=False),
            patch.object(main_module, "MaintenanceScheduler") as scheduler_cls,
        ):
            report = asyncio_run(plugin._run_maintenance_test("extract"))
        self.assertIn("失败", report)
        self.assertIn("4.30.0", report)
        scheduler_cls.assert_not_called()

    def test_scheduler_start_is_blocked_below_430(self):
        plugin = MemoryPlugin(None, {"maintenance_enabled": True})
        plugin.memory_mgr = SimpleNamespace(is_kb_connected=True)
        with patch.object(main_module, "maintenance_supported", return_value=False):
            asyncio_run(plugin._start_scheduler())
        self.assertIsNone(plugin._scheduler)


def asyncio_run(coro):
    return asyncio.run(coro)


if __name__ == "__main__":
    unittest.main()
