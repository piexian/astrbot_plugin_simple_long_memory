"""后台采集、缓冲表与准入闸门真实链路测试：真实 AstrBot 4.30 栈端到端。

不模拟任何 AstrBot 接口：Context、ConversationManager、AstrMessageEvent、
Message、LLMResponse、ProviderRequest、MemoryPlugin、MemoryManager、
MaintenanceInbox 全部为真实实例，LLM 响应由标准库本地 HTTP 服务脚本化应答。
"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import yaml
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.core.agent.message import Message
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_runtime.subagents.context import SubAgentContext
from real_stack import (
    CHAT_PROVIDER_ID,
    RealStackTestCase,
    plugin_main,
    real_message_event,
)

from maintenance.inbox import INBOX_MAX_RETRIES, InboxRecord
from maintenance.version_gate import maintenance_supported, parse_release_tuple

main_module = plugin_main


def _now() -> datetime:
    return datetime(2026, 11, 27, 12, 0, tzinfo=timezone.utc)


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


class _RealChainCase(RealStackTestCase):
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
        await self.start_stack(
            with_decision=False,
            config=self._config(),
            with_plugin=True,
        )
        self.inbox = self.mgr.inbox
        self.assertIsNotNone(self.inbox)
        self.assertTrue(self.inbox.ready)

    async def _restart(self):
        """换一套真实 KB 与缓冲表，隔离同一用例内的多轮断言。"""
        self.server.shutdown()
        self.server.server_close()
        await self.start_stack(
            with_decision=False,
            config=self._config(),
            with_plugin=True,
        )
        self.inbox = self.mgr.inbox

    def _run_context(
        self, event, messages, *, end_reason="completed", is_subagent=False
    ):
        inner = (
            SubAgentContext(context=self.context, event=event, is_subagent=True)
            if is_subagent
            else None
        )
        return ContextWrapper(
            context=inner,
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
            event,
            self._run_context(event, messages, end_reason=end_reason),
            response,
        )

    async def _rows(self):
        return await self.inbox._fetch_rows(3)


class InboxCollectionRealChainTests(_RealChainCase):
    async def _single_turn(
        self, message_str="我最近搬到杭州工作了，做后端开发", **kwargs
    ):
        event = real_message_event(message_str=message_str)
        messages = [Message(role="user", content=message_str)]
        await self.plugin.on_agent_begin(event, self._run_context(event, messages))
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
        await self.plugin.on_agent_begin(
            prefixed, self._run_context(prefixed, messages)
        )
        await self._finish_turn(prefixed, messages)
        self.assertEqual(await self._rows(), [])

        # 普通文本不带前缀标记：同一词也保留
        plain = real_message_event(message_str="help")
        plain_messages = [Message(role="user", content="help")]
        await self.plugin.on_agent_begin(
            plain, self._run_context(plain, plain_messages)
        )
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
        await self.plugin.on_agent_begin(wakeup, self._run_context(wakeup, messages))
        await self._finish_turn(wakeup, messages)
        # 子代理轮次
        sub_event = real_message_event(message_str="子代理的中间过程")
        sub_messages = [Message(role="user", content="子代理的中间过程")]
        await self.plugin.on_agent_begin(
            sub_event,
            self._run_context(sub_event, sub_messages, is_subagent=True),
        )
        await self._finish_turn(sub_event, sub_messages, assistant_text="子代理回复")
        # 后台整理内部调用
        internal = real_message_event(message_str="后台整理内部轮次")
        internal._is_maintenance_internal = True
        internal_messages = [Message(role="user", content="后台整理内部轮次")]
        await self.plugin.on_agent_begin(
            internal, self._run_context(internal, internal_messages)
        )
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
        await self.plugin.on_agent_begin(event, self._run_context(event, messages))
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
        await self.plugin.on_agent_begin(event, self._run_context(event, messages))
        messages.append(
            Message(role="assistant", content="上午有例会，下午两点和设计对齐")
        )
        await self.plugin.on_agent_done(
            event,
            self._run_context(event, messages),
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
        await self.plugin.on_agent_begin(event, self._run_context(event, messages))
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


class InboxAggregationTests(_RealChainCase):
    async def _append(
        self,
        *,
        minutes_ago,
        text="聊" * 200,
        cid="c1",
        umo="qq:private:u1",
        sender_id="u1",
        sender_name="小明",
        brief="助手回复",
        realtime=False,
    ):
        record = InboxRecord(
            umo=umo,
            platform_id=umo.split(":", 1)[0],
            session_type="private",
            session_id=umo.rsplit(":", 1)[-1],
            conversation_id=cid,
            sender_id=sender_id,
            sender_name=sender_name,
            user_text=text,
            assistant_brief=brief,
            created_at=(_now() - timedelta(minutes=minutes_ago)).isoformat(),
        )
        row_id = await self.inbox.append(record)
        self.assertIsNotNone(row_id)
        if realtime:
            self.assertTrue(await self.inbox.mark_realtime_extracted(row_id))
        return row_id

    async def _collect(self, **overrides):
        params = {
            "gap_minutes": 30,
            "max_chars": 8000,
            "min_user_chars": 50,
            "max_blocks": 20,
            "commit": True,
            "now": _now(),
        }
        params.update(overrides)
        return await self.inbox.collect_blocks(**params)

    async def test_time_gap_and_char_cap_split_blocks(self):
        await self._append(minutes_ago=90)
        await self._append(minutes_ago=40)  # 与上行间隔 50 分钟，切新块
        await self._append(minutes_ago=35)  # 与上行同窗，并入
        result = await self._collect()
        blocks = result["blocks"]
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].row_ids, [1])
        self.assertEqual(blocks[1].row_ids, [2, 3])
        self.assertEqual(blocks[1].conv_key, "qq:private:u1:c1")
        self.assertEqual(result["stats"]["blocks_emitted"], 2)

        # 字符上限独立于时间窗：连续轮次超上限同样切块
        await self._restart()
        for _ in range(3):
            await self._append(minutes_ago=60, text="聊" * 300)
        result2 = await self._collect(max_chars=500)
        self.assertEqual(len(result2["blocks"]), 3)
        self.assertTrue(all(b.char_count <= 500 for b in result2["blocks"]))

    async def test_quiet_window_holds_recent_tail(self):
        await self._append(minutes_ago=90)
        await self._append(minutes_ago=5)  # 距当前不足 10 分钟静默阈值
        result = await self._collect()
        self.assertEqual(len(result["blocks"]), 1)
        self.assertEqual(result["stats"]["blocks_held_quiet"], 1)
        # 被保留的尾块与已产出块都保持 pending，前者等待下周期合并
        counts = await self.inbox.status_counts()
        self.assertEqual(counts.get("pending"), 2)

        # 尾块已达字符上限时不适用静默保留
        await self._restart()
        await self._append(minutes_ago=5, text="聊" * 600)
        result2 = await self._collect(max_chars=500)
        self.assertEqual(len(result2["blocks"]), 1)
        self.assertEqual(result2["stats"]["blocks_held_quiet"], 0)

    async def test_block_cap_leaves_remaining_rows_pending(self):
        for n in range(3):
            await self._append(minutes_ago=60, umo=f"qq:private:u{n}")
        result = await self._collect(max_blocks=2)
        self.assertEqual(len(result["blocks"]), 2)
        self.assertTrue(result["stats"]["block_cap_reached"])
        counts = await self.inbox.status_counts()
        # collect 只标 skipped 终态；已产出与未轮到的行都保持 pending，
        # 由 Runner 在操作完成后标记 processed
        self.assertEqual(counts.get("pending"), 3)

    async def test_invalid_umo_rows_are_skipped(self):
        await self._append(minutes_ago=60, umo="not-a-valid-umo")
        await self._append(minutes_ago=60, umo="qq:private:u1")
        result = await self._collect()
        self.assertEqual(result["stats"]["invalid_umo_skipped"], 1)
        self.assertEqual(len(result["blocks"]), 1)
        counts = await self.inbox.status_counts()
        self.assertEqual(counts.get("skipped"), 1)
        self.assertEqual(counts.get("pending"), 1)

    async def test_short_block_is_skipped_only_in_commit_mode(self):
        await self._append(minutes_ago=60, text="嗯嗯")
        result = await self._collect()
        self.assertEqual(result["stats"]["blocks_skipped_short"], 1)
        self.assertEqual(result["blocks"], [])
        counts = await self.inbox.status_counts()
        self.assertEqual(counts.get("skipped"), 1)

        await self._restart()
        await self._append(minutes_ago=60, text="嗯嗯")
        dry = await self._collect(commit=False)
        self.assertEqual(dry["stats"]["blocks_skipped_short"], 1)
        counts2 = await self.inbox.status_counts()
        self.assertEqual(counts2.get("pending"), 1)  # dry-run 不写行状态

    async def test_size_split_tail_bypasses_short_filter(self):
        await self._append(minutes_ago=60, text="聊" * 200)
        await self._append(minutes_ago=59, text="尾" * 10)

        result = await self._collect(max_chars=300, min_user_chars=50)

        self.assertEqual(len(result["blocks"]), 2)
        self.assertEqual(result["blocks"][1].row_ids, [2])
        self.assertEqual(result["stats"]["blocks_skipped_short"], 0)

    async def test_block_text_keeps_sender_labels_and_realtime_flag(self):
        text = "用户喜欢美式咖啡，每天早上都要喝两杯，周末还会去店里手冲单品豆，最近换了深烘的肯尼亚豆，还说以后想学着自己在家里烘豆子。"
        await self._append(
            minutes_ago=60,
            text=text,
            sender_id="10086",
            sender_name="小明",
            brief="好的，记住了",
            realtime=True,
        )
        result = await self._collect()
        block = result["blocks"][0]
        stamp = (_now() - timedelta(minutes=60)).strftime("%m-%d %H:%M")
        self.assertIn(f"[{stamp}] 用户(sender_id=10086, 小明): {text}", block.text)
        self.assertIn(f"[{stamp}] 助手: 好的，记住了", block.text)
        self.assertEqual(block.sender_ids, ["10086"])
        self.assertTrue(block.realtime_extracted)

    async def test_groups_are_separated_by_umo_and_conversation(self):
        # 交错插入两个会话，SQL 排序须保证同组行连续
        await self._append(minutes_ago=60, umo="qq:private:u1", cid="c1")
        await self._append(minutes_ago=60, umo="qq:private:u2", cid="c9")
        await self._append(minutes_ago=60, umo="qq:private:u1", cid="c1")
        result = await self._collect()
        keys = [b.conv_key for b in result["blocks"]]
        self.assertEqual(keys, ["qq:private:u1:c1", "qq:private:u2:c9"])
        self.assertEqual(result["blocks"][0].message_count, 2)

    async def test_failed_rows_retry_until_cap_then_stay_out(self):
        row_id = await self._append(minutes_ago=60)
        await self.inbox.mark_failed([row_id])
        await self.inbox.mark_failed([row_id])
        # 重试额度内：failed 行继续参与提取
        fetched = await self._collect()
        self.assertEqual(fetched["stats"]["rows_fetched"], 1)
        counts = await self.inbox.status_counts()
        self.assertEqual(counts.get("failed"), 1)

        await self.inbox.mark_failed([row_id])  # 第 3 次失败，重试额度用尽
        exhausted = await self._collect()
        self.assertEqual(exhausted["stats"]["rows_fetched"], 0)
        self.assertEqual(exhausted["stats"]["blocks_emitted"], 0)


class InboxPurgeTests(_RealChainCase):
    async def _seed(self, *, days_ago, status=None, retries=0, text="聊" * 200):
        created_at = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
        record = InboxRecord(
            umo="qq:private:u1",
            platform_id="qq",
            session_type="private",
            session_id="u1",
            conversation_id="c1",
            sender_id="u1",
            sender_name="小明",
            user_text=text,
            assistant_brief="",
            created_at=created_at,
        )
        row_id = await self.inbox.append(record)
        self.assertIsNotNone(row_id)
        for _ in range(retries):
            self.assertTrue(await self.inbox.mark_failed([row_id]))
        if status == "processed":
            self.assertTrue(await self.inbox.mark_rows([row_id], "processed"))
        elif status == "skipped":
            self.assertTrue(await self.inbox.mark_rows([row_id], "skipped"))
        return row_id

    async def test_purge_removes_only_terminal_rows_past_retention(self):
        old_processed = await self._seed(days_ago=15, status="processed")
        old_skipped = await self._seed(days_ago=15, status="skipped")
        old_pending = await self._seed(days_ago=15)
        old_failed = await self._seed(days_ago=15, retries=INBOX_MAX_RETRIES)
        retrying_failed = await self._seed(days_ago=15, retries=1)
        fresh_processed = await self._seed(days_ago=1, status="processed")

        result = await self.inbox.purge_terminal()
        self.assertEqual(result["candidates"], 3)
        self.assertEqual(result["purged"], 3)
        counts = await self.inbox.status_counts()
        # pending 与重试中的 failed 超期不删；新近终态保留
        self.assertEqual(counts.get("pending"), 1)
        self.assertEqual(counts.get("failed"), 1)
        self.assertEqual(counts.get("processed"), 1)
        kept = {old_pending, retrying_failed, fresh_processed}
        self.assertNotIn(old_processed, kept)
        self.assertNotIn(old_skipped, kept)
        self.assertNotIn(old_failed, kept)

    async def test_purge_dry_run_counts_without_deleting(self):
        await self._seed(days_ago=15, status="processed")
        result = await self.inbox.purge_terminal(dry_run=True)
        self.assertEqual(result["candidates"], 1)
        self.assertEqual(result["purged"], 0)
        counts = await self.inbox.status_counts()
        self.assertEqual(counts.get("processed"), 1)


class MemoryManagerPurgeInboxTests(_RealChainCase):
    async def test_purge_deprecated_cleans_terminal_inbox_rows(self):
        created_at = (datetime.now(timezone.utc) - timedelta(days=15)).isoformat()

        async def _seed(status):
            record = InboxRecord(
                umo="qq:private:u1",
                platform_id="qq",
                session_type="private",
                session_id="u1",
                conversation_id="c1",
                sender_id="u1",
                sender_name="小明",
                user_text="聊" * 200,
                assistant_brief="",
                created_at=created_at,
            )
            row_id = await self.inbox.append(record)
            if status:
                self.assertTrue(await self.inbox.mark_rows([row_id], status))
            return row_id

        await _seed("processed")
        await _seed(None)  # pending 行超期不删

        result = await self.mgr.purge_deprecated(after_days=7)

        self.assertEqual(result["inbox"]["purged"], 1)
        counts = await self.inbox.status_counts()
        self.assertEqual(counts.get("pending"), 1)
        self.assertNotIn("processed", counts)


class MaintenanceVersionGateTests(RealStackTestCase):
    async def asyncSetUp(self):
        await self.start_stack(
            with_decision=False,
            config={"maintenance_enabled": True},
            with_plugin=True,
        )

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

    async def test_maintenance_test_entry_is_blocked_below_430(self):
        self.assertTrue(self.mgr.is_kb_connected)
        with (
            patch.object(main_module, "maintenance_supported", return_value=False),
            patch.object(main_module, "MaintenanceScheduler") as scheduler_cls,
        ):
            report = await self.plugin._run_maintenance_test("extract")
        self.assertIn("失败", report)
        self.assertIn("4.30.0", report)
        scheduler_cls.assert_not_called()

    async def test_scheduler_start_is_blocked_below_430(self):
        with patch.object(main_module, "maintenance_supported", return_value=False):
            await self.plugin._start_scheduler()
        self.assertIsNone(self.plugin._scheduler)


if __name__ == "__main__":
    unittest.main()
