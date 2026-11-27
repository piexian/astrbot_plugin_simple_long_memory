"""审核卡片视图契约与 ui_action 回调的真实 AstrBot 栈回归测试。"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import unittest
import uuid
from pathlib import Path

import yaml
from astrbot.api.interactive_ui import ViewSpec
from astrbot.core.star.filter.ui_action import UIActionFilter
from astrbot.core.star.star_handler import EventType, star_handlers_registry
from real_stack import build_real_context, real_message_event, start_server
from test_memory_commands import MemoryManager, plugin_main

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIEW_DIR = PROJECT_ROOT / "views" / "memory_review"

VIEW_ACTIONS = {"approve", "reject"}


def _registered_view_actions() -> set[str]:
    """收集本插件通过真实注册表登记的 ui_action 回调 ID。"""
    handlers = star_handlers_registry.get_handlers_by_event_type(
        EventType.AdapterMessageEvent, only_activated=False
    )
    actions = set()
    for handler in handlers:
        if not handler.handler_module_path.endswith("simple_long_memory.main"):
            continue
        for event_filter in handler.event_filters:
            if isinstance(event_filter, UIActionFilter):
                actions.add(event_filter.action_id)
    return actions


class ReviewViewContractTests(unittest.TestCase):
    """schema 折叠组、metadata 视图声明、注册回调与视图资源的静态契约。"""

    def test_schema_has_single_visible_maintenance_group(self):
        schema = json.loads(
            (PROJECT_ROOT / "_conf_schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(schema["maintenance"]["type"], "object")
        self.assertTrue(schema["maintenance"]["collapsed"])
        visible = [
            key
            for key, item in schema.items()
            if not item.get("invisible") and key != "maintenance"
        ]
        for key in visible:
            self.assertFalse(key.startswith("maintenance_"), key)
        # 废弃键不再出现在 schema 中，交由核心清理旧配置
        for legacy in (
            "maintenance_segmenter_model_id",
            "maintenance_extract_conv2_enabled",
            "maintenance_extract_conv2_chunk_messages",
            "maintenance_segment_max_extensions",
            "maintenance_tasks",
        ):
            self.assertNotIn(legacy, schema)
        # 旧顶层键仅作为 invisible 迁移入口存在
        self.assertTrue(
            schema["maintenance_enabled"].get("invisible"),
        )
        self.assertTrue(schema["maintenance_config_migrated"].get("invisible"))
        # 对象内 _special 提示保留
        # 对象内 _special 提示保留
        items = schema["maintenance"]["items"]
        self.assertEqual(items["model_id"]["_special"], "select_provider")
        self.assertEqual(items["review_notify_umo"]["_special"], "umo")
        # 巡检与 merge 采样是两个独立键，保留各自默认行为
        self.assertEqual(items["organizer_batch_size"]["default"], 30)
        self.assertEqual(items["organizer_merge_batch_size"]["default"], 200)
        # 已确认无代码读取的预留键在折叠组内也保持不可见
        for reserved in (
            "context_max_rounds",
            "context_max_chars",
            "context_max_age_days",
            "review_notify_schedule",
            "review_notify_time",
        ):
            self.assertTrue(items[reserved].get("invisible"), reserved)

    def test_metadata_view_spec_matches_registered_actions(self):
        meta = yaml.safe_load(
            (PROJECT_ROOT / "metadata.yaml").read_text(encoding="utf-8")
        )
        specs = [ViewSpec.model_validate(view) for view in meta["views"]]
        self.assertEqual(len(specs), 1)
        spec = specs[0]
        self.assertEqual(spec.name, plugin_main.REVIEW_VIEW_NAME)
        self.assertEqual(spec.target, "chat_message")
        self.assertEqual(set(spec.actions), VIEW_ACTIONS)
        # 入口文件必须真实存在
        self.assertTrue((PROJECT_ROOT / spec.entry).is_file())
        self.assertEqual(_registered_view_actions(), VIEW_ACTIONS)

    def test_view_js_uses_official_bridge_and_safe_dom(self):
        script = (VIEW_DIR / "app.js").read_text(encoding="utf-8")
        self.assertIn("window.AstrBotPluginView", script)
        self.assertIn("bridge.ready()", script)
        self.assertIn("bridge.view.subscribe", script)
        self.assertIn("bridge.view.action", script)
        for action in VIEW_ACTIONS:
            self.assertIn(f'"{action}"', script)
        self.assertIn("textContent", script)
        self.assertIn("disabled", script)
        # 桥接初始化失败必须落到错误区域，不允许未处理的 Promise
        self.assertIn("start().catch", script)
        for banned in (
            "innerHTML",
            "insertAdjacentHTML",
            "document.write",
            "postMessage",
            "fetch(",
        ):
            self.assertNotIn(banned, script)
        html = (VIEW_DIR / "index.html").read_text(encoding="utf-8")
        self.assertIn("./app.js", html)
        node = shutil.which("node")
        if node:
            result = subprocess.run(
                [node, "--check", str(VIEW_DIR / "app.js")],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_i18n_covers_view_and_object_items(self):
        for locale in ("zh-CN", "en-US"):
            data = json.loads(
                (
                    PROJECT_ROOT / ".astrbot-plugin" / "i18n" / f"{locale}.json"
                ).read_text(encoding="utf-8")
            )
            view_text = data["views"][plugin_main.REVIEW_VIEW_NAME]
            for key in (
                "title",
                "description",
                "button_approve",
                "button_reject",
                "empty",
                "action_failed",
            ):
                self.assertTrue(str(view_text.get(key, "")).strip(), f"{locale}:{key}")


class ReviewViewRealStackTests(unittest.IsolatedAsyncioTestCase):
    """真实 Context + 真实 KB + 真实事件下的卡片与回调行为。"""

    def _queue_item(self, item_id: int, op: dict) -> dict:
        return {
            "id": item_id,
            "session_id": "test-session",
            "op_type": op.get("type", ""),
            "op": op,
            "op_signature": f"sig-{item_id}",
            "verdict_reason": "审核证据不足",
            "controversial": True,
            "created_at": "2026-11-30 10:00:00",
            "status": "pending",
        }

    async def asyncSetUp(self):
        self.server = start_server()
        self.kb_name = f"review-{uuid.uuid4().hex[:10]}"
        self.context = await build_real_context(
            self.server, with_decision=False, kb_name=self.kb_name
        )
        self.mgr = MemoryManager(
            kb_mgr=self.context.kb_manager, config={"kb_name": self.kb_name}
        )
        self.mgr.initialize()
        await self.mgr.connect_kb()
        self.plugin = plugin_main.MemoryPlugin(
            self.context,
            {
                "kb_name": [self.kb_name],
                "maintenance_enabled": False,
                "auto_memorize": False,
            },
        )
        self.plugin.plugin_id = f"memory-review-{uuid.uuid4().hex[:12]}"
        self.plugin.memory_mgr = self.mgr
        # 走真实启动路径：maintenance_enabled 关闭时不创建后台循环
        await self.plugin._start_scheduler()
        self.assertIsNotNone(self.plugin._scheduler)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    async def _seed_queue(self, *items: dict) -> None:
        await self.plugin.put_kv_data("maintenance_pending_review", list(items))

    async def _queue(self) -> list:
        return await self.plugin.get_kv_data("maintenance_pending_review", None) or []

    def _admin_event(self, **kwargs):
        event = real_message_event(**kwargs)
        event.role = "admin"
        return event

    def _interaction_event(self, action_id: str, target_id: int):
        from astrbot.core.message.components import UIInteraction, ViewContext

        event = self._admin_event()
        event.message_obj.message = [
            UIInteraction(
                action_id=action_id,
                interaction_id=f"it-{uuid.uuid4().hex[:8]}",
                data={"id": target_id},
                view=ViewContext(
                    id="view-1",
                    name=plugin_main.REVIEW_VIEW_NAME,
                    version=1,
                    revision=1,
                    data={"items": []},
                ),
            )
        ]
        return event

    async def _collect(self, generator):
        return [result async for result in generator]

    async def test_chatui_event_yields_real_view_result(self):
        from astrbot.api.interactive_ui import ViewResult

        op = {
            "type": "create",
            "content": "用户喜欢拿铁",
            "umo": "qq:private:u1",
            "subject": "u1",
        }
        await self._seed_queue(self._queue_item(1, op))
        event = self._admin_event(
            message_str="memory review list", support_ui_interaction=True
        )
        results = await self._collect(self.plugin.cmd_review(event))
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertIsInstance(result, ViewResult)
        self.assertEqual(result.name, plugin_main.REVIEW_VIEW_NAME)
        self.assertEqual(result.title, "记忆待审列表")
        # 文本回退必须保留 approve/reject 用法说明
        self.assertIn("/memory review approve <id>", result.text)
        self.assertIn("/memory review reject <id>", result.text)
        self.assertEqual(result.data["items"][0]["id"], 1)
        self.assertEqual(result.data["items"][0]["op_type"], "create")
        self.assertEqual(result.data["items"][0]["summary"], "用户喜欢拿铁")

    async def test_plain_platform_yields_text_listing(self):
        from astrbot.api.interactive_ui import ViewResult

        op = {"type": "archive", "uri": "facts://old", "reason": "过时"}
        await self._seed_queue(self._queue_item(2, op))
        event = self._admin_event(message_str="memory review list")
        results = await self._collect(self.plugin.cmd_review(event))
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertNotIsInstance(result, ViewResult)
        text = result.chain[0].text
        self.assertIn("待审争议 (1 条)", text)
        self.assertIn("#2", text)
        self.assertIn("archive", text)
        self.assertIn("/memory review approve <id>", text)

    async def test_overflow_is_visible_and_clearable(self):
        await self._seed_queue()
        await self.plugin.put_kv_data(
            "maintenance_pending_review_overflow",
            [{"op": {"type": "archive", "uri": "facts://overflow"}}],
        )

        listed = await self._collect(
            self.plugin.cmd_review(self._admin_event(message_str="memory review list"))
        )
        self.assertIn("1 条操作等待队列容量释放", listed[0].chain[0].text)

        cleared = await self._collect(
            self.plugin.cmd_review(self._admin_event(message_str="memory review clear"))
        )
        self.assertIn("溢出暂存已清空", cleared[0].chain[0].text)
        self.assertEqual(await self._queue(), [])
        self.assertEqual(
            await self.plugin.get_kv_data("maintenance_pending_review_overflow", None),
            [],
        )

    async def test_ui_action_non_admin_is_rejected(self):
        op = {
            "type": "create",
            "content": "用户喜欢拿铁",
            "umo": "qq:private:u1",
            "subject": "u1",
        }
        await self._seed_queue(self._queue_item(1, op))
        event = self._interaction_event("approve", 1)
        event.role = "member"
        with self.assertRaises(PermissionError):
            await self._collect(self.plugin.review_approve_action(event))
        queue = await self._queue()
        self.assertEqual(queue[0]["status"], "pending")

    async def test_ui_action_approve_success_updates_card_and_kb(self):
        op = {
            "type": "create",
            "content": "用户喜欢拿铁",
            "umo": "qq:private:u1",
            "subject": "u1",
        }
        await self._seed_queue(self._queue_item(1, op))
        event = self._interaction_event("approve", 1)
        results = await self._collect(self.plugin.review_approve_action(event))
        self.assertEqual(len(results), 1)
        result = results[0]
        # 更新卡片：不带 name/text/title
        self.assertIsNone(result.name)
        self.assertEqual(result.text, "")
        self.assertEqual(result.title, "")
        self.assertEqual(result.data["items"], [])
        self.assertIn("已批准并执行", result.data["notice"])

        queue = await self._queue()
        self.assertEqual(queue[0]["status"], "approved")
        docs = await self.mgr.vec_db.document_storage.get_documents(
            metadata_filters={"is_memory_record": True, "deprecated": False},
            limit=10,
            offset=0,
        )
        self.assertEqual(len(docs), 1)
        self.assertIn("拿铁", docs[0]["text"])

    async def test_ui_action_approve_failure_keeps_pending_for_retry(self):
        op = {"type": "create", "content": "", "umo": "qq:private:u1"}
        await self._seed_queue(self._queue_item(3, op))
        event = self._interaction_event("approve", 3)
        results = await self._collect(self.plugin.review_approve_action(event))
        result = results[0]
        self.assertIsNone(result.name)
        self.assertIn("执行失败", result.data["error"])
        # 失败不标 approved：条目保留 pending 可重试
        self.assertEqual(result.data["items"][0]["id"], 3)
        queue = await self._queue()
        self.assertEqual(queue[0]["status"], "pending")

    async def test_ui_action_reject_removes_item_from_card(self):
        op = {"type": "archive", "uri": "facts://old", "reason": "过时"}
        await self._seed_queue(self._queue_item(4, op))
        event = self._interaction_event("reject", 4)
        results = await self._collect(self.plugin.review_reject_action(event))
        result = results[0]
        self.assertIn("已废案", result.data["notice"])
        self.assertEqual(result.data["items"], [])
        queue = await self._queue()
        self.assertEqual(queue[0]["status"], "rejected")

    async def test_ui_action_missing_item_reports_notice_with_fresh_list(self):
        op = {"type": "archive", "uri": "facts://old", "reason": "过时"}
        await self._seed_queue(self._queue_item(5, op))
        event = self._interaction_event("approve", 999)
        results = await self._collect(self.plugin.review_approve_action(event))
        result = results[0]
        self.assertIn("未找到待审项 #999", result.data["notice"])
        # 已处理/不存在的条目：卡片按最新队列刷新
        self.assertEqual(result.data["items"][0]["id"], 5)
        queue = await self._queue()
        self.assertEqual(queue[0]["status"], "pending")

    async def test_ui_action_invalid_payload_is_rejected(self):
        from astrbot.core.message.components import UIInteraction, ViewContext

        await self._seed_queue()
        event = self._admin_event()
        event.message_obj.message = [
            UIInteraction(
                action_id="approve",
                interaction_id="it-bad",
                data={"id": "not-a-number"},
                view=ViewContext(
                    id="view-1",
                    name=plugin_main.REVIEW_VIEW_NAME,
                    version=1,
                    revision=1,
                    data={"items": []},
                ),
            )
        ]
        with self.assertRaises(ValueError):
            await self._collect(self.plugin.review_approve_action(event))

    async def test_command_approve_failure_keeps_pending_for_retry(self):
        op = {"type": "create", "content": "", "umo": "qq:private:u1"}
        await self._seed_queue(self._queue_item(6, op))
        event = self._admin_event(message_str="memory review approve 6")
        results = await self._collect(self.plugin.cmd_review(event))
        self.assertEqual(len(results), 1)
        text = results[0].chain[0].text
        self.assertIn("执行失败", text)
        self.assertIn("保留待审可重试", text)
        queue = await self._queue()
        self.assertEqual(queue[0]["status"], "pending")

    async def test_concurrent_approve_of_same_id_executes_only_once(self):
        op = {
            "type": "create",
            "content": "用户喜欢拿铁",
            "umo": "qq:private:u1",
            "subject": "u1",
        }
        await self._seed_queue(self._queue_item(7, op))
        events = [self._interaction_event("approve", 7) for _ in range(2)]
        # 同一 pending id 的两个并发批准：互斥锁下只允许一个真正执行
        first, second = await asyncio.gather(
            self._collect(self.plugin.review_approve_action(events[0])),
            self._collect(self.plugin.review_approve_action(events[1])),
        )
        notices = sorted(
            [first[0].data.get("notice", ""), second[0].data.get("notice", "")]
        )
        self.assertIn("已批准并执行", notices[0] + notices[1])
        self.assertEqual(sum("已批准并执行" in n for n in notices), 1)
        self.assertEqual(sum("未找到待审项 #7" in n for n in notices), 1)
        queue = await self._queue()
        self.assertEqual(queue[0]["status"], "approved")
        docs = await self.mgr.vec_db.document_storage.get_documents(
            metadata_filters={"is_memory_record": True, "deprecated": False},
            limit=10,
            offset=0,
        )
        self.assertEqual(len(docs), 1)

    async def test_card_summary_covers_merge_and_link_ops(self):
        merge_op = {
            "type": "merge",
            "uris": ["facts://a", "facts://b"],
            "merged_content": "合并后的偏好陈述",
            "reason": "高度相似",
        }
        link_op = {
            "type": "new_link",
            "source": "facts://a",
            "target": "facts://b",
            "relation": "supports",
            "reason": "相互印证",
        }
        await self._seed_queue(
            self._queue_item(8, merge_op),
            self._queue_item(9, link_op),
        )
        event = self._admin_event(
            message_str="memory review list", support_ui_interaction=True
        )
        results = await self._collect(self.plugin.cmd_review(event))
        summaries = {item["id"]: item["summary"] for item in results[0].data["items"]}
        self.assertEqual(summaries[8], "合并后的偏好陈述")
        self.assertIn("facts://a", summaries[9])
        self.assertIn("facts://b", summaries[9])
        self.assertIn("相互印证", summaries[9])


if __name__ == "__main__":
    unittest.main()
