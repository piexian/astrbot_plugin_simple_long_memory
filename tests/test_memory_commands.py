"""记忆命令与核心记忆工具真实栈测试：真实 AstrBot 4.30 栈端到端。

MemoryPlugin、MemoryManager、SQLite/FAISS 知识库与 AstrMessageEvent 全部为真实
实例；确认会话经真实 SessionWaiter 投递回复，LLM/Embedding/Decision 请求由标准库
本地 HTTP 服务脚本化应答。
"""

from __future__ import annotations

import asyncio
import json
import unittest
import uuid

from astrbot.core.utils.session_waiter import USER_SESSIONS, SessionWaiter
from real_stack import (
    CHAT_PROVIDER_ID,
    EMBEDDING_PROVIDER_ID,
    MemoryManager,
    MemoryPlugin,
    RealStackTestCase,
    collect_command_text,
    decisions_by_request_probability,
    make_memory_meta,
    memory_manager_module,
    plugin_main,
    real_message_event,
)

__all__ = [
    "MemoryManager",
    "MemoryPlugin",
    "memory_manager_module",
    "plugin_main",
]
"""保持模块顶层暴露真实管理器、插件与记忆模块，供其他测试导入。"""

PRIVATE_UMO = "qq:private:u1"
OTHER_PRIVATE_UMO = "qq:private:u2"
GROUP_UMO = "qq:GroupMessage:g1"

CONFIRM_WORD = "确认"
CANCEL_WORD = "取消"

APPROVED_PROBABILITIES = [0.95, 0.05, 0.05, 0.05]
"""闸门四头：durable 达标且三个风险头全部低于阈值。"""

REJECTED_PROBABILITIES = [0.95, 0.90, 0.05, 0.05]
"""闸门四头：人格控制风险头超过阈值。"""


def _event(*, sender="u1", admin=False, message_str="", umo=None, group_id=""):
    """构造真实消息事件：默认私聊 u1，群聊由 umo/group_id 显式指定。"""
    return real_message_event(
        umo=umo or f"qq:private:{sender}",
        sender_id=sender,
        message_str=message_str,
        admin=admin,
        group_id=group_id,
    )


def _reply_event(event, text):
    """按原事件的身份构造确认回复事件，保持 UMO、发送者和管理员标记一致。"""
    return real_message_event(
        umo=event.unified_msg_origin,
        sender_id=event.get_sender_id(),
        sender_name=event.get_sender_name(),
        message_str=text,
        admin=event.is_admin(),
        group_id=getattr(event.message_obj, "group_id", "") or "",
    )


def _record(uri, *, owner="qq_u1", scope="personal", umo=PRIVATE_UMO, **extra):
    """按运行期写入形状构造记忆元数据，user_id 与归属保持一致。"""
    return make_memory_meta(
        uri, scope=scope, owner=owner, umo=umo, user_id=owner, **extra
    )


async def _active_uris(mgr):
    """返回真实 KB 中未废弃记忆的 URI 列表。"""
    docs = await mgr.vec_db.document_storage.get_documents(
        metadata_filters={"is_memory_record": True, "deprecated": False}, limit=200
    )
    uris = []
    for doc in docs:
        metadata = doc.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        uris.append(str(metadata.get("uri", "")))
    return uris


async def _await_session(session_id, timeout=5.0):
    """等待命令真正注册确认会话，避免与命令生成器抢跑。"""
    for _ in range(int(timeout / 0.01)):
        if session_id in USER_SESSIONS:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"confirmation session not registered: {session_id}")


async def _run_command(plugin, handler, event, *replies):
    """执行命令生成器，并按注册顺序投递真实确认回复。"""
    session_id = plugin_main._ConfirmationSessionFilter(event).session_id
    collector = asyncio.ensure_future(collect_command_text(handler(plugin, event)))
    for reply in replies:
        await _await_session(session_id)
        await SessionWaiter.trigger(session_id, _reply_event(event, reply))
    return await asyncio.wait_for(collector, timeout=60)


async def _collect(handler, plugin, event):
    """消费不需要确认会话的命令生成器。"""
    return await collect_command_text(handler(plugin, event))


class ConfirmationSessionTests(RealStackTestCase):
    def test_confirmation_filter_binds_umo_and_sender(self):
        original = _event(
            sender="admin",
            admin=True,
            umo=GROUP_UMO,
            group_id="g1",
        )
        session_filter = plugin_main._ConfirmationSessionFilter(original)

        self.assertEqual(session_filter.filter(original), session_filter.session_id)
        self.assertEqual(
            session_filter.filter(_event(sender="admin", umo=GROUP_UMO, group_id="g1")),
            session_filter.session_id,
        )
        # 同群不同发送者、同发送者不同群都不得进入该确认会话
        self.assertNotEqual(
            session_filter.filter(
                _event(sender="member", umo=GROUP_UMO, group_id="g1")
            ),
            session_filter.session_id,
        )
        self.assertNotEqual(
            session_filter.filter(
                _event(sender="admin", umo="qq:GroupMessage:g2", group_id="g2")
            ),
            session_filter.session_id,
        )

    async def test_confirmation_ignores_invalid_input_then_confirms(self):
        event = _event(message_str="memory clear")
        session_id = plugin_main._ConfirmationSessionFilter(event).session_id
        waiters: set = set()
        waiter_task = asyncio.ensure_future(
            plugin_main._wait_for_confirmation(event, waiters, timeout=5)
        )

        await _await_session(session_id)
        await SessionWaiter.trigger(session_id, _reply_event(event, "其他消息"))
        await asyncio.sleep(0.05)
        self.assertFalse(waiter_task.done())

        await SessionWaiter.trigger(session_id, _reply_event(event, CONFIRM_WORD))
        result = await asyncio.wait_for(waiter_task, timeout=5)

        self.assertEqual(result.status, "confirmed")
        self.assertEqual(result.event.message_str, CONFIRM_WORD)
        self.assertEqual(waiters, set())
        self.assertNotIn(session_id, USER_SESSIONS)

    async def test_confirmation_times_out_without_reply(self):
        event = _event(message_str="memory clear")
        waiters: set = set()

        result = await plugin_main._wait_for_confirmation(event, waiters, timeout=0.1)

        self.assertEqual(result.status, "timeout")
        self.assertEqual(waiters, set())

    async def test_confirmation_can_be_cancelled(self):
        event = _event(message_str="memory clear")
        session_id = plugin_main._ConfirmationSessionFilter(event).session_id
        waiters: set = set()
        waiter_task = asyncio.ensure_future(
            plugin_main._wait_for_confirmation(event, waiters, timeout=5)
        )

        await _await_session(session_id)
        await SessionWaiter.trigger(session_id, _reply_event(event, CANCEL_WORD))
        result = await asyncio.wait_for(waiter_task, timeout=5)

        self.assertEqual(result.status, "cancelled")
        self.assertEqual(waiters, set())


class MemoryCommandTests(RealStackTestCase):
    async def asyncSetUp(self):
        await self.start_stack(
            with_decision=False,
            with_plugin=True,
            config={
                "use_reranker": False,
                "max_memories_inject": 12,
                "max_memories_per_inject": 12,
                "extraction_provider_id": CHAT_PROVIDER_ID,
                "maintenance_model_id": CHAT_PROVIDER_ID,
            },
        )
        self.plugin.plugin_id = f"test/{uuid.uuid4().hex}"

    async def seed(self, uri, content, *, sender="u1", **extra):
        """通过真实管理器写入一条记忆。"""
        event = _event(sender=sender, umo=f"qq:private:{sender}")
        return await self.mgr.store_memory(event, content, "facts", uri=uri, **extra)

    async def seed_record(self, uri, content, **extra):
        """直接写入真实 KB 记录（用于构造压缩历史等特殊状态）。"""
        await self.mgr.vec_db.insert(content, _record(uri, **extra), f"doc-{uri}")

    def test_memory_mutation_authorization_uses_canonical_user_id(self):
        member = _event(sender="u1")
        self.assertIsNone(plugin_main.authorize_memory_mutation(member))
        self.assertIsNone(
            plugin_main.authorize_memory_mutation(member, target_user_id="qq_u1")
        )
        self.assertEqual(
            plugin_main.authorize_memory_mutation(member, target_user_id="qq_u2"),
            "普通用户只能修改自己的记忆",
        )
        self.assertEqual(
            plugin_main.authorize_memory_mutation(member, all_users=True),
            "该操作需要管理员权限",
        )
        self.assertIsNone(
            plugin_main.authorize_memory_mutation(
                _event(sender="admin", admin=True),
                target_user_id="qq_u2",
                all_users=True,
            )
        )

    async def test_clear_confirms_and_keeps_other_users_memories(self):
        await self.seed("facts://own-1", "u1 的第一条记忆")
        await self.seed("facts://own-2", "u1 的第二条记忆")
        await self.seed("facts://other", "u2 的记忆", sender="u2")

        result = await _run_command(
            self.plugin,
            plugin_main.MemoryPlugin.cmd_clear,
            _event(message_str="memory clear"),
            CONFIRM_WORD,
        )

        self.assertIn("范围: 当前用户记忆", result[0])
        self.assertIn("预计影响: 2 条", result[0])
        self.assertEqual(result[-1], "已清空 2 条记忆")
        self.assertEqual(set(await _active_uris(self.mgr)), {"facts://other"})
        self.assertEqual((await self.doc_meta("facts://other"))["uri"], "facts://other")

    async def test_clear_cancel_keeps_memories(self):
        await self.seed("facts://own", "u1 的记忆")

        result = await _run_command(
            self.plugin,
            plugin_main.MemoryPlugin.cmd_clear,
            _event(message_str="memory clear"),
            CANCEL_WORD,
        )

        self.assertEqual(result[-1], "操作已取消，未删除任何记忆")
        self.assertEqual((await self.doc_meta("facts://own"))["uri"], "facts://own")

    async def test_clear_user_flag_allows_own_and_rejects_foreign_target(self):
        await self.seed("facts://own", "u1 的记忆")
        await self.seed("facts://other", "u2 的记忆", sender="u2")

        result = await _run_command(
            self.plugin,
            plugin_main.MemoryPlugin.cmd_clear,
            _event(message_str="memory clear --user qq_u1"),
            CONFIRM_WORD,
        )
        self.assertIn("范围: 用户 qq_u1 的记忆", result[0])
        self.assertEqual(result[-1], "已清空用户 qq_u1 的 1 条记忆")
        self.assertNotIn("facts://own", await _active_uris(self.mgr))

        result = await _collect(
            plugin_main.MemoryPlugin.cmd_clear,
            self.plugin,
            _event(message_str="memory clear --user qq_u2"),
        )
        self.assertEqual(result, ["普通用户只能修改自己的记忆"])
        self.assertEqual((await self.doc_meta("facts://other"))["uri"], "facts://other")

    async def test_admin_can_clear_all_users(self):
        await self.seed("facts://own-1", "u1 的第一条记忆")
        await self.seed("facts://own-2", "u1 的第二条记忆")
        await self.seed("facts://other", "u2 的记忆", sender="u2")

        result = await _run_command(
            self.plugin,
            plugin_main.MemoryPlugin.cmd_clear,
            _event(sender="admin", admin=True, message_str="memory clear --all"),
            CONFIRM_WORD,
        )

        self.assertIn("范围: 全部记忆", result[0])
        self.assertIn("预计影响: 3 条", result[0])
        self.assertEqual(result[-1], "已清空全部 3 条记忆")
        self.assertEqual(await self.active_docs(), [])

    async def test_legacy_confirm_parameter_is_rejected(self):
        for command, handler in (
            ("memory clear --confirm clear-old", plugin_main.MemoryPlugin.cmd_clear),
            (
                "memory rebuild --confirm rebuild-old",
                plugin_main.MemoryPlugin.cmd_rebuild,
            ),
            (
                "memory rebuild --clear-cache --confirm cache-old",
                plugin_main.MemoryPlugin.cmd_rebuild,
            ),
        ):
            with self.subTest(command=command):
                result = await _collect(
                    handler,
                    self.plugin,
                    _event(sender="admin", admin=True, message_str=command),
                )
                self.assertEqual(result, ["未知参数: --confirm"])

    async def test_forget_covers_own_foreign_and_admin_targets(self):
        await self.seed("facts://own-1", "u1 的记忆一")
        await self.seed("facts://own-2", "u1 的记忆二")
        await self.seed("facts://other", "u2 的记忆", sender="u2")
        member = _event(sender="u1")

        self.assertEqual(
            await plugin_main.MemoryPlugin.tool_forget(
                self.plugin, member, "facts://own-2"
            ),
            "Memory deleted: facts://own-2 (1 record(s))",
        )

        result = await _collect(
            plugin_main.MemoryPlugin.cmd_forget,
            self.plugin,
            _event(message_str="memory forget facts://own-1"),
        )
        self.assertEqual(result, ["已删除记忆: facts://own-1"])
        self.assertNotIn("facts://own-1", await _active_uris(self.mgr))

        result = await _collect(
            plugin_main.MemoryPlugin.cmd_forget,
            self.plugin,
            _event(message_str="memory forget facts://other --user qq_u2"),
        )
        self.assertEqual(result, ["普通用户只能修改自己的记忆"])
        self.assertEqual((await self.doc_meta("facts://other"))["uri"], "facts://other")

        result = await _collect(
            plugin_main.MemoryPlugin.cmd_forget,
            self.plugin,
            _event(
                sender="admin", admin=True, message_str="memory forget facts://other"
            ),
        )
        self.assertEqual(result, ["已删除 1 条记忆: facts://other"])
        self.assertNotIn("facts://other", await _active_uris(self.mgr))

    async def test_list_rejects_invalid_page(self):
        await self.seed("facts://own", "u1 的记忆")

        for value in ("invalid", "0", "-1"):
            with self.subTest(value=value):
                result = await _collect(
                    plugin_main.MemoryPlugin.cmd_list,
                    self.plugin,
                    _event(message_str=f"memory list {value}"),
                )
                self.assertEqual(result, ["页码必须是正整数"])

        self.assertEqual((await self.doc_meta("facts://own"))["uri"], "facts://own")

    async def test_search_renders_all_results_as_one_page(self):
        for index in range(1, 13):
            await self.seed_record(f"facts://memory-{index}", f"关键词 记忆 {index}")

        result = await _collect(
            plugin_main.MemoryPlugin.cmd_search,
            self.plugin,
            _event(
                sender="admin", admin=True, message_str="memory search --all 关键词"
            ),
        )

        self.assertTrue(result[0].startswith("[全局搜索]"))
        self.assertIn("第 1/1 页，共 12 条", result[0])
        self.assertIn("\n1. [facts://", result[0])
        self.assertIn("\n12. [facts://", result[0])
        self.assertNotIn("查看下一页", result[0])

    async def test_stats_label_marks_compressed_count_as_history(self):
        await self.seed("facts://normal", "u1 的普通记忆")
        await self.seed("facts://permanent", "u1 的永久记忆", memory_type="permanent")
        for index in (1, 2):
            await self.seed_record(
                f"facts://compressed-{index}",
                f"u1 的历史记录 {index}",
                compressed=True,
                deprecated=True,
                memory_type="normal",
            )

        result = await _collect(
            plugin_main.MemoryPlugin.cmd_stats,
            self.plugin,
            _event(message_str="memory stats"),
        )

        self.assertEqual(
            result[0],
            "[个人记忆统计]\n  总数: 2\n  永久记忆: 1\n  普通记忆: 1\n  历史已压缩: 2",
        )

    async def test_memory_test_dispatches_maintenance_stage_as_dry_run(self):
        await self.seed("facts://s1", "用户住在杭州，做后端开发")
        await self.seed("facts://s2", "用户的居住地是杭州")
        self.server.queue_chat(
            json.dumps(
                {"verdict": "merge", "reason": "同一事实", "weight": 0.9},
                ensure_ascii=False,
            )
        )

        result = await _collect(
            plugin_main.MemoryPlugin.cmd_test,
            self.plugin,
            _event(sender="admin", admin=True, message_str="memory test organizer"),
        )

        self.assertIn("模式: dry-run", result[0])
        self.assertIn("  organizer: 解析 成功，操作 1", result[0])
        self.assertIn("LLM: 调用", result[0])
        # dry-run 不写入记忆：两条原始记录保持不变
        self.assertEqual(len(await self.active_docs()), 2)

    async def test_memory_test_rejects_unknown_maintenance_stage(self):
        result = await _collect(
            plugin_main.MemoryPlugin.cmd_test,
            self.plugin,
            _event(sender="admin", admin=True, message_str="memory test unknown"),
        )

        self.assertEqual(
            result,
            ["未知测试项。合法值: purge, extract, organizer, analyst, reviewer, cycle"],
        )

    async def test_rebuild_completion_uses_configured_prefix(self):
        await self.seed("facts://r1", "用户喜欢手冲咖啡")
        event = _event(sender="admin", admin=True, message_str="memory rebuild")
        prefix = self.plugin._get_cmd_prefix(event.unified_msg_origin) or "/"

        result = await _run_command(
            self.plugin,
            plugin_main.MemoryPlugin.cmd_rebuild,
            event,
            CONFIRM_WORD,
        )

        self.assertTrue(result[0].startswith("[重建确认]"))
        self.assertIn("[完整性校验]", result[-1])
        self.assertIn("数据校验通过", result[-1])
        self.assertIn(f"{prefix}memory rebuild --clear-cache", result[-1])
        self.assertEqual(await _active_uris(self.mgr), ["facts://r1"])


class AdministratorMemoryToolTests(RealStackTestCase):
    async def asyncSetUp(self):
        await self.start_stack(
            with_decision=False,
            with_plugin=True,
            config={
                "use_reranker": False,
                "enable_admin_global_memory_tool": True,
                "global_memory": True,
            },
        )
        self.plugin.plugin_id = f"test/{uuid.uuid4().hex}"

    async def seed_rules(self):
        """按作用域各写一条真实记忆，供跨作用域管理断言。"""
        await self.mgr.vec_db.insert(
            "个人规则 rule",
            _record(
                "facts://personal-rule",
                owner="qq_alice",
                umo="qq:private:alice",
            ),
            "doc-personal",
        )
        await self.mgr.vec_db.insert(
            "群规则 rule",
            _record(
                "facts://group-rule",
                owner="qq_alice",
                scope="group",
                umo=GROUP_UMO,
                owner_session_id="qq_g1",
            ),
            "doc-group",
        )
        await self.mgr.vec_db.insert(
            "全局规则 rule",
            _record(
                "facts://global-rule",
                owner="qq_admin",
                scope="global",
                umo="qq:private:admin",
                visibility="group",
            ),
            "doc-global",
        )

    async def test_admin_search_requires_enabled_admin_and_searches_all_scopes(self):
        await self.seed_rules()
        admin = _event(sender="admin", admin=True)

        self.plugin.config["enable_admin_global_memory_tool"] = False
        self.assertEqual(
            await plugin_main.MemoryPlugin.tool_search_admin(
                self.plugin, admin, "rule"
            ),
            "Administrator memory management tool is disabled in plugin config",
        )
        self.plugin.config["enable_admin_global_memory_tool"] = True

        self.assertEqual(
            await plugin_main.MemoryPlugin.tool_search_admin(
                self.plugin, _event(sender="u1"), "rule"
            ),
            "Only administrators can search memories across scopes",
        )

        result = await plugin_main.MemoryPlugin.tool_search_admin(
            self.plugin, admin, "rule", top_k=10
        )
        self.assertTrue(result.startswith("<administrator_memory_management_results>"))
        self.assertIn(
            "- uri=facts://personal-rule; scope=personal; owner_id=qq_alice", result
        )
        self.assertIn("- uri=facts://group-rule; scope=group; owner_id=qq_g1", result)
        self.assertIn(
            "- uri=facts://global-rule; scope=global; owner_id=global", result
        )
        # 管理检索不增加召回频次
        personal = await self.doc_meta("facts://personal-rule")
        self.assertEqual(personal.get("recall_count", 0), 0)

    async def test_admin_search_filters_scope_and_rejects_invalid_parameters(self):
        await self.seed_rules()
        admin = _event(sender="admin", admin=True)

        self.assertIn(
            "Error: invalid scope 'invalid'",
            await plugin_main.MemoryPlugin.tool_search_admin(
                self.plugin, admin, "rule", scope="invalid"
            ),
        )
        self.assertIn(
            "Error: invalid domain 'invalid'",
            await plugin_main.MemoryPlugin.tool_search_admin(
                self.plugin, admin, "rule", domain="invalid"
            ),
        )
        self.assertEqual(
            await plugin_main.MemoryPlugin.tool_search_admin(
                self.plugin, admin, "rule", top_k=0
            ),
            "Error: top_k must be between 1 and 20, got 0",
        )
        self.assertEqual(
            await plugin_main.MemoryPlugin.tool_search_admin(
                self.plugin, admin, "rule", top_k=True
            ),
            "Error: top_k must be an integer between 1 and 20, got 'True'",
        )
        self.assertEqual(
            await plugin_main.MemoryPlugin.tool_search_admin(self.plugin, admin, " "),
            "Error: query is required",
        )

        self.assertEqual(
            await plugin_main.MemoryPlugin.tool_search_admin(
                self.plugin, admin, "rule", scope="conversation"
            ),
            "No active memories found",
        )
        result = await plugin_main.MemoryPlugin.tool_search_admin(
            self.plugin, admin, "rule", scope="group", top_k=2
        )
        self.assertIn("facts://group-rule", result)
        self.assertNotIn("facts://personal-rule", result)
        self.assertNotIn("facts://global-rule", result)

    async def test_admin_remove_previews_before_exact_owner_confirmed_deletion(self):
        await self.seed_rules()
        admin = _event(sender="admin", admin=True)
        preview = await self.mgr.preview_admin_memory_removal(
            "facts://personal-rule", "personal", "qq_alice"
        )
        confirmation = plugin_main._confirmation_code(
            "remove-admin", preview["fingerprint"]
        )

        result = await plugin_main.MemoryPlugin.tool_remove_admin(
            self.plugin, admin, "facts://personal-rule", "personal", "qq_alice"
        )
        self.assertIn("No deletion has occurred", result)
        self.assertIn("Matching active records: 1", result)
        self.assertEqual(
            (await self.doc_meta("facts://personal-rule"))["uri"],
            "facts://personal-rule",
        )

        result = await plugin_main.MemoryPlugin.tool_remove_admin(
            self.plugin,
            admin,
            "facts://personal-rule",
            "personal",
            "qq_alice",
            confirmation,
        )
        self.assertEqual(result, "Memory deleted: facts://personal-rule (1 record(s))")
        self.assertNotIn("facts://personal-rule", await _active_uris(self.mgr))
        self.assertEqual(
            (await self.doc_meta("facts://group-rule"))["uri"], "facts://group-rule"
        )

    async def test_admin_remove_requires_exact_scope_owner_and_admin(self):
        await self.seed_rules()
        admin = _event(sender="admin", admin=True)
        remove = plugin_main.MemoryPlugin.tool_remove_admin

        self.assertEqual(
            await remove(
                self.plugin, _event(sender="u1"), "facts://global-rule", "global"
            ),
            "Only administrators can remove memories across scopes",
        )
        self.assertEqual(
            await remove(self.plugin, admin, "facts://personal-rule", "personal"),
            "Error: owner_id is required for scope 'personal' (owner_user_id)",
        )
        self.assertTrue(
            (await remove(self.plugin, admin, "not-a-uri", "global")).startswith(
                "Error: invalid memory URI 'not-a-uri'"
            )
        )
        self.assertEqual(
            await remove(self.plugin, admin, "facts://personal-rule", "invalid"),
            "Error: invalid scope 'invalid'. Valid scopes: "
            "conversation, global, group, personal",
        )
        self.assertEqual(
            await remove(
                self.plugin, admin, "facts://global-rule", "global", "qq_alice"
            ),
            "Error: owner_id must be empty for scope 'global'",
        )
        self.assertEqual(
            await remove(
                self.plugin, admin, "facts://personal-rule", "personal", "qq_bob"
            ),
            "Error: active memory not found for the exact URI, scope, and owner",
        )

    async def test_admin_removal_rechecks_fingerprint_before_deleting(self):
        await self.seed_rules()
        admin = _event(sender="admin", admin=True)
        preview = await self.mgr.preview_admin_memory_removal(
            "facts://personal-rule", "personal", "qq_alice"
        )
        stale_confirmation = plugin_main._confirmation_code(
            "remove-admin", preview["fingerprint"]
        )

        # 预览之后同一精确目标多出一条记录：旧确认码必须失效
        await self.mgr.vec_db.insert(
            "个人规则 rule（更新）",
            _record(
                "facts://personal-rule",
                owner="qq_alice",
                umo="qq:private:alice",
            ),
            "doc-personal-new",
        )
        result = await plugin_main.MemoryPlugin.tool_remove_admin(
            self.plugin,
            admin,
            "facts://personal-rule",
            "personal",
            "qq_alice",
            stale_confirmation,
        )
        # 指纹已变：旧确认码不再匹配，只回新的预览，不执行任何删除
        self.assertTrue(
            result.startswith("[Administrator memory removal confirmation]"),
            result,
        )
        self.assertIn("Matching active records: 2", result)
        self.assertIn("No deletion has occurred", result)
        self.assertIn("facts://personal-rule", await _active_uris(self.mgr))
        current = await self.mgr.preview_admin_memory_removal(
            "facts://personal-rule", "personal", "qq_alice"
        )
        self.assertNotEqual(current["fingerprint"], preview["fingerprint"])
        self.assertIn(
            plugin_main._confirmation_code("remove-admin", current["fingerprint"]),
            result,
        )

        current_confirmation = plugin_main._confirmation_code(
            "remove-admin", current["fingerprint"]
        )
        result = await plugin_main.MemoryPlugin.tool_remove_admin(
            self.plugin,
            admin,
            "facts://personal-rule",
            "personal",
            "qq_alice",
            current_confirmation,
        )
        self.assertEqual(result, "Memory deleted: facts://personal-rule (2 record(s))")
        self.assertNotIn("facts://personal-rule", await _active_uris(self.mgr))
        self.assertEqual(
            (await self.doc_meta("facts://group-rule"))["uri"], "facts://group-rule"
        )


class MemoryManagerTests(RealStackTestCase):
    async def test_stats_use_active_records_and_include_compressed_history(self):
        await self.start_stack(
            with_decision=False,
            config={"use_reranker": False},
            memories=[
                (
                    "n1",
                    "u1 的普通记忆",
                    _record("facts://n1", memory_type="normal"),
                ),
                (
                    "p1",
                    "u1 的永久记忆",
                    _record("facts://p1", memory_type="permanent"),
                ),
                (
                    "u2",
                    "u2 的普通记忆",
                    _record(
                        "facts://u2",
                        owner="qq_u2",
                        umo=OTHER_PRIVATE_UMO,
                        memory_type="normal",
                    ),
                ),
                (
                    "c1",
                    "u1 的历史记录",
                    _record(
                        "facts://c1",
                        memory_type="normal",
                        compressed=True,
                        deprecated=True,
                    ),
                ),
            ],
        )
        event = _event(sender="u1")

        self.assertEqual(
            await self.mgr.get_memory_stats(event),
            {"total": 2, "permanent": 1, "normal": 1, "compressed": 1},
        )
        self.assertEqual(
            await self.mgr.get_memory_stats(event, all_users=True),
            {"total": 3, "permanent": 1, "normal": 2, "compressed": 1},
        )

    async def test_empty_migration_commits_target_switch(self):
        await self.start_stack(with_decision=False)
        target_name = f"{self.kb_name}-target"
        await self.context.kb_manager.create_kb(
            target_name,
            embedding_provider_id=EMBEDDING_PROVIDER_ID,
            rerank_provider_id=None,
        )

        result = await self.mgr.rebuild_memories(target_kb_name=target_name)

        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["is_migration"])
        self.assertTrue(result["migration_committed"])
        self.assertTrue(result["verification"]["passed"])
        self.assertEqual(self.mgr.current_kb_name, target_name)
        self.assertEqual(self.mgr._kb_helper.kb.kb_name, target_name)


class UserMemoryToolTests(RealStackTestCase):
    async def asyncSetUp(self):
        await self.start_stack(
            with_decision=True,
            with_plugin=True,
            config={
                "use_reranker": False,
                "enable_admin_global_memory_tool": True,
                "extraction_provider_id": CHAT_PROVIDER_ID,
            },
        )
        self.plugin.plugin_id = f"test/{uuid.uuid4().hex}"
        self.server.set_decision_factory(
            decisions_by_request_probability(APPROVED_PROBABILITIES)
        )

    async def test_recall_validates_filters_and_hides_foreign_private_memories(self):
        member = _event(sender="u1")

        self.assertEqual(
            await plugin_main.MemoryPlugin.tool_recall(
                self.plugin, member, "杭州", domain="invalid"
            ),
            "Error: invalid domain 'invalid'. Valid domains: "
            "context, events, facts, preferences, user_profile",
        )
        self.assertEqual(
            await plugin_main.MemoryPlugin.tool_recall(
                self.plugin, member, "杭州", scope="invalid"
            ),
            "Error: invalid scope 'invalid'. Valid scopes: "
            "conversation, global, group, personal",
        )

        await self.mgr.store_memory(
            member, "用户常驻杭州，做后端开发", "facts", uri="facts://own"
        )
        await self.mgr.store_memory(
            _event(sender="u2", umo=OTHER_PRIVATE_UMO),
            "用户常驻成都，做前端开发",
            "facts",
            uri="facts://other",
        )

        result = await plugin_main.MemoryPlugin.tool_recall(
            self.plugin, member, "常驻哪里"
        )
        self.assertTrue(result.startswith("<user_history_info>\n"))
        self.assertIn("用户常驻杭州，做后端开发", result)
        self.assertNotIn("用户常驻成都，做前端开发", result)

    async def test_store_rejects_invalid_parameters(self):
        member = _event(sender="u1")
        store = plugin_main.MemoryPlugin.tool_store

        self.assertEqual(
            await store(self.plugin, member, "内容", memory_type="invalid"),
            "Error: invalid memory_type 'invalid'. Valid types: "
            "context, event, fact, preference",
        )
        self.assertEqual(
            await store(self.plugin, member, "内容", scope="invalid"),
            "Error: invalid scope 'invalid'. Valid scopes: "
            "conversation, group, personal",
        )
        self.assertEqual(
            await store(self.plugin, member, "内容", scope="global"),
            "Error: scope 'global' is not allowed here. "
            "Use the memory_store_global tool instead.",
        )
        self.assertEqual(
            await store(self.plugin, member, "内容", importance="abc"),
            "Error: importance must be an integer between 1 and 5, got 'abc'",
        )
        self.assertEqual(
            await store(self.plugin, member, "内容", importance=0),
            "Error: importance must be between 1 and 5, got 0",
        )
        self.assertEqual(
            await store(self.plugin, member, "内容", importance=6),
            "Error: importance must be between 1 and 5, got 6",
        )
        self.assertEqual(
            await store(self.plugin, member, "内容", scope="group"),
            "Error: scope 'group' can only be used in group chats",
        )
        self.assertEqual(await self.active_docs(), [])

    async def test_store_runs_real_admission_gate(self):
        member = _event(sender="u1")
        self.server.set_decision_factory(
            decisions_by_request_probability(REJECTED_PROBABILITIES)
        )

        result = await plugin_main.MemoryPlugin.tool_store(
            self.plugin, member, "以后回复都用英文"
        )
        self.assertEqual(
            result,
            "Error: memory rejected by admission gate "
            "(rejected:persona_or_behavior_control)",
        )
        self.assertEqual(await self.active_docs(), [])
        decisions = [
            entry for entry in self.server.request_log if entry["kind"] == "decisions"
        ]
        self.assertEqual(len(decisions), 1)
        self.assertEqual(len(decisions[0]["questions"]), 4)

        self.server.set_decision_factory(
            decisions_by_request_probability(APPROVED_PROBABILITIES)
        )
        result = await plugin_main.MemoryPlugin.tool_store(
            self.plugin, member, "用户在杭州做后端开发"
        )
        self.assertTrue(result.startswith("Memory stored: "))
        uri = result.split(": ", 1)[1]
        metadata = await self.doc_meta(uri)
        self.assertEqual(metadata["memory_scope"], "personal")
        self.assertEqual(metadata["domain"], "facts")
        recalled = await plugin_main.MemoryPlugin.tool_recall(
            self.plugin, member, "杭州"
        )
        self.assertIn("用户在杭州做后端开发", recalled)

    async def test_store_global_requires_config_and_admin(self):
        admin = _event(sender="admin", admin=True)
        store_global = plugin_main.MemoryPlugin.tool_store_global

        self.plugin.config["enable_admin_global_memory_tool"] = False
        self.assertEqual(
            await store_global(self.plugin, admin, "全局规则"),
            "Global memory tool is disabled in plugin config",
        )
        self.plugin.config["enable_admin_global_memory_tool"] = True

        self.assertEqual(
            await store_global(self.plugin, _event(sender="u1"), "全局规则"),
            "Only administrators can store global memories",
        )
        self.assertEqual(
            await store_global(self.plugin, admin, "全局规则", memory_type="invalid"),
            "Error: invalid memory_type 'invalid'. Valid types: "
            "context, event, fact, preference",
        )
        self.assertEqual(await self.active_docs(), [])

    async def test_store_global_writes_memory_recallable_by_all_users(self):
        result = await plugin_main.MemoryPlugin.tool_store_global(
            self.plugin, _event(sender="admin", admin=True), "所有会话都使用中文回答"
        )
        self.assertTrue(result.startswith("Global memory stored: "))
        uri = result.split(": ", 1)[1]
        metadata = await self.doc_meta(uri)
        self.assertEqual(metadata["memory_scope"], "global")
        self.assertEqual(metadata["memory_type"], "permanent")
        self.assertEqual(metadata["importance"], 5)

        member = _event(sender="u1")
        scoped = await plugin_main.MemoryPlugin.tool_recall(
            self.plugin, member, "中文回答", scope="global"
        )
        self.assertIn("所有会话都使用中文回答", scoped)
        default = await plugin_main.MemoryPlugin.tool_recall(
            self.plugin, member, "中文回答"
        )
        self.assertIn("所有会话都使用中文回答", default)

    async def test_update_preserves_metadata_and_denies_foreign_users(self):
        owner = _event(sender="u1")
        uri = await self.mgr.store_memory(
            owner, "用户喜欢手冲咖啡", "facts", importance=4
        )

        result = await plugin_main.MemoryPlugin.tool_update(
            self.plugin, owner, uri, "用户喜欢意式咖啡"
        )
        self.assertTrue(result.startswith("Memory updated: "))
        new_uri = result.split(" -> ", 1)[1]
        metadata = await self.doc_meta(new_uri)
        self.assertEqual(metadata["domain"], "facts")
        self.assertEqual(metadata["memory_scope"], "personal")
        self.assertEqual(metadata["importance"], 4)
        self.assertEqual(metadata["updated_by"], "user")
        self.assertFalse(metadata["deprecated"])
        self.assertEqual(metadata["memory_content"], "用户喜欢意式咖啡")
        self.assertNotIn(uri, await _active_uris(self.mgr))

        result = await plugin_main.MemoryPlugin.tool_update(
            self.plugin, _event(sender="u2", umo=OTHER_PRIVATE_UMO), new_uri, "被篡改"
        )
        self.assertEqual(result, f"Error: memory not found or not yours: {new_uri}")
        self.assertEqual(
            (await self.doc_meta(new_uri))["memory_content"], "用户喜欢意式咖啡"
        )

        global_result = await plugin_main.MemoryPlugin.tool_store_global(
            self.plugin, _event(sender="admin", admin=True), "所有会话都使用中文回答"
        )
        global_uri = global_result.split(": ", 1)[1]
        result = await plugin_main.MemoryPlugin.tool_update(
            self.plugin, owner, global_uri, "普通用户尝试修改全局记忆"
        )
        self.assertEqual(result, f"Error: memory not found or not yours: {global_uri}")
        self.assertEqual(
            (await self.doc_meta(global_uri))["memory_content"],
            "所有会话都使用中文回答",
        )


if __name__ == "__main__":
    unittest.main()
