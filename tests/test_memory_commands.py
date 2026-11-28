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
    MemoryPlugin,
    RealStackTestCase,
    collect_command_text,
    decisions_by_request_probability,
    make_memory_meta,
    plugin_main,
    real_message_event,
)

PRIVATE_UMO = "qq:private:u1"
OTHER_PRIVATE_UMO = "qq:private:u2"
GROUP_UMO = "qq:GroupMessage:g1"

APPROVED_PROBABILITIES = [0.95, 0.05, 0.05, 0.05]
"""闸门四头：durable 达标且三个风险头全部低于阈值。"""

REJECTED_PROBABILITIES = [0.95, 0.90, 0.05, 0.05]
"""闸门四头：人格控制风险头超过阈值。"""

# 参数校验与权限边界的固定报错文本，断言按原文逐字比对
RECALL_INVALID_DOMAIN = "Error: invalid domain 'invalid'. Valid domains: context, events, facts, preferences, user_profile"
RECALL_INVALID_SCOPE = "Error: invalid scope 'invalid'. Valid scopes: conversation, global, group, personal"
STORE_INVALID_TYPE = "Error: invalid memory_type 'invalid'. Valid types: context, event, fact, preference"
STORE_INVALID_SCOPE = (
    "Error: invalid scope 'invalid'. Valid scopes: conversation, group, personal"
)
STORE_GLOBAL_REJECTED = "Error: scope 'global' is not allowed here. Use the memory_store_global tool instead."
IMPORTANCE_NOT_INT = "Error: importance must be an integer between 1 and 5, got 'abc'"
IMPORTANCE_LOW = "Error: importance must be between 1 and 5, got 0"
IMPORTANCE_HIGH = "Error: importance must be between 1 and 5, got 6"
GROUP_SCOPE_ONLY = "Error: scope 'group' can only be used in group chats"
GLOBAL_DISABLED = "Global memory tool is disabled in plugin config"
GLOBAL_ADMIN_ONLY = "Only administrators can store global memories"
ADMIN_REMOVE_ONLY = "Only administrators can remove memories across scopes"
ADMIN_SEARCH_DISABLED = (
    "Administrator memory management tool is disabled in plugin config"
)
ADMIN_SEARCH_ONLY = "Only administrators can search memories across scopes"
ADMIN_OWNER_REQUIRED = (
    "Error: owner_id is required for scope 'personal' (owner_user_id)"
)
ADMIN_OWNER_FORBIDDEN = "Error: owner_id must be empty for scope 'global'"
ADMIN_INVALID_SCOPE = "Error: invalid scope 'invalid'. Valid scopes: conversation, global, group, personal"
ADMIN_TARGET_MISSING = (
    "Error: active memory not found for the exact URI, scope, and owner"
)
ADMIN_TOPK_RANGE = "Error: top_k must be between 1 and 20, got 0"
ADMIN_TOPK_TYPE = "Error: top_k must be an integer between 1 and 20, got 'True'"


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
    metas = [
        json.loads(doc["metadata"])
        if isinstance(doc["metadata"], str)
        else doc["metadata"]
        for doc in docs
    ]
    return [str(meta.get("uri", "")) for meta in metas]


async def _await_session(session_id, timeout=5.0):
    """等待命令真正注册确认会话，避免与命令生成器抢跑。"""
    for _ in range(int(timeout / 0.01)):
        if session_id in USER_SESSIONS:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"confirmation session not registered: {session_id}")


class _PluginCase(RealStackTestCase):
    """按名字调用真实插件工具入口与命令生成器。"""

    async def call_tool(self, name, event, *args, **kwargs):
        """调用真实插件工具入口，参数原样透传。"""
        handler = getattr(MemoryPlugin, name)
        return await handler(self.plugin, event, *args, **kwargs)

    async def cmd(self, name, event, *replies):
        """执行命令生成器；给出回复时按注册顺序投递到真实确认会话。"""
        handler = getattr(MemoryPlugin, name)
        if not replies:
            return await collect_command_text(handler(self.plugin, event))
        session_id = plugin_main._ConfirmationSessionFilter(event).session_id
        collector = asyncio.ensure_future(
            collect_command_text(handler(self.plugin, event))
        )
        for reply in replies:
            await _await_session(session_id)
            await SessionWaiter.trigger(session_id, _reply_event(event, reply))
        return await asyncio.wait_for(collector, timeout=60)

    async def assert_memory_uri(self, uri):
        """断言该 URI 的记忆仍可读且未被改写。"""
        self.assertEqual((await self.doc_meta(uri))["uri"], uri)


class ConfirmationSessionTests(RealStackTestCase):
    async def test_confirmation_filter_and_lifecycle(self):
        # 过滤器只认原事件的身份组合：同群不同发送者、同发送者不同群都不得进入
        original = _event(sender="admin", admin=True, umo=GROUP_UMO, group_id="g1")
        session_filter = plugin_main._ConfirmationSessionFilter(original)
        self.assertEqual(session_filter.filter(original), session_filter.session_id)
        self.assertEqual(
            session_filter.filter(_event(sender="admin", umo=GROUP_UMO, group_id="g1")),
            session_filter.session_id,
        )
        for sender, umo, group_id in (
            ("member", GROUP_UMO, "g1"),
            ("admin", "qq:GroupMessage:g2", "g2"),
        ):
            with self.subTest(sender=sender, umo=umo):
                other = _event(sender=sender, umo=umo, group_id=group_id)
                self.assertNotEqual(
                    session_filter.filter(other), session_filter.session_id
                )

        # 确认：无关消息不结束等待，确认词返回 confirmed 并清空会话与等待集合
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
        await SessionWaiter.trigger(session_id, _reply_event(event, "确认"))
        result = await asyncio.wait_for(waiter_task, timeout=5)
        self.assertEqual(result.status, "confirmed")
        self.assertEqual(result.event.message_str, "确认")
        self.assertEqual(waiters, set())
        self.assertNotIn(session_id, USER_SESSIONS)

        # 超时与取消各自返回对应状态，同样不残留等待者
        timeout_waiters: set = set()
        timed_out = await plugin_main._wait_for_confirmation(
            event, timeout_waiters, timeout=0.1
        )
        self.assertEqual(timed_out.status, "timeout")
        self.assertEqual(timeout_waiters, set())
        cancel_waiters: set = set()
        cancel_task = asyncio.ensure_future(
            plugin_main._wait_for_confirmation(event, cancel_waiters, timeout=5)
        )
        await _await_session(session_id)
        await SessionWaiter.trigger(session_id, _reply_event(event, "取消"))
        cancelled = await asyncio.wait_for(cancel_task, timeout=5)
        self.assertEqual(cancelled.status, "cancelled")
        self.assertEqual(cancel_waiters, set())


class MemoryCommandTests(_PluginCase):
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

    async def test_clear_and_forget_scope_paths(self):
        # 普通用户只能改自己的记忆，跨用户与全量清空都需要管理员
        member = _event(sender="u1")
        self.assertIsNone(plugin_main.authorize_memory_mutation(member))
        self.assertIsNone(
            plugin_main.authorize_memory_mutation(member, target_user_id="qq_u1")
        )
        rejected = plugin_main.authorize_memory_mutation(member, target_user_id="qq_u2")
        self.assertEqual(rejected, "普通用户只能修改自己的记忆")
        rejected = plugin_main.authorize_memory_mutation(member, all_users=True)
        self.assertEqual(rejected, "该操作需要管理员权限")
        self.assertIsNone(
            plugin_main.authorize_memory_mutation(
                _event(sender="admin", admin=True),
                target_user_id="qq_u2",
                all_users=True,
            )
        )

        await self.seed("facts://own-1", "u1 的第一条记忆")
        await self.seed("facts://own-2", "u1 的第二条记忆")
        await self.seed("facts://other", "u2 的记忆", sender="u2")

        # 取消：一条也不删
        result = await self.cmd("cmd_clear", _event(message_str="memory clear"), "取消")
        self.assertEqual(result[-1], "操作已取消，未删除任何记忆")
        self.assertEqual(len(await _active_uris(self.mgr)), 3)

        # 确认：只清当前用户，其他用户记忆不受影响
        result = await self.cmd("cmd_clear", _event(message_str="memory clear"), "确认")
        self.assertIn("范围: 当前用户记忆", result[0])
        self.assertIn("预计影响: 2 条", result[0])
        self.assertEqual(result[-1], "已清空 2 条记忆")
        self.assertEqual(set(await _active_uris(self.mgr)), {"facts://other"})
        await self.assert_memory_uri("facts://other")

        # --user 只允许指向本人
        await self.seed("facts://own", "u1 的记忆")
        event = _event(message_str="memory clear --user qq_u1")
        result = await self.cmd("cmd_clear", event, "确认")
        self.assertIn("范围: 用户 qq_u1 的记忆", result[0])
        self.assertEqual(result[-1], "已清空用户 qq_u1 的 1 条记忆")
        self.assertNotIn("facts://own", await _active_uris(self.mgr))
        event = _event(message_str="memory clear --user qq_u2")
        self.assertEqual(
            await self.cmd("cmd_clear", event), ["普通用户只能修改自己的记忆"]
        )

        # forget：本人删除、越权删除被拒、管理员按 URI 代删
        await self.seed("facts://own-1", "u1 的记忆一")
        await self.seed("facts://own-2", "u1 的记忆二")
        result = await self.call_tool("tool_forget", member, "facts://own-2")
        self.assertEqual(result, "Memory deleted: facts://own-2 (1 record(s))")
        result = await self.cmd(
            "cmd_forget", _event(message_str="memory forget facts://own-1")
        )
        self.assertEqual(result, ["已删除记忆: facts://own-1"])
        self.assertNotIn("facts://own-1", await _active_uris(self.mgr))
        event = _event(message_str="memory forget facts://other --user qq_u2")
        self.assertEqual(
            await self.cmd("cmd_forget", event), ["普通用户只能修改自己的记忆"]
        )
        await self.assert_memory_uri("facts://other")
        event = _event(
            sender="admin", admin=True, message_str="memory forget facts://other"
        )
        self.assertEqual(
            await self.cmd("cmd_forget", event), ["已删除 1 条记忆: facts://other"]
        )
        self.assertNotIn("facts://other", await _active_uris(self.mgr))

        # 管理员 --all 清空全部用户
        await self.seed("facts://own-1", "u1 的第一条记忆")
        await self.seed("facts://own-2", "u1 的第二条记忆")
        await self.seed("facts://other", "u2 的记忆", sender="u2")
        event = _event(sender="admin", admin=True, message_str="memory clear --all")
        result = await self.cmd("cmd_clear", event, "确认")
        self.assertIn("范围: 全部记忆", result[0])
        self.assertIn("预计影响: 3 条", result[0])
        self.assertEqual(result[-1], "已清空全部 3 条记忆")
        self.assertEqual(await self.active_docs(), [])

    async def test_read_only_commands_and_rebuild_rendering(self):
        # 旧版 --confirm 参数一律拒绝
        for command, name in (
            ("memory clear --confirm clear-old", "cmd_clear"),
            ("memory rebuild --confirm rebuild-old", "cmd_rebuild"),
            ("memory rebuild --clear-cache --confirm cache-old", "cmd_rebuild"),
        ):
            with self.subTest(command=command):
                event = _event(sender="admin", admin=True, message_str=command)
                self.assertEqual(await self.cmd(name, event), ["未知参数: --confirm"])

        await self.seed("facts://normal", "u1 的普通记忆")
        # 非法页码一律拒绝，不改动记忆
        for value in ("invalid", "0", "-1"):
            with self.subTest(value=value):
                event = _event(message_str=f"memory list {value}")
                self.assertEqual(
                    await self.cmd("cmd_list", event), ["页码必须是正整数"]
                )
        await self.assert_memory_uri("facts://normal")

        # 统计口径：压缩历史单独计数，不并入总数
        await self.seed("facts://permanent", "u1 的永久记忆", memory_type="permanent")
        for index in (1, 2):
            await self.seed_record(
                f"facts://compressed-{index}",
                f"u1 的历史记录 {index}",
                compressed=True,
                deprecated=True,
                memory_type="normal",
            )
        stats = await self.cmd("cmd_stats", _event(message_str="memory stats"))
        self.assertEqual(
            stats[0],
            "[个人记忆统计]\n  总数: 2\n  永久记忆: 1\n  普通记忆: 1\n  历史已压缩: 2",
        )

        # 全局搜索单页返回全部命中
        for index in range(1, 13):
            await self.seed_record(f"facts://memory-{index}", f"关键词 记忆 {index}")
        event = _event(
            sender="admin", admin=True, message_str="memory search --all 关键词"
        )
        search = await self.cmd("cmd_search", event)
        self.assertTrue(search[0].startswith("[全局搜索]"))
        self.assertIn("第 1/1 页，共 12 条", search[0])
        self.assertIn("\n1. [facts://", search[0])
        self.assertIn("\n12. [facts://", search[0])
        self.assertNotIn("查看下一页", search[0])

        # memory test：dry-run 只解析不写入，未知测试项报出全部合法值
        await self.seed("facts://s1", "用户住在杭州，做后端开发")
        await self.seed("facts://s2", "用户的居住地是杭州")
        verdict = {"verdict": "merge", "reason": "同一事实", "weight": 0.9}
        self.server.queue_chat(json.dumps(verdict, ensure_ascii=False))
        before = len(await self.active_docs())
        event = _event(sender="admin", admin=True, message_str="memory test organizer")
        result = await self.cmd("cmd_test", event)
        self.assertIn("模式: dry-run", result[0])
        self.assertIn("  organizer: 解析 成功，操作 1", result[0])
        self.assertIn("LLM: 调用", result[0])
        self.assertEqual(len(await self.active_docs()), before)
        event = _event(sender="admin", admin=True, message_str="memory test unknown")
        self.assertEqual(
            await self.cmd("cmd_test", event),
            ["未知测试项。合法值: purge, extract, organizer, analyst, reviewer, cycle"],
        )

        # 重建：确认后给出完整性校验与配置前缀
        await self.seed("facts://r1", "用户喜欢手冲咖啡")
        before = set(await _active_uris(self.mgr))
        event = _event(sender="admin", admin=True, message_str="memory rebuild")
        prefix = self.plugin._get_cmd_prefix(event.unified_msg_origin) or "/"
        result = await self.cmd("cmd_rebuild", event, "确认")
        self.assertTrue(result[0].startswith("[重建确认]"))
        self.assertIn("[完整性校验]", result[-1])
        self.assertIn("数据校验通过", result[-1])
        self.assertIn(f"{prefix}memory rebuild --clear-cache", result[-1])
        self.assertEqual(set(await _active_uris(self.mgr)), before)


class AdministratorMemoryToolTests(_PluginCase):
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
        await self.seed_rules()

    async def seed_rules(self):
        """按作用域各写一条真实记忆，供跨作用域管理断言。"""
        rows = (
            (
                "doc-personal",
                "个人规则 rule",
                "facts://personal-rule",
                {"owner": "qq_alice", "umo": "qq:private:alice"},
            ),
            (
                "doc-group",
                "群规则 rule",
                "facts://group-rule",
                {
                    "owner": "qq_alice",
                    "scope": "group",
                    "umo": GROUP_UMO,
                    "owner_session_id": "qq_g1",
                },
            ),
            (
                "doc-global",
                "全局规则 rule",
                "facts://global-rule",
                {
                    "owner": "qq_admin",
                    "scope": "global",
                    "umo": "qq:private:admin",
                    "visibility": "group",
                },
            ),
        )
        for doc_id, content, uri, meta in rows:
            await self.mgr.vec_db.insert(content, _record(uri, **meta), doc_id)

    async def search_admin(self, event, query="rule", **kwargs):
        return await self.call_tool("tool_search_admin", event, query, **kwargs)

    async def remove_admin(self, event, uri, scope, owner_id="", confirm=""):
        return await self.call_tool(
            "tool_remove_admin", event, uri, scope, owner_id, confirm
        )

    async def preview_removal(self, uri, scope, owner_id):
        return await self.mgr.preview_admin_memory_removal(uri, scope, owner_id)

    async def test_admin_search_requires_enabled_admin_and_searches_all_scopes(self):
        admin = _event(sender="admin", admin=True)
        self.plugin.config["enable_admin_global_memory_tool"] = False
        self.assertEqual(await self.search_admin(admin), ADMIN_SEARCH_DISABLED)
        self.plugin.config["enable_admin_global_memory_tool"] = True
        member = _event(sender="u1")
        self.assertEqual(await self.search_admin(member), ADMIN_SEARCH_ONLY)

        result = await self.search_admin(admin, top_k=10)
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

        # 参数校验与作用域过滤分支
        for kwargs, expected in (
            ({"scope": "invalid"}, "Error: invalid scope 'invalid'"),
            ({"domain": "invalid"}, "Error: invalid domain 'invalid'"),
        ):
            with self.subTest(**kwargs):
                self.assertIn(expected, await self.search_admin(admin, **kwargs))
        for kwargs, expected in (
            ({"top_k": 0}, ADMIN_TOPK_RANGE),
            ({"top_k": True}, ADMIN_TOPK_TYPE),
            ({"scope": "conversation"}, "No active memories found"),
        ):
            with self.subTest(**kwargs):
                result = await self.search_admin(admin, **kwargs)
                self.assertEqual(result, expected)
        self.assertEqual(
            await self.search_admin(admin, " "), "Error: query is required"
        )

        scoped = await self.search_admin(admin, scope="group", top_k=2)
        self.assertIn("facts://group-rule", scoped)
        self.assertNotIn("facts://personal-rule", scoped)
        self.assertNotIn("facts://global-rule", scoped)

    async def test_admin_removal_validates_target_and_rechecks_fingerprint(self):
        admin = _event(sender="admin", admin=True)
        # 非管理员一律拒绝
        member = _event(sender="u1")
        self.assertEqual(
            await self.remove_admin(member, "facts://global-rule", "global"),
            ADMIN_REMOVE_ONLY,
        )
        # 缺 owner、非法作用域、owner 不匹配、目标不存在都在写库前拒绝
        for uri, scope, owner_id, expected in (
            ("facts://personal-rule", "personal", "", ADMIN_OWNER_REQUIRED),
            ("facts://global-rule", "global", "qq_alice", ADMIN_OWNER_FORBIDDEN),
            ("facts://personal-rule", "invalid", "", ADMIN_INVALID_SCOPE),
            ("facts://personal-rule", "personal", "qq_bob", ADMIN_TARGET_MISSING),
        ):
            with self.subTest(uri=uri, scope=scope):
                result = await self.remove_admin(admin, uri, scope, owner_id)
                self.assertEqual(result, expected)
        result = await self.remove_admin(admin, "not-a-uri", "global")
        self.assertTrue(result.startswith("Error: invalid memory URI 'not-a-uri'"))

        # 预览给出精确目标与指纹，确认码匹配后才真正删除
        uri = "facts://personal-rule"
        preview = await self.preview_removal(uri, "personal", "qq_alice")
        confirmation = plugin_main._confirmation_code(
            "remove-admin", preview["fingerprint"]
        )
        result = await self.remove_admin(admin, uri, "personal", "qq_alice")
        self.assertIn("No deletion has occurred", result)
        self.assertIn("Matching active records: 1", result)
        await self.assert_memory_uri(uri)
        result = await self.remove_admin(
            admin, uri, "personal", "qq_alice", confirmation
        )
        self.assertEqual(result, f"Memory deleted: {uri} (1 record(s))")
        self.assertNotIn(uri, await _active_uris(self.mgr))

        # 预览之后同一精确目标多出一条记录：旧确认码必须失效
        record = _record(uri, owner="qq_alice", umo="qq:private:alice")
        await self.mgr.vec_db.insert("个人规则 rule", record, "doc-personal-restored")
        preview = await self.preview_removal(uri, "personal", "qq_alice")
        fingerprint = preview["fingerprint"]
        await self.mgr.vec_db.insert(
            "个人规则 rule（更新）", record, "doc-personal-new"
        )
        stale = plugin_main._confirmation_code("remove-admin", fingerprint)
        result = await self.remove_admin(admin, uri, "personal", "qq_alice", stale)
        # 指纹已变：旧确认码不再匹配，只回新的预览，不执行任何删除
        self.assertTrue(
            result.startswith("[Administrator memory removal confirmation]"), result
        )
        self.assertIn("Matching active records: 2", result)
        self.assertIn("No deletion has occurred", result)
        self.assertIn(uri, await _active_uris(self.mgr))
        current = await self.preview_removal(uri, "personal", "qq_alice")
        self.assertNotEqual(current["fingerprint"], fingerprint)
        self.assertIn(
            plugin_main._confirmation_code("remove-admin", current["fingerprint"]),
            result,
        )
        fresh = plugin_main._confirmation_code("remove-admin", current["fingerprint"])
        result = await self.remove_admin(admin, uri, "personal", "qq_alice", fresh)
        self.assertEqual(result, f"Memory deleted: {uri} (2 record(s))")
        self.assertNotIn(uri, await _active_uris(self.mgr))
        await self.assert_memory_uri("facts://group-rule")


class MemoryManagerTests(RealStackTestCase):
    async def test_stats_use_active_records_and_include_compressed_history(self):
        normal = {"memory_type": "normal"}
        await self.start_stack(
            with_decision=False,
            config={"use_reranker": False},
            memories=[
                ("n1", "u1 的普通记忆", _record("facts://n1", **normal)),
                ("p1", "u1 的永久记忆", _record("facts://p1", memory_type="permanent")),
                (
                    "u2",
                    "u2 的普通记忆",
                    _record(
                        "facts://u2", owner="qq_u2", umo=OTHER_PRIVATE_UMO, **normal
                    ),
                ),
                (
                    "c1",
                    "u1 的历史记录",
                    _record("facts://c1", compressed=True, deprecated=True, **normal),
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


class UserMemoryToolTests(_PluginCase):
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

    async def test_tools_reject_invalid_parameters_before_writing(self):
        member = _event(sender="u1")
        admin = _event(sender="admin", admin=True)
        recall = self.plugin.tool_recall
        store = self.plugin.tool_store
        store_global = self.plugin.tool_store_global

        self.assertEqual(
            await recall(member, "杭州", domain="invalid"), RECALL_INVALID_DOMAIN
        )
        self.assertEqual(
            await recall(member, "杭州", scope="invalid"), RECALL_INVALID_SCOPE
        )
        for kwargs, expected in (
            ({"memory_type": "invalid"}, STORE_INVALID_TYPE),
            ({"scope": "invalid"}, STORE_INVALID_SCOPE),
            ({"scope": "global"}, STORE_GLOBAL_REJECTED),
            ({"importance": "abc"}, IMPORTANCE_NOT_INT),
            ({"importance": 0}, IMPORTANCE_LOW),
            ({"importance": 6}, IMPORTANCE_HIGH),
            ({"scope": "group"}, GROUP_SCOPE_ONLY),
        ):
            with self.subTest(**kwargs):
                self.assertEqual(await store(member, "内容", **kwargs), expected)

        # 全局记忆工具：开关、管理员权限与类型校验都在写库前拒绝
        self.plugin.config["enable_admin_global_memory_tool"] = False
        self.assertEqual(await store_global(admin, "全局规则"), GLOBAL_DISABLED)
        self.plugin.config["enable_admin_global_memory_tool"] = True
        self.assertEqual(await store_global(member, "全局规则"), GLOBAL_ADMIN_ONLY)
        result = await store_global(admin, "全局规则", memory_type="invalid")
        self.assertEqual(result, STORE_INVALID_TYPE)
        self.assertEqual(await self.active_docs(), [])

    async def test_store_gate_recall_update_and_global_paths(self):
        member = _event(sender="u1")
        admin = _event(sender="admin", admin=True)
        recall = self.plugin.tool_recall
        store = self.plugin.tool_store
        update = self.plugin.tool_update

        # 闸门拒绝：不写库，且确实经过四头判定
        self.server.set_decision_factory(
            decisions_by_request_probability(REJECTED_PROBABILITIES)
        )
        result = await store(member, "以后回复都用英文")
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

        # 闸门放行：写入个人记忆，元数据与召回结果一致
        self.server.set_decision_factory(
            decisions_by_request_probability(APPROVED_PROBABILITIES)
        )
        result = await store(member, "用户在杭州做后端开发")
        self.assertTrue(result.startswith("Memory stored: "))
        uri = result.split(": ", 1)[1]
        metadata = await self.doc_meta(uri)
        self.assertEqual(metadata["memory_scope"], "personal")
        self.assertEqual(metadata["domain"], "facts")
        self.assertIn("用户在杭州做后端开发", await recall(member, "杭州"))

        # 他人私聊记忆不进入当前用户的召回结果
        other = _event(sender="u2", umo=OTHER_PRIVATE_UMO)
        await self.mgr.store_memory(other, "用户常驻成都，做前端开发", "facts")
        await self.mgr.store_memory(member, "用户常驻杭州，做后端开发", "facts")
        recalled = await recall(member, "常驻哪里")
        self.assertTrue(recalled.startswith("<user_history_info>\n"))
        self.assertIn("用户常驻杭州，做后端开发", recalled)
        self.assertNotIn("用户常驻成都，做前端开发", recalled)

        # update：保留域、作用域与重要度，旧记录下线，越权修改被拒
        uri = await self.mgr.store_memory(
            member, "用户喜欢手冲咖啡", "facts", importance=4
        )
        result = await update(member, uri, "用户喜欢意式咖啡")
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
        result = await update(other, new_uri, "被篡改")
        self.assertEqual(result, f"Error: memory not found or not yours: {new_uri}")
        metadata = await self.doc_meta(new_uri)
        self.assertEqual(metadata["memory_content"], "用户喜欢意式咖啡")

        # store_global：写入全局永久记忆，所有用户都能按 scope 与默认召回
        result = await self.call_tool(
            "tool_store_global", admin, "所有会话都使用中文回答"
        )
        self.assertTrue(result.startswith("Global memory stored: "))
        global_uri = result.split(": ", 1)[1]
        metadata = await self.doc_meta(global_uri)
        self.assertEqual(metadata["memory_scope"], "global")
        self.assertEqual(metadata["memory_type"], "permanent")
        self.assertEqual(metadata["importance"], 5)
        scoped = await recall(member, "中文回答", scope="global")
        self.assertIn("所有会话都使用中文回答", scoped)
        self.assertIn("所有会话都使用中文回答", await recall(member, "中文回答"))
        result = await update(member, global_uri, "普通用户尝试修改全局记忆")
        self.assertEqual(result, f"Error: memory not found or not yours: {global_uri}")
        metadata = await self.doc_meta(global_uri)
        self.assertEqual(metadata["memory_content"], "所有会话都使用中文回答")


if __name__ == "__main__":
    unittest.main()
