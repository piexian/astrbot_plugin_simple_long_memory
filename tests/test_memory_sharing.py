"""共享开关、关联权限和多主体候选查询回归：真实 AstrBot 栈底座。

MemoryManager、SQLite/FAISS 知识库与 AstrMessageEvent 全部使用真实实现，
LLM/Embedding 请求由标准库本地 HTTP 服务应答。
"""

from __future__ import annotations

import json
from pathlib import Path

from real_stack import (
    MemoryManager,
    RealStackTestCase,
    make_memory_meta,
    plugin_main,
    real_message_event,
)

UMOInfo = plugin_main.UMOInfo

GROUP_UMO = "qq:GroupMessage:g1"
OTHER_GROUP_UMO = "qq:GroupMessage:g2"
PRIVATE_UMO = "qq:FriendMessage:bob"


def _meta(
    uri="facts://coffee", scope="personal", owner="qq_bob", umo=GROUP_UMO, **extra
):
    """按运行期写入形状构造记忆元数据。"""
    parsed = UMOInfo.parse(umo)
    return make_memory_meta(
        uri,
        scope=scope,
        owner=owner,
        umo=umo,
        platform_id=parsed.platform_id,
        owner_session_id=f"{parsed.platform_id}_{parsed.session_id}",
        owner_user_ids=extra.pop("owner_user_ids", [owner]),
        visibility=extra.pop("visibility", "private"),
        **extra,
    )


def _memory(**kwargs):
    return {"text": "喜欢咖啡 coffee", "metadata": _meta(**kwargs)}


def _group_event(umo=GROUP_UMO, sender="bob"):
    return real_message_event(umo=umo, sender_id=sender, group_id=umo.split(":", 2)[2])


def _schema_defaults():
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "_conf_schema.json").read_text(encoding="utf-8"))
    return schema, {key: item.get("default") for key, item in schema.items()}


class SharingConfigTests(RealStackTestCase):
    async def test_legacy_key_and_new_defaults_are_preserved(self):
        root = Path(__file__).resolve().parents[1]
        schema, upgraded = _schema_defaults()
        self.assertEqual(schema["global_memory"]["type"], "bool")
        self.assertNotIn("share_personal_across_sessions", schema)
        self.assertIs(schema["share_group_across_groups"]["default"], False)

        upgraded.pop("kb_name")
        upgraded["global_memory"] = False
        await self.start_stack(with_decision=False, config=upgraded)

        group_event = _group_event()
        await self.mgr.store_memory(group_event, "喜欢咖啡", "facts")
        docs = await self.active_docs()
        self.assertEqual(len(docs), 1)

        self.assertTrue(self.mgr._is_memory_visible(group_event, docs[0]))
        stranger = real_message_event(umo="qq:private:u9", sender_id="u9")
        self.assertFalse(self.mgr._is_memory_visible(stranger, docs[0]))

        for locale in ("zh-CN", "en-US"):
            translations = json.loads(
                (root / ".astrbot-plugin" / "i18n" / f"{locale}.json").read_text(
                    encoding="utf-8"
                )
            )["config"]
            for key in (
                "global_memory",
                "share_group_across_groups",
                "recall_query_max_chars",
                "recall_timeout_seconds",
            ):
                self.assertTrue(translations[key]["description"])
                self.assertTrue(translations[key]["hint"])


class SharingPolicyTests(RealStackTestCase):
    async def asyncSetUp(self):
        await self.start_stack(with_decision=False)

    def manager(self, **config):
        mgr = MemoryManager(
            self.context.kb_manager, {"kb_name": self.kb_name, **config}
        )
        mgr.initialize()
        return mgr

    def test_umo_normalizes_without_rewriting_origin(self):
        for raw, expected in (
            ("GroupMessage", "group"),
            ("FriendMessage", "private"),
            ("OtherMessage", "other"),
            ("group", "group"),
        ):
            parsed = UMOInfo.parse(f"qq:{raw}:id:with:colons")
            self.assertEqual(parsed.session_type, expected)
            self.assertEqual(parsed.session_id, "id:with:colons")
            self.assertTrue(parsed.is_valid)
        for raw in ("HEX", "qq::id", "qq:unknown:id", ":group:id", "qq:private:"):
            self.assertFalse(UMOInfo.parse(raw).is_valid)

        event = _group_event()
        self.assertEqual(event.unified_msg_origin, GROUP_UMO)
        self.assertEqual(UMOInfo.parse(event.unified_msg_origin).session_type, "group")

    def test_independent_switch_matrix(self):
        destinations = (GROUP_UMO, OTHER_GROUP_UMO, PRIVATE_UMO)
        for personal in (False, True):
            for group in (False, True):
                mgr = self.manager(
                    global_memory=personal, share_group_across_groups=group
                )
                for destination in destinations:
                    event = real_message_event(
                        umo=destination,
                        sender_id="bob",
                        group_id=destination.split(":", 2)[2]
                        if ":GroupMessage:" in destination
                        else "",
                    )
                    same = destination == GROUP_UMO
                    is_group_destination = ":GroupMessage:" in destination
                    with self.subTest(
                        personal=personal, group=group, destination=destination
                    ):
                        self.assertEqual(
                            mgr._is_memory_visible(event, _memory()),
                            personal or same,
                        )
                        self.assertFalse(
                            mgr._is_memory_visible(event, _memory(owner="qq_alice"))
                        )
                        multi = _memory(
                            owner="qq_alice",
                            visibility="group",
                            owner_user_ids=["qq_alice", "qq_bob"],
                        )
                        self.assertEqual(
                            mgr._is_memory_visible(event, multi), personal or same
                        )
                        self.assertEqual(
                            mgr._is_memory_visible(
                                event, _memory(scope="group", domain="")
                            ),
                            same,
                        )
                        self.assertEqual(
                            mgr._is_memory_visible(
                                event, _memory(scope="group", domain="facts")
                            ),
                            same or (group and is_group_destination),
                        )
                        self.assertEqual(
                            mgr._is_memory_visible(
                                event, _memory(scope="group", domain="preferences")
                            ),
                            same,
                        )
                        self.assertEqual(
                            mgr._is_memory_visible(
                                event, _memory(scope="group", domain="events")
                            ),
                            same,
                        )
                        self.assertEqual(
                            mgr._is_memory_visible(
                                event, _memory(scope="conversation")
                            ),
                            same,
                        )
                        self.assertTrue(
                            mgr._is_memory_visible(event, _memory(scope="global"))
                        )
                        self.assertFalse(
                            mgr._is_memory_visible(event, _memory(deprecated=True))
                        )
                        self.assertFalse(
                            mgr._is_memory_visible(
                                event,
                                _memory(umo="tg:GroupMessage:g1", owner="tg_other_bob"),
                            )
                        )
                        self.assertFalse(
                            mgr._is_memory_visible(
                                event, _memory(scope="group", umo="tg:GroupMessage:g1")
                            )
                        )

    def test_explicit_group_search_in_private_has_no_filters(self):
        mgr = self.manager(share_group_across_groups=True)
        private_event = real_message_event(umo=PRIVATE_UMO, sender_id="bob")
        self.assertEqual(
            mgr._build_recall_filters(private_event, True, memory_scope="group"), []
        )
        broken = real_message_event(umo=":GroupMessage:g1", sender_id="bob")
        self.assertEqual(mgr._build_recall_filters(broken, True), [])
        self.assertFalse(
            mgr._is_memory_visible(
                real_message_event(umo="qq:private:g1", sender_id="g1"),
                _memory(scope="group"),
            )
        )

    def test_group_filter_never_relaxes_personal_scope(self):
        mgr = self.manager(global_memory=False, share_group_across_groups=True)
        event = _group_event(OTHER_GROUP_UMO)
        personal_filter = mgr._scope_filter(event, "personal")
        self.assertEqual(personal_filter["umo"], event.unified_msg_origin)
        self.assertEqual(personal_filter["owner_user_id"], "qq_bob")
        group_filter = mgr._scope_filter(event, "group")
        self.assertEqual(group_filter["owner_session_id"], "qq_g2")
        self.assertNotIn("platform_id", group_filter)

        recall_filters = mgr._build_recall_filters(event, True)
        cross = [f for f in recall_filters if f.get("platform_id") == "qq"]
        self.assertEqual({f["domain"] for f in cross}, {"fact", "facts"})
        self.assertTrue(all("owner_session_id" not in f for f in cross))

        # 显式 domain 参数保持精确语义：fact 只查 fact，facts 只查 facts
        fact_filters = mgr._build_recall_filters(event, True, domain="fact")
        self.assertEqual(
            {f["domain"] for f in fact_filters if f.get("platform_id") == "qq"},
            {"fact"},
        )
        facts_filters = mgr._build_recall_filters(event, True, domain="facts")
        self.assertEqual(
            {f["domain"] for f in facts_filters if f.get("platform_id") == "qq"},
            {"facts"},
        )
        pref_filters = mgr._build_recall_filters(event, True, domain="preferences")
        self.assertEqual([f for f in pref_filters if f.get("platform_id") == "qq"], [])
        self.assertEqual(mgr._build_user_filter(event), {"user_id": "qq_bob"})


class SharingReadTests(RealStackTestCase):
    async def restart(self, **kwargs):
        server = getattr(self, "server", None)
        if server is not None:
            server.shutdown()
            server.server_close()
        await self.start_stack(with_decision=False, **kwargs)

    def stack_config(self, **overrides):
        return {
            "use_reranker": False,
            "recall_sparse_fusion": False,
            "max_memories_per_inject": 5,
            **overrides,
        }

    async def test_list_does_not_bypass_sharing_or_owner_checks(self):
        seeded = [
            ("foreign", "喜欢咖啡 coffee", _meta("facts://foreign", owner="qq_alice")),
            ("own", "喜欢咖啡 coffee", _meta("facts://own")),
            ("group", "喜欢咖啡 coffee", _meta("facts://group", scope="group")),
            ("global", "喜欢咖啡 coffee", _meta("facts://global", scope="global")),
        ]
        for personal in (False, True):
            await self.restart(
                config=self.stack_config(global_memory=personal),
                memories=seeded,
            )
            event = real_message_event(umo=PRIVATE_UMO, sender_id="bob")
            visible, total, truncated = await self.mgr.list_memories(event)
            expected = {"facts://global"} | ({"facts://own"} if personal else set())
            self.assertEqual({d["metadata"]["uri"] for d in visible}, expected)
            self.assertEqual(total, len(expected))
            self.assertFalse(truncated)

    async def test_linked_foreign_private_memory_is_not_injected(self):
        await self.start_stack(
            with_decision=False,
            config=self.stack_config(global_memory=False),
            memories=[
                ("global", "喜欢咖啡 coffee", _meta("facts://global", scope="global")),
                (
                    "foreign",
                    "喜欢咖啡 coffee",
                    _meta("facts://foreign", owner="qq_alice"),
                ),
                ("own", "喜欢咖啡 coffee", _meta("facts://own", umo=PRIVATE_UMO)),
            ],
        )
        await self.mgr._link_manager.add_link(
            "facts://global", "facts://foreign", "related"
        )
        await self.mgr._link_manager.add_link(
            "facts://global", "facts://own", "related"
        )
        event = real_message_event(umo=PRIVATE_UMO, sender_id="bob")
        main_memory = {
            "text": "喜欢咖啡 coffee",
            "metadata": await self.doc_meta("facts://global"),
        }

        linked = await self.mgr._inject_linked_memories([main_memory], event)

        self.assertEqual({m["metadata"]["uri"] for m in linked}, {"facts://own"})
        recalled = await self.mgr.recall_memories(event, "coffee", top_k=5)
        self.assertNotIn("facts://foreign", {m["metadata"]["uri"] for m in recalled})

    async def test_shared_personal_forget_requires_primary_owner(self):
        await self.start_stack(
            with_decision=False, config=self.stack_config(global_memory=True)
        )
        await self.mgr.store_memory(
            _group_event(sender="alice"),
            "喜欢咖啡",
            "facts",
            uri="facts://coffee",
            owner_sender_ids=["alice", "bob"],
            allowed_sender_ids=["alice", "bob"],
        )
        shared = await self.doc_meta("facts://coffee")
        self.assertEqual(shared["owner_user_id"], "qq_alice")
        self.assertEqual(shared["owner_user_ids"], ["qq_alice", "qq_bob"])
        self.assertEqual(shared["visibility"], "group")

        secondary = _group_event(sender="bob")
        self.assertEqual(await self.mgr.forget_memory(secondary, "facts://coffee"), 0)
        self.assertEqual(
            (await self.doc_meta("facts://coffee"))["uri"], "facts://coffee"
        )

        primary = _group_event(sender="alice")
        self.assertEqual(await self.mgr.forget_memory(primary, "facts://coffee"), 1)
        self.assertFalse(
            await self.mgr.vec_db.document_storage.get_documents(
                metadata_filters={"uri": "facts://coffee"}, limit=1
            )
        )

    async def test_membership_query_filters_before_limit_and_rechecks_switches(self):
        own = (
            "own",
            "喜欢咖啡 coffee",
            _meta(
                "facts://own",
                owner="qq_alice",
                visibility="group",
                owner_user_ids=["qq_alice", "qq_bob"],
            ),
        )
        foreign = [
            (
                f"foreign-{index}",
                "喜欢咖啡 coffee",
                _meta(
                    uri=f"facts://foreign-{index}",
                    owner="qq_carol",
                    visibility="group",
                    owner_user_ids=["qq_carol", "qq_dan"],
                ),
            )
            for index in range(13)
        ]
        await self.start_stack(
            with_decision=False,
            config=self.stack_config(global_memory=True),
            memories=[own, *foreign],
        )
        # 本群之外的候选不得挤占 limit 窗口：成员过滤发生在 limit 之前
        found = await self.mgr._shared_personal_documents(_group_event(), limit=1)
        self.assertEqual([d["metadata"]["uri"] for d in found], ["facts://own"])

        shared = await self.mgr._retrieve_shared_personal(
            _group_event(), "coffee", 5, None
        )
        self.assertEqual([d["metadata"]["uri"] for d in shared], ["facts://own"])

        self.mgr.config["global_memory"] = False
        friend_event = real_message_event(umo=PRIVATE_UMO, sender_id="bob")
        self.assertEqual(
            await self.mgr._shared_personal_documents(friend_event, limit=1), []
        )
        self.assertEqual(
            len(await self.mgr._shared_personal_documents(_group_event(), limit=1)),
            1,
        )

        recalled = await self.mgr.recall_memories(_group_event(), "coffee")
        self.assertIn("facts://own", {m["metadata"]["uri"] for m in recalled})
        self.assertFalse(
            any(m["metadata"]["uri"].startswith("facts://foreign") for m in recalled)
        )

    async def test_final_read_guard_applies_even_after_candidate_selection(self):
        await self.start_stack(
            with_decision=False,
            config=self.stack_config(global_memory=False),
            memories=[
                ("global", "喜欢咖啡 coffee", _meta("facts://global", scope="global")),
                (
                    "foreign",
                    "喜欢咖啡 coffee",
                    _meta("facts://foreign", owner="qq_alice"),
                ),
                (
                    "shared-other-umo",
                    "喜欢咖啡 coffee",
                    _meta(
                        "facts://shared-other-umo",
                        owner="qq_alice",
                        umo="qq:GroupMessage:g9",
                        visibility="group",
                        owner_user_ids=["qq_alice", "qq_bob"],
                    ),
                ),
            ],
        )
        event = real_message_event(umo=PRIVATE_UMO, sender_id="bob")

        recalled = await self.mgr.recall_memories(event, "coffee", top_k=5)
        self.assertTrue(recalled)
        self.assertEqual(
            {item["metadata"]["memory_scope"] for item in recalled}, {"global"}
        )

        candidates = [
            {"text": "喜欢咖啡 coffee", "metadata": await self.doc_meta(uri)}
            for uri in ("facts://global", "facts://foreign", "facts://shared-other-umo")
        ]
        visible = self.mgr._filter_visible_shared_personal(event, candidates)
        self.assertEqual([d["metadata"]["uri"] for d in visible], ["facts://global"])

        sharing_mgr = MemoryManager(
            self.context.kb_manager,
            {"kb_name": self.kb_name, "global_memory": True},
        )
        sharing_mgr.initialize()
        self.assertTrue(
            sharing_mgr._is_memory_visible(
                event, {"metadata": candidates[2]["metadata"]}
            )
        )

    async def test_cross_group_prefilter_sql_only_matches_facts(self):
        rows = [
            (
                "own-pref",
                "喜欢咖啡 coffee",
                _meta(
                    "facts://own-pref",
                    scope="group",
                    domain="preferences",
                    umo=OTHER_GROUP_UMO,
                ),
            ),
            (
                "own-fact",
                "喜欢咖啡 coffee",
                _meta(
                    "facts://own-fact",
                    scope="group",
                    domain="facts",
                    umo=OTHER_GROUP_UMO,
                ),
            ),
            (
                "other-fact",
                "喜欢咖啡 coffee",
                _meta(
                    "facts://other-fact", scope="group", domain="facts", umo=GROUP_UMO
                ),
            ),
            (
                "other-fact-legacy",
                "喜欢咖啡 coffee",
                _meta(
                    "facts://other-fact-legacy",
                    scope="group",
                    domain="fact",
                    umo=GROUP_UMO,
                ),
            ),
            (
                "other-pref",
                "喜欢咖啡 coffee",
                _meta(
                    "facts://other-pref",
                    scope="group",
                    domain="preferences",
                    umo=GROUP_UMO,
                ),
            ),
            (
                "other-event",
                "喜欢咖啡 coffee",
                _meta(
                    "facts://other-event", scope="group", domain="events", umo=GROUP_UMO
                ),
            ),
            (
                "other-nodomain",
                "喜欢咖啡 coffee",
                _meta(
                    "facts://other-nodomain", scope="group", domain="", umo=GROUP_UMO
                ),
            ),
        ]
        await self.start_stack(
            with_decision=False,
            config=self.stack_config(share_group_across_groups=True),
            memories=rows,
        )

        async def run_filters(filters):
            hits = {}
            for item in filters:
                docs = await self.mgr.vec_db.document_storage.get_documents(
                    metadata_filters=item, limit=100
                )
                for doc in docs:
                    metadata = doc.get("metadata", {})
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                    uri = metadata["uri"]
                    hits[uri] = hits.get(uri, 0) + 1
            return hits

        event = _group_event(OTHER_GROUP_UMO)
        filters_list = self.mgr._build_recall_filters(event, True)
        hits = await run_filters(filters_list)
        # 本群记忆不受限，跨群 fact/facts 新旧别名均可召回
        self.assertEqual(
            set(hits),
            {
                "facts://own-pref",
                "facts://own-fact",
                "facts://other-fact",
                "facts://other-fact-legacy",
            },
        )
        self.assertEqual(
            await run_filters(
                self.mgr._build_recall_filters(event, True, domain="fact")
            ),
            {"facts://other-fact-legacy": 1},
        )
        self.assertEqual(
            set(
                await run_filters(
                    self.mgr._build_recall_filters(event, True, domain="facts")
                )
            ),
            {"facts://own-fact", "facts://other-fact"},
        )
        self.assertEqual(
            await run_filters(
                self.mgr._build_recall_filters(event, True, domain="preferences")
            ),
            {"facts://own-pref": 1},
        )
        # 本群 facts 被本群与跨群过滤器重复命中，按 uri 去重后分页不受影响
        self.assertGreater(hits["facts://own-fact"], 1)

        # 稀疏通道本地过滤与 SQL 预过滤一致
        for _, _, metadata in rows:
            local_hit = any(
                MemoryManager._matches_filters(metadata, filters)
                for filters in filters_list
            )
            self.assertEqual(metadata["uri"] in hits, local_hit, metadata["uri"])
