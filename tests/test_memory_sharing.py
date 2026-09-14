"""共享开关、关联权限和多主体候选查询回归。"""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from test_memory_commands import MemoryManager, plugin_main

UMOInfo = plugin_main.UMOInfo


class Event:
    def __init__(self, umo="sl:FriendMessage:bob", sender="bob"):
        self.unified_msg_origin = umo
        self.sender = sender

    def get_sender_id(self):
        return self.sender

    def get_platform_id(self):
        return self.unified_msg_origin.split(":", 1)[0]


def memory(scope="personal", owner="sl_bob", umo="sl:GroupMessage:g1", **extra):
    parsed = UMOInfo.parse(umo)
    return {
        "text": "喜欢咖啡 coffee",
        "metadata": {
            "uri": "facts://coffee",
            "is_memory_record": True,
            "deprecated": False,
            "memory_scope": scope,
            "owner_user_id": owner,
            "owner_user_ids": [owner],
            "platform_id": parsed.platform_id,
            "owner_session_id": f"{parsed.platform_id}_{parsed.session_id}",
            "umo": umo,
            "visibility": "private",
            **extra,
        },
    }


class SharingConfigTests(unittest.TestCase):
    def test_legacy_key_and_new_defaults_are_preserved(self):
        root = Path(__file__).resolve().parents[1]
        schema = json.loads((root / "_conf_schema.json").read_text())
        self.assertEqual(schema["global_memory"]["type"], "bool")
        self.assertNotIn("share_personal_across_sessions", schema)
        self.assertIs(schema["share_group_across_groups"]["default"], False)
        upgraded = {key: item.get("default") for key, item in schema.items()}
        upgraded["global_memory"] = False
        mgr = MemoryManager(None, upgraded)
        self.assertFalse(mgr._is_memory_visible(Event(), memory()))
        for locale in ("zh-CN", "en-US"):
            translations = json.loads(
                (root / ".astrbot-plugin" / "i18n" / f"{locale}.json").read_text()
            )["config"]
            for key in (
                "global_memory",
                "share_group_across_groups",
                "recall_query_max_chars",
                "recall_timeout_seconds",
            ):
                self.assertTrue(translations[key]["description"])
                self.assertTrue(translations[key]["hint"])


class SharingPolicyTests(unittest.TestCase):
    def test_umo_normalizes_without_rewriting_origin(self):
        for raw, expected in (
            ("GroupMessage", "group"),
            ("FriendMessage", "private"),
            ("OtherMessage", "other"),
            ("group", "group"),
        ):
            parsed = UMOInfo.parse(f"sl:{raw}:id:with:colons")
            self.assertEqual(parsed.session_type, expected)
            self.assertEqual(parsed.session_id, "id:with:colons")
            self.assertTrue(parsed.is_valid)
        for raw in ("HEX", "sl::id", "sl:unknown:id", ":group:id", "sl:private:"):
            self.assertFalse(UMOInfo.parse(raw).is_valid)

    def test_independent_switch_matrix(self):
        for personal in (False, True):
            for group in (False, True):
                mgr = MemoryManager(
                    None,
                    {"global_memory": personal, "share_group_across_groups": group},
                )
                for destination in (
                    "sl:GroupMessage:g1",
                    "sl:GroupMessage:g2",
                    "sl:FriendMessage:bob",
                ):
                    event = Event(destination)
                    with self.subTest(
                        personal=personal, group=group, destination=destination
                    ):
                        same = destination == "sl:GroupMessage:g1"
                        self.assertEqual(
                            mgr._is_memory_visible(event, memory()), personal or same
                        )
                        self.assertFalse(
                            mgr._is_memory_visible(event, memory(owner="sl_alice"))
                        )
                        multi = memory(
                            owner="sl_alice",
                            visibility="group",
                            owner_user_ids=["sl_alice", "sl_bob"],
                        )
                        self.assertEqual(
                            mgr._is_memory_visible(event, multi), personal or same
                        )
                        self.assertEqual(
                            mgr._is_memory_visible(event, memory(scope="group")),
                            same or (group and ":GroupMessage:" in destination),
                        )
                        self.assertEqual(
                            mgr._is_memory_visible(event, memory(scope="conversation")),
                            same,
                        )
                        self.assertTrue(
                            mgr._is_memory_visible(event, memory(scope="global"))
                        )
                        self.assertFalse(
                            mgr._is_memory_visible(event, memory(deprecated=True))
                        )
                        self.assertFalse(
                            mgr._is_memory_visible(
                                event,
                                memory(umo="other:GroupMessage:g1", owner="other_bob"),
                            )
                        )
                        self.assertFalse(
                            mgr._is_memory_visible(
                                event,
                                memory(scope="group", umo="other:GroupMessage:g1"),
                            )
                        )

    def test_explicit_group_search_in_private_has_no_filters(self):
        mgr = MemoryManager(None, {"share_group_across_groups": True})
        self.assertEqual(
            mgr._build_recall_filters(Event(), True, memory_scope="group"), []
        )
        self.assertEqual(mgr._build_recall_filters(Event("HEX"), True), [])
        self.assertFalse(
            mgr._is_memory_visible(
                Event("sl:FriendMessage:g1", "g1"), memory(scope="group")
            )
        )

    def test_group_filter_never_relaxes_personal_scope(self):
        mgr = MemoryManager(
            None, {"global_memory": False, "share_group_across_groups": True}
        )
        event = Event("sl:GroupMessage:g2")
        personal_filter = mgr._scope_filter(event, "personal")
        self.assertEqual(personal_filter["umo"], event.unified_msg_origin)
        self.assertEqual(personal_filter["owner_user_id"], "sl_bob")
        group_filter = mgr._scope_filter(event, "group")
        self.assertEqual(group_filter["platform_id"], "sl")
        self.assertNotIn("owner_session_id", group_filter)
        self.assertEqual(mgr._build_user_filter(event), {"user_id": "sl_bob"})


class SharingReadTests(unittest.IsolatedAsyncioTestCase):
    async def test_list_does_not_bypass_sharing_or_owner_checks(self):
        for personal in (False, True):
            mgr = MemoryManager(
                None, {"global_memory": personal, "share_group_across_groups": True}
            )
            docs = [
                memory(owner="sl_alice", uri="facts://foreign"),
                memory(),
                memory(scope="group", uri="facts://group"),
                memory(scope="global", uri="facts://global"),
            ]
            mgr._kb_helper = SimpleNamespace(
                vec_db=SimpleNamespace(
                    document_storage=SimpleNamespace(
                        get_documents=AsyncMock(return_value=docs)
                    )
                )
            )
            mgr._shared_personal_documents = AsyncMock(return_value=[])
            visible, total, truncated = await mgr.list_memories(Event())
            expected = (
                {"facts://global", "facts://coffee"} if personal else {"facts://global"}
            )
            self.assertEqual({d["metadata"]["uri"] for d in visible}, expected)
            self.assertEqual(total, len(expected))
            self.assertFalse(truncated)

    async def test_linked_foreign_private_memory_is_not_injected(self):
        mgr = MemoryManager(None, {"global_memory": False})
        mgr._link_manager = SimpleNamespace(
            get_links_for_uri=AsyncMock(
                return_value=[
                    {"target_uri": "facts://foreign", "relation_type": "related"}
                ]
            ),
            get_links_to_uri=AsyncMock(return_value=[]),
        )
        for candidate in (memory(owner="sl_alice"), memory()):
            mgr._get_memory_by_uri = AsyncMock(return_value=candidate)
            self.assertEqual(
                await mgr._inject_linked_memories([memory(scope="global")], Event()), []
            )

    async def test_membership_query_filters_before_limit_and_rechecks_switches(self):
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        self.addAsyncCleanup(engine.dispose)
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "CREATE TABLE documents (id INTEGER PRIMARY KEY, text TEXT, metadata TEXT)"
                )
            )
            own = memory(
                owner="sl_alice",
                visibility="group",
                owner_user_ids=["sl_alice", "sl_bob"],
            )
            await conn.execute(
                text("INSERT INTO documents VALUES (1, :text, :metadata)"),
                {"text": own["text"], "metadata": json.dumps(own["metadata"])},
            )
            for i in range(2, 15):
                foreign = memory(
                    owner="sl_carol",
                    visibility="group",
                    owner_user_ids=["sl_carol", "sl_dan"],
                )
                await conn.execute(
                    text("INSERT INTO documents VALUES (:id, :text, :metadata)"),
                    {
                        "id": i,
                        "text": foreign["text"],
                        "metadata": json.dumps(foreign["metadata"]),
                    },
                )
        mgr = MemoryManager(None, {"global_memory": True})
        mgr._kb_helper = SimpleNamespace(
            vec_db=SimpleNamespace(
                document_storage=SimpleNamespace(
                    get_session=lambda: AsyncSession(engine)
                )
            )
        )
        found = await mgr._shared_personal_documents(Event(), limit=1)
        self.assertEqual([d["id"] for d in found], [1])
        mgr.has_any_active_memory = AsyncMock(return_value=True)
        mgr._retrieve_with_filter = AsyncMock(return_value=[])
        mgr._sparse_tokenize = False
        recalled = await mgr.recall_memories(Event(), "coffee")
        self.assertEqual(
            [doc["metadata"]["uri"] for doc in recalled], ["facts://coffee"]
        )
        mgr.config["global_memory"] = False
        mgr.config["share_group_across_groups"] = True
        self.assertEqual(await mgr._shared_personal_documents(Event(), limit=1), [])
        self.assertEqual(
            len(
                await mgr._shared_personal_documents(
                    Event("sl:GroupMessage:g1"), limit=1
                )
            ),
            1,
        )

    async def test_final_read_guard_applies_even_after_candidate_selection(self):
        mgr = MemoryManager(None, {"global_memory": False})
        mgr._kb_helper = SimpleNamespace()
        mgr.has_any_active_memory = AsyncMock(return_value=True)
        mgr._retrieve_with_filter = AsyncMock(
            return_value=[memory(owner="sl_alice"), memory(scope="global")]
        )
        mgr._retrieve_shared_personal = AsyncMock(return_value=[])
        mgr._link_manager = None
        result = await mgr.recall_memories(Event(), "coffee", top_k=5)
        self.assertTrue(result)
        self.assertTrue(
            all(item["metadata"]["memory_scope"] == "global" for item in result)
        )
