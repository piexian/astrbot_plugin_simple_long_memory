import unittest

from memory_protocol import format_memory_for_injection, format_memory_for_user


def _memory(index: int) -> dict:
    return {
        "text": f"memory: 记忆 {index}",
        "metadata": {
            "uri": f"facts://memory-{index}",
            "created_at": "2026-07-10T00:00:00+00:00",
        },
    }


class FormatMemoryForUserTests(unittest.TestCase):
    def test_empty_memory_store(self):
        result = format_memory_for_user([], page=1, total=0)

        self.assertEqual(result, "暂无记忆")

    def test_out_of_range_page(self):
        result = format_memory_for_user([], page=3, total=20)

        self.assertEqual(result, "当前分页不存在")

    def test_last_page_does_not_show_next_page_hint(self):
        result = format_memory_for_user(
            [_memory(index) for index in range(11, 21)],
            page=2,
            total=20,
        )

        self.assertNotIn("查看下一页", result)

    def test_non_last_global_page_shows_next_page_hint(self):
        result = format_memory_for_user(
            [_memory(index) for index in range(1, 11)],
            page=1,
            total=20,
            all_mode=True,
        )

        self.assertIn("提示: /memory list --all 2 查看下一页", result)

    def test_injection_uses_configured_timezone_and_update_state(self):
        result = format_memory_for_injection(
            [
                {
                    "text": "memory: 用户喜欢茶",
                    "metadata": {
                        "domain": "preferences",
                        "memory_scope": "personal",
                        "owner_user_id": "telegram_alice",
                        "owner_user_ids": ["telegram_alice", "telegram_bob"],
                        "visibility": "group",
                        "_is_linked": True,
                        "_linked_relation_types": ["related"],
                        "created_at": "2026-08-18T16:30:00+00:00",
                        "updated_at": "2026-08-18T17:00:00+00:00",
                        "updated_by": "organizer",
                        "version": 2,
                    },
                }
            ],
            timezone_name="Asia/Shanghai",
            viewer_user_id="telegram_alice",
        )

        self.assertIn("owner: current user; associated users: telegram_bob", result)
        self.assertIn("created: 2026-08-19 00:30 Asia/Shanghai", result)
        self.assertIn("updated: 2026-08-19 01:00 Asia/Shanghai by organizer", result)
        self.assertIn("relation: related", result)

    def test_injection_marks_curated_merge(self):
        result = format_memory_for_injection(
            [
                {
                    "text": "memory: 合并后的事实",
                    "metadata": {
                        "domain": "facts",
                        "memory_scope": "global",
                        "merged_from": ["facts://a", "facts://b"],
                        "curated_at": "2026-08-18T16:30:00+00:00",
                        "created_at": "2026-08-18T16:30:00+00:00",
                    },
                }
            ],
            timezone_name="Asia/Shanghai",
        )

        self.assertIn(
            "curated: merged 2 records on 2026-08-19 00:30 Asia/Shanghai", result
        )


if __name__ == "__main__":
    unittest.main()
