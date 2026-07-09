import unittest

from memory_protocol import format_memory_for_user


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


if __name__ == "__main__":
    unittest.main()
