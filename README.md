# AstrBot 简单长期记忆插件

为 AstrBot 提供简易的长期记忆能力，基于内置知识库实现用户偏好、历史交互和重要事实的记忆存储与召回。

## 功能特性

- **记忆存储与召回**：自动在 LLM 请求前注入相关记忆上下文
- **用户隔离**：每个用户的记忆完全隔离，互不干扰
- **全局/会话记忆**：支持跨会话的全局记忆模式和仅当前会话的记忆模式
- **LLM 工具**：提供 `memory_recall`、`memory_store`、`memory_forget` 工具供 AI 主动操作
- **用户命令**：通过 `/memory` 命令管理记忆

## 安装

1. 在 AstrBot 插件市场安装本插件
2. 在知识库管理中创建一个用于存储记忆的知识库（需配置嵌入模型）
3. 在插件设置中配置：
   - **记忆知识库**：选择创建的知识库

## 配置说明

| 配置项 | 说明 | 必填 |
|--------|------|------|
| kb_name | 记忆知识库 | 是 |
| extraction_provider_id | 记忆提取模型（留空使用会话主LLM） | 否 |
| summarization_provider_id | 记忆总结模型（留空使用会话主LLM） | 否 |
| auto_memorize | 自动记忆模式开关 | 否 |
| extraction_interval | 记忆提取间隔（每N轮对话提取一次，范围5-200，默认20） | 否 |
| extraction_min_content_length | 最小提取内容长度（低于此值跳过提取，默认500） | 否 |
| global_memory | 全局记忆模式 | 否 |
| max_memories_per_inject | 每次注入的记忆数量 | 否 |
| max_memories_per_recall | 记忆召回数量 | 否 |
| memory_domains | 记忆域配置 | 否 |
| memory_ttl_days | 记忆生命周期(天) | 否 |

## 使用方法

### 用户命令

```
/memory list [domain]    - 列出记忆
/memory search <query>   - 搜索记忆
/memory forget <uri>     - 删除记忆
/memory clear            - 清空所有记忆
/memory stats            - 查看统计
```

### LLM 工具

AI 可以通过以下工具主动操作记忆：

- `memory_recall(query)` - 搜索长期记忆
- `memory_store(content, memory_type, disclosure)` - 存储记忆
- `memory_forget(uri)` - 删除记忆

## 工作原理

1. **记忆注入**：在每次 LLM 请求前，根据用户输入召回相关记忆并注入到用户消息的最前面（prompt 前置）
2. **用户隔离**：通过 metadata 中的 `user_id` 字段实现用户级别的记忆隔离
3. **记忆存储**：记忆以向量形式存储在知识库中，支持语义检索

## 注意事项

- 请确保先创建知识库并配置嵌入模型
- 记忆数据存储在知识库中，删除知识库将丢失所有记忆


<div align="center">

**如果这个插件对你有帮助，请给个 ⭐ Star 支持一下！**

</div>
