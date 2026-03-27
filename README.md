# AstrBot 简单长期记忆插件

为 AstrBot 提供简易的长期记忆能力，基于内置知识库实现用户偏好、历史交互和重要事实的记忆存储与召回。

## 功能特性

- **自动记忆提取**：每隔 N 轮对话自动调用 LLM 从对话中提取值得记忆的信息
- **记忆注入**：在每次 LLM 请求前，自动召回相关记忆并注入到对话上下文
- **用户隔离**：通过 `user_id` 实现用户级记忆隔离，互不干扰
- **全局/会话记忆**：支持跨会话的全局记忆模式和仅当前会话的记忆模式
- **LLM 工具**：提供 `memory_recall`、`memory_store`、`memory_forget` 工具供 AI 主动操作
- **用户命令**：通过 `/memory` 指令组管理记忆

## 安装

1. 在 AstrBot 插件市场安装本插件
2. 在知识库管理中创建一个用于存储记忆的知识库（需配置嵌入模型）
3. 在插件设置中配置：
   - **记忆知识库**：选择创建的知识库

## 配置说明

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| kb_name | 记忆知识库（必填） | — |
| extraction_provider_id | 记忆提取 LLM 模型 | 留空使用会话主 LLM |
| summarization_provider_id | 记忆总结 LLM 模型（预留） | 留空使用会话主 LLM |
| auto_memorize | 自动记忆模式开关 | `true` |
| extraction_interval | 每 N 轮对话触发一次记忆提取 | `20` |
| extraction_min_content_length | 对话总长度低于此值时跳过提取 | `500` |
| global_memory | 全局记忆模式（跨会话召回） | `true` |
| max_memories_per_inject | 每次 LLM 请求注入的最大记忆条数 | `5` |
| memory_domains | 记忆分类域 | `["user_profile", "preferences", "facts", "events", "context"]` |
| memory_ttl_days | 记忆生命周期（天） | `30` |
| install_skill | 安装 AI 记忆指南 Skill | `false` |
| use_reranker | 记忆召回时启用重排序（需知识库已配置重排序模型） | `true` |
| optimize_recall_query | 启用检索优化（LLM 提炼关键词） | `false` |

## 使用方法

### 用户命令

```
/memory list [--all] [页码]            - 列出记忆（支持翻页）
/memory search [--all] <关键词>        - 搜索记忆
/memory stats [--all]                  - 查看记忆统计
/memory test                           - 测试记忆读写功能（管理员）
/memory forget <uri> [--user <用户ID>] - 删除指定记忆（管理员）
/memory clear [--all] [--user <用户ID>]- 清空记忆（管理员）
/memory rebuild [--to <知识库名>]      - 重建或迁移记忆（管理员）
/memory rebuild --clear-cache          - 清理重建缓存（管理员）
```

- `test`、`forget`、`clear`、`rebuild` 需要管理员权限
- `--all`：管理员可查看/搜索/统计/清空所有用户的记忆
- `--user <用户ID>`：管理员可删除/清空指定用户的记忆（`--all` 与 `--user` 不可同时使用）
- 无标志时行为不变，仅操作当前用户数据

### 记忆重建与迁移

`/memory rebuild` 提供两个能力：

- **原地重建**：将所有记忆重新嵌入写入当前知识库，适用于修复损坏的向量数据
- **迁移**：将所有记忆迁移到目标知识库，适用于切换知识库或更换嵌入模型（目标知识库需配置好新的嵌入模型）

```
/memory rebuild                        # 原地重建（重新嵌入）
/memory rebuild --to <知识库名>        # 迁移到目标知识库
/memory rebuild --clear-cache          # 清理重建缓存
```

**工作流程**：
1. 拉取所有记忆到本地并持久化到 KV 数据库（防崩溃丢失）
2. 原地重建：清空当前 KB → 重新嵌入写入；迁移：写入目标 KB → 清空源 KB
3. 重建期间新产生的记忆会被缓冲，完成后批量语义去重写入
4. 自动进行完整性校验，对比预期与实际记忆数量
5. 确认数据无误后，手动执行 `--clear-cache` 清理缓存

**安全机制**：
- 本地优先：先拉全量数据再执行破坏性操作
- 进程崩溃恢复：中间状态通过 KV 持久化，重启后自动恢复
- 迁移安全保护：写入失败时不清空源知识库，防止数据丢失
- 手动缓存清理：需用户确认数据无误后才清理缓存

### LLM 工具

AI 可以通过以下工具主动操作记忆：

- `memory_recall(query)` — 搜索长期记忆
- `memory_store(content, memory_type, disclosure)` — 存储记忆
- `memory_forget(uri)` — 删除记忆

### 记忆类型

| memory_type | 说明 |
|-------------|------|
| `fact` | 用户主动告知的客观信息 |
| `preference` | 用户表达的喜好、习惯、风格 |
| `event` | 计划、纪念日、里程碑等事件 |
| `context` | 正在进行的项目或当前状况 |

## 工作原理

1. **记忆注入**：在每次 LLM 请求前，根据用户输入通过 embedding 检索召回相关记忆，以 `user` 角色注入到对话上下文顶部（不占用 system prompt）
2. **自动提取**：每隔 `extraction_interval` 轮对话，将累积的对话内容发送给 LLM 提取值得记忆的信息并自动存储
3. **用户隔离**：所有记忆操作通过 metadata 中的 `user_id` 字段过滤，确保用户间记忆完全隔离
4. **记忆存储**：记忆以向量形式存储在知识库中，支持语义检索

## 注意事项

- 请确保先创建知识库并配置嵌入模型
- 记忆数据存储在知识库中，删除知识库将丢失所有记忆
- **请勿将记忆知识库挂载到 AstrBot 全局知识库配置中**。本插件通过 `user_id` 实现用户级记忆隔离，而 AstrBot 原生知识库检索不支持用户隔离，挂载后会导致所有用户共享彼此的记忆。仅个人独占使用时可忽略此限制

## AI 记忆 Skill（可选）

本插件内置了一个 Skill 文件，可引导 AI 主动使用记忆工具（而非被动等待调用）。

### 启用方法

1. 在「使用电脑能力」中将运行环境设置为 `local` 或 `sandbox`
2. 在本插件配置中开启 **安装记忆 Skill**
3. 重启或重载插件，Skill 将自动安装到 AstrBot 的 skills 目录并激活


<div align="center">

**如果这个插件对你有帮助，请给个 Star 支持一下！**

</div>
