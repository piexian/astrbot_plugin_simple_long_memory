# AstrBot 简单长期记忆插件

为 AstrBot 提供长期记忆能力，基于内置知识库实现用户偏好、历史交互和重要事实的记忆存储与召回。

## 功能特性

- **自动记忆提取**：每隔 N 轮对话自动调用 LLM 从对话中提取值得记忆的信息
- **记忆注入**：在每次 LLM 请求前，自动召回相关记忆并注入到对话上下文
- **用户隔离**：通过 `user_id` 实现用户级记忆隔离，互不干扰
- **群聊记忆作用域**：支持群聊场景下的记忆归属区分（个人记忆 vs 群共享记忆）
- **LLM 工具**：提供 `memory_recall`、`memory_store`、`memory_forget` 工具供 AI 主动操作
- **用户命令**：通过 `/memory` 指令组管理记忆
- **知识库管理**：记忆直接存储在 AstrBot 内置知识库中，可在知识库管理界面直接查看、检索、删除记忆

## 安装

**方式一**：AstrBot 插件市场搜索「简单长期记忆」安装。

**方式二**：插件界面右下角加号 → 从链接安装，输入：
```
https://github.com/piexian/astrbot_plugin_simple_long_memory
```

安装后，在知识库管理中创建一个用于存储记忆的知识库（需配置嵌入模型），然后在插件设置中选该知识库。

## 配置说明

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| kb_name | 记忆知识库（必填） | — |
| extraction_provider_id | 记忆提取 LLM 模型 | 留空使用会话主 LLM |
| summarization_provider_id | 记忆总结 LLM 模型（预留） | 留空使用会话主 LLM |
| auto_memorize | 自动记忆模式开关 | `true` |
| extraction_interval | 每 N 轮对话触发一次记忆提取 | `20` |
| extraction_min_content_length | 对话总长度低于此值时跳过提取 | `150` |
| global_memory | 全局记忆模式（跨会话召回） | `true` |
| max_memories_per_inject | 每次 LLM 请求注入的最大记忆条数 | `5` |
| max_memory_list_scan | 记忆列表扫描上限 | `200` |
| memory_delete_scan_page_size | 记忆删除扫描分页大小 | `1000` |
| memory_ttl_days | 记忆生命周期（天） | `30` |
| use_reranker | 记忆召回时启用重排序（需知识库已配置重排序模型） | `true` |
| optimize_recall_query | 启用检索优化（LLM 提炼关键词） | `false` |
| optimize_recall_query_timeout | 检索优化超时（秒） | `10` |
| enable_admin_global_memory_tool | 启用管理员全局记忆工具 | `false` |
| enable_admin_global_memory_tool | 启用管理员全局记忆工具 | `false` |
| maintenance_enabled | 后台记忆整理总开关 | `false` |
| maintenance_model_id | 整理模型（建议廉价模型） | 留空 |
| maintenance_cron | 整理周期 cron 表达式 | `0 2 * * *` |
| maintenance_max_llm_calls | 每周期 LLM 调用上限 | `50` |
| auto_purge_enabled | 自动清理废弃记忆 | `true` |
| auto_purge_after_days | 清理超期天数 | `7` |
| maintenance_organizer_enabled | 启用整理师（去重合并） | `true` |
| maintenance_analyst_enabled | 启用分析师（关联发现） | `true` |
| maintenance_reviewer_enabled | 启用审核员（复核操作） | `true` |
| maintenance_reviewer_model_id | 审核模型（留空用全局模型） | 留空 |
| context_max_rounds | 最大拉取对话轮数 | `50` |
| context_max_chars | 最大拉取字符数 | `30000` |
| context_max_age_days | 对话最大回溯天数 | `7` |
| persona_mode | 人格加载模式 | `auto` |
| review_notify_enabled | 待审通知开关 | `true` |
## 使用方法

### 用户命令

```
/memory list [--all] [页码]            - 列出记忆（支持翻页）
/memory search [--all] <关键词>        - 搜索记忆
/memory stats [--all]                  - 查看记忆统计
/memory test                           - 测试记忆读写功能（管理员）
/memory forget <uri> [--user <用户ID>] - 删除指定记忆
/memory clear [--all] [--user <用户ID>] [--confirm <确认码>] - 清空记忆（管理员，需确认码）
/memory rebuild [--to <知识库名>] [--confirm <确认码>] - 重建或迁移记忆（管理员，需确认码）
/memory rebuild --clear-cache [--confirm <确认码>] - 清理重建缓存（管理员，需确认码）
```

- `test`、`clear`、`rebuild` 需要管理员权限
- `forget`：普通用户可删除自己的记忆，管理员可删除任意记忆
- `--all`：管理员可查看/搜索/统计/清空所有用户的记忆
- `--user <用户ID>`：管理员可删除/清空指定用户的记忆（`--all` 与 `--user` 不可同时使用）

### 群聊场景

群聊中记忆按归属分为三种，机器人自动判断无需手动设置：

| 作用域 | 说明 | 日常例子 |
|--------|------|----------|
| `global` | 全局记忆，所有会话可召回（仅管理员工具可写入） | "机器人回答某项目问题时优先使用内部术语表" |
| `personal` | 个人记忆，仅自己可见 | "我比较喜欢喝拿铁"、"下周要出差" |
| `group` | 群共享记忆，群友都可见 | "群里约了每周五打游戏"、"这个群的固定梗" |
| `conversation` | 当前会话临时上下文 | "刚才说的那个 bug 还没修完" |

私聊默认召回 `personal` 记忆，并可使用 `conversation` 记录当前私聊会话上下文。

> 从旧版本升级到 v0.3 后，请执行 `/memory rebuild`。运行时召回和列表只认新 metadata 结构；旧格式记录需要通过重建补齐 `memory_scope`、owner、visibility 等字段后才会进入新作用域模型。
> 重建只处理当前知识库 `kb_id` 下的记忆记录，避免误迁移其它知识库或无法可靠归属的数据。

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
- `memory_store_global(content, memory_type, disclosure)` — 存储全局记忆（需开启 `enable_admin_global_memory_tool`，仅管理员可用）
- `memory_forget(uri)` — 删除记忆

### 记忆类型

| memory_type | 说明 |
|-------------|------|
| `fact` | 用户主动告知的客观信息 |
| `preference` | 用户表达的喜好、习惯、风格 |
| `event` | 计划、纪念日、里程碑等事件 |
| `context` | 正在进行的项目或当前状况 |

## 工作原理

1. **记忆注入**：在每次 LLM 请求前，根据用户输入通过 embedding 检索召回相关记忆；AstrBot v4.24+ 优先注入到临时用户内容区（仅本轮请求生效），旧版回退到最早的 `user` 上下文位置（不占用 system prompt，不覆盖当前输入）
2. **自动提取**：每隔 `extraction_interval` 轮对话，将累积的对话内容发送给 LLM 提取值得记忆的信息并自动存储
3. **用户隔离**：所有记忆操作通过 metadata 中的 `user_id` 字段过滤，确保用户间记忆完全隔离
4. **记忆存储**：记忆以向量形式存储在知识库中，支持语义检索
4. **记忆存储**：记忆以向量形式存储在知识库中，支持语义检索
5. **后台整理**（可选）：开启 `maintenance_enabled` 后，定时执行记忆去重合并、关联发现、矛盾检测和质量精炼，保持记忆池长期健康
### 后台记忆整理

开启 `maintenance_enabled` 后，系统会在后台定时执行记忆整理任务，保持记忆池长期健康：

**整理团队角色**：

| 角色 | 职责 | 默认状态 |
|------|------|----------|
| 📋 整理师 | 去重合并相似记忆、精炼措辞、归档无价值条目 | 启用 |
| 🔍 分析师 | 拉取对话历史，发现记忆间隐含关联，检测矛盾 | 启用 |
| ✅ 审核员 | 复核整理师和分析师的操作建议，防止误删误改 | 启用 |

**整理流程**：
1. **物理清理**：删除废弃超期的记忆（FAISS + SQLite + KB 文档记录）
2. **整理师**：余弦 ≥0.9 预筛候选对，LLM 裁决 merge/none，merge 走 supersede 语义（不物理删除）
3. **分析师**：余弦 ≥0.7 预筛候选对，排除已连边，发现关联和矛盾
4. **审核员**：复核操作建议，approve 执行 / reject 跳过 / controversial 待人工审核

**安全机制**：
- 两级预筛：候选先经向量余弦预筛，再交 LLM 裁决，不全量喂记忆池
- 裁决缓存：候选对按内容 hash 磁盘缓存，同对同模型永不重判
- 三态裁决：LLM 不可用/解析失败返回"不确定"，保持现状
- 保守原则：不确定时选择 reject，保持现状永远比误删安全
- 关联召回：只注入 related/supports/context，排除 contradicts/supersedes

**配置建议**：
- `maintenance_model_id`：建议使用廉价模型（如 gpt-4o-mini）
- `maintenance_cron`：建议设置在低峰期（如凌晨 2 点）
- `maintenance_max_llm_calls`：根据记忆数量调整，默认 50 次/周期

## 注意事项

- 请确保先创建知识库并配置嵌入模型
- 记忆数据存储在知识库中，删除知识库将丢失所有记忆
- **请勿将记忆知识库挂载到 AstrBot 全局知识库配置中**。本插件通过 `user_id` 实现用户级记忆隔离，而 AstrBot 原生知识库检索不支持用户隔离，挂载后会导致所有用户共享彼此的记忆。仅个人独占使用时可忽略此限制


### 启用方法

1. 在「使用电脑能力」中将运行环境设置为 `local` 或 `sandbox`
2. 在本插件配置中开启 **安装记忆 Skill**
3. 重启或重载插件，Skill 将自动安装到 AstrBot 的 skills 目录并激活


<div align="center">

**如果这个插件对你有帮助，请给个 Star 支持一下！**

</div>
