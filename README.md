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
| enable_admin_global_memory_tool | 启用管理员全局记忆写入及全 scope 精确搜索、确认删除工具 | `false` |
| maintenance_enabled | 后台记忆整理总开关 | `false` |
| maintenance_model_id | 整理模型（建议廉价模型） | 留空 |
| maintenance_cron | 整理周期 cron 表达式 | `0 3 * * *` |
| maintenance_max_llm_calls | 每周期 LLM 调用上限 | `50` |
| auto_purge_enabled | 自动清理废弃记忆 | `true` |
| auto_purge_after_days | 清理超期天数 | `7` |
| maintenance_organizer_enabled | 启用整理师（去重合并） | `true` |
| maintenance_analyst_enabled | 启用分析师（关联发现） | `true` |
| maintenance_reviewer_enabled | 启用审核员（复核操作） | `true` |
| maintenance_reviewer_model_id | 审核模型（留空用全局模型） | 留空 |
| maintenance_analyst_max_contradictions | 分析师每次最多报告矛盾数 | `20` |
| maintenance_window | 整理执行时间窗口；cron 命中窗口外时跳过并记 DEBUG | `02:00-06:00` |
| maintenance_max_ops_per_cycle | 每周期操作数硬上限 | `100` |
| maintenance_pending_queue_max | 待审队列容量上限 | `500` |
| context_max_rounds | 最大拉取对话轮数 | `50` |
| context_max_chars | 最大拉取字符数 | `30000` |
| context_max_age_days | 对话最大回溯天数 | `7` |
| persona_mode | 人格加载模式 | `auto` |
| review_notify_enabled | 待审通知开关 | `true` |
## 使用方法

### 用户命令

```
/memory list [--all] [页码]                                  - 列出记忆（默认第 1 页）
/memory search [--all] <关键词>                            - 搜索记忆
/memory stats [--all]                                      - 查看记忆统计
/memory test [purge|organizer|analyst|reviewer|cycle]     - 管理员测试（后台项固定 dry-run）
/memory forget <URI> [--user <用户ID>]                    - 删除指定记忆
/memory clear [--all|--user <用户ID>] [--confirm <确认码>]  - 清空记忆（管理员）
/memory rebuild [--to <知识库名>] [--confirm <确认码>]     - 重建或迁移记忆（管理员）
/memory rebuild --clear-cache [--confirm <确认码>]        - 清理重建缓存（管理员）
/memory review [list|approve <ID>|reject <ID>|clear]       - 处理后台待审队列（管理员）
```

- 所有命令前缀跟随 AstrBot 的 `wake_prefix` 配置，下面以 `/` 为例。参数中的 `<...>` 表示必填值，`[...]` 表示可选值。
- 未知参数、缺少参数值、禁止的位置参数都会直接报错；参数值含空格时请使用引号。
- `test`、`clear`、`rebuild`、`review` 需要管理员权限；`list`、`search`、`stats` 默认只操作当前用户可见范围。
- `--all` 仅管理员可用；`--all` 与 `--user` 不能同时使用。
- 后台测试项 `purge`、`organizer`、`analyst`、`reviewer`、`cycle` 会调用真实分析链路，但固定为 dry-run，不写入记忆、向量、关联或待审队列。

#### 命令参数
| 命令 | 参数说明 | 默认行为与限制 |
|------|----------|----------------|
| `list` | `[--all]`、`[页码]` | 页码必须是正整数，默认 `1`，每页 `10` 条；`--all` 查看全部用户，管理员专用。 |
| `search` | `[--all]`、`<关键词>` | 关键词必填；多个词会作为一个搜索短语；`--all` 查看全部用户，管理员专用。 |
| `stats` | `[--all]` | 无参数时统计当前用户；`--all` 统计全部用户，管理员专用；不接受位置参数。 |
| `test` | `[purge\|organizer\|analyst\|reviewer\|cycle]` | 无子参数执行记忆写入、召回、删除测试；指定后台阶段时固定 dry-run；管理员专用。 |
| `forget` | `<URI>`、`[--user <用户ID>]` | URI 必填；普通用户只能删除自己的 URI；管理员不带 `--user` 按 URI 删除全部匹配记录，带 `--user` 只删除指定用户记录。 |
| `clear` | `[--all\|--user <用户ID>]`、`[--confirm <确认码>]` | 默认清空当前用户；`--all` 清空全部用户；`--user` 清空指定用户；两者互斥。首次执行不带确认码只显示影响数量和确认命令。 |
| `rebuild` | `[--to <知识库名>]`、`[--confirm <确认码>]` | 不带 `--to` 原地重建当前知识库；带 `--to` 迁移到目标知识库；首次执行不带确认码只预览，不会执行。 |
| `rebuild --clear-cache` | `[--confirm <确认码>]` | 清理中断重建的 KV 缓存；不能与 `--to` 同时使用；重建进行中禁止清理。 |
| `review` | `[list\|approve <ID>\|reject <ID>\|clear]` | 不带参数等同 `list`；`approve` 执行对应操作后标记已批准；`reject` 标记废案；`clear` 清空待审队列；管理员专用。 |

#### 命令示例
```text
/memory list
/memory list 2
/memory list --all 3
/memory search "咖啡 偏好"
/memory stats --all
/memory test organizer
/memory forget facts://abcd1234
/memory forget facts://abcd1234 --user user_123
/memory clear --user user_123
/memory rebuild --to 新知识库
/memory review
/memory review approve 12
/memory review reject 12
/memory review clear
```

#### 确认码
`clear`、`rebuild` 和 `rebuild --clear-cache` 的确认码不是固定文本。先执行不带 `--confirm` 的命令，插件会根据当前操作目标生成确认命令；只有复制该次预览中的确认码，且预览状态未变化时才会执行。


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
- `memory_search_admin(query, domain, scope, top_k)` — 搜索所有活跃记忆并返回精确管理目标（需开启 `enable_admin_global_memory_tool`，仅管理员；`domain`、`scope` 可选，`top_k` 为 1-20）
- `memory_remove_admin(uri, scope, owner_id, confirm)` — 预览后按 URI、scope 和归属确认删除（同一开关、仅管理员；global 不传 `owner_id`，personal 使用 `owner_user_id`，group 使用 `owner_session_id`，conversation 使用完整 `umo`）
- `memory_forget(uri)` — 删除当前用户有权限的记忆

`memory_remove_admin` 首次调用只返回 URI、scope、归属、计数、正文长度、SHA-256 短指纹、80 字符预览和确认码，不会删除。确认调用会再次核对当前记录指纹；管理员可管理 global、personal、group 和 conversation 记忆，但必须指定精确 scope，非 global 还必须指定对应归属。
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
### DEBUG 诊断日志

将 AstrBot 日志级别设为 DEBUG 后，可按 `trace_id` 串联一次召回：过滤器、dense/sparse/disclosure 通道、权限过滤、去重、重排、反馈更新、关联补充及最终结果。自动注入额外记录实际注入目标、格式化长度和每条记忆摘要。

每条召回或注入记忆摘要包含 `uri`、scope、是否关联、`content_len`、内容 SHA-256 短指纹及压缩后的前 80 个字符 `preview`；注入上下文还会显示 AstrBot 时区下的 `created`/`updated`/`curated` 状态，以及共享记忆的 `current user`、`associated users` 或 `current group` 身份标签。写入、替换和删除记录其存储阶段和统计；自动提取记录快照、解析、LLM 耗时与写入统计。日志不记录完整记忆正文、原始对话或 LLM prompt。
- 请确保先创建知识库并配置嵌入模型
- 记忆数据存储在知识库中，删除知识库将丢失所有记忆
- **请勿将记忆知识库挂载到 AstrBot 全局知识库配置中**。本插件通过 `user_id` 实现用户级记忆隔离，而 AstrBot 原生知识库检索不支持用户隔离，挂载后会导致所有用户共享彼此的记忆。仅个人独占使用时可忽略此限制
- **后台定时任务为插件自管的 asyncio 循环，不写入 AstrBot 的 cron_jobs 数据库表**，插件重启后自动恢复。若之前使用过旧版本（v0.4.1 及以下）将任务注册到了 cron_jobs 表，重启后会残留无效条目，需手动清理：
  ```sql
  DELETE FROM cron_jobs WHERE name IN ('memory_purge', 'memory_maintenance_cycle');
  ```


### 启用方法

1. 在「使用电脑能力」中将运行环境设置为 `local` 或 `sandbox`
2. 在本插件配置中开启 **安装记忆 Skill**
3. 重启或重载插件，Skill 将自动安装到 AstrBot 的 skills 目录并激活


<div align="center">

**如果这个插件对你有帮助，请给个 Star 支持一下！**

</div>
