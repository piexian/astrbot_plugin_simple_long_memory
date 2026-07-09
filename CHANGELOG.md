# Changelog

## v0.3.4 (2026-07-10)

### 修复
- `/memory list` 仅在确有下一页时显示翻页提示；越界页返回“当前分页不存在”，非法页码返回明确错误。
- `/memory search` 将全部召回结果按单页展示，不再出现页数、条目数和下一页提示互相矛盾的问题。
- `/memory stats` 统一个人与全局统计口径：总数、永久和普通记忆只统计活跃记录，压缩历史单独统计并明确标注。
- `/memory test` 在召回异常时仍能完成清理并返回诊断报告，不再因未初始化变量二次崩溃。
- `/memory rebuild` 的后续命令提示使用实际指令前缀；空知识库迁移会正常提交并切换目标知识库。

### 变更
- 新增记忆指令和管理器回归测试，覆盖分页、搜索、统计、异常诊断、自定义前缀及空库迁移。

## v0.3.3 (2026-06-28)

### 新增
- **召回反馈**：记忆召回时 `recall_count` 递增，记录每条记忆的命中频次，用于后续加权与巩固判断。
- **召回加权重排**：召回结果按 `importance`（重要性）/ `recall_count`（频次）/ 时间衰减（时效）三路信号加权二次排序；时效采用可配置半衰期。
- **稠密+稀疏 RRF 融合**：召回在稠密向量之外追加 FTS5 关键词稀疏检索，用 RRF 融合两路结果，提升关键词/专有名词召回率。FTS5 不可用时自动回退纯稠密。
- **TTL 过期**：对长期未命中的低频记忆标记过期，`permanent` / `global`（管理员全局）记忆排除，不会被误过期。
- **记忆巩固**：定期将低频老旧记忆交由 LLM 压缩为摘要印象，原文标记为已废弃；可配置最小年龄、召回频次上限、批量大小。
- 新增 9 项召回/巩固相关配置项，同步 `zh-CN` / `en-US` 双语 i18n 文案。

### 修复
- 巩固任务 `MemoryType.CONTEXT`（原不存在）改为 `NORMAL`，否则巩固必崩且原文已标记却无摘要导致数据丢失。
- 巩固产物 `domain` 从 `consolidated` 改为 `context`，进入白名单，不再回退到 `facts`。
- TTL 过期排除 `permanent` / `global` 记忆，管理员全局记忆不再被误过期。
- 后台 LLM 长任务使用 `asyncio.create_task` 时加入 `_background_tasks` 强引用集合，防止任务被 GC 提前回收。
- `_rerank` 对数值做安全转换并 `max(0, Δt)`，防止 `int()` 崩溃与 `exp` 溢出。
- 记录字段访问改用 `getattr` 兜底，嵌套 `json_set` 合并为单次调用。

### 变更
- 将 `expire_stale_memories` / `fetch_consolidation_candidates` / `mark_consolidated` 去掉 `_` 前缀升为公共 API；`fetch_consolidation_candidates` 改接 `event`，`owner` 由内部推导，解除 `main.py` 对 `MemoryManager` 受保护成员的耦合（回应 PR#4 Sourcery 反馈）。
- 更新 README 安装说明，移除插件待发布的提示。

## v0.3.2 (2026-05-14)

### 新增
- 新增 AstrBot 插件 i18n 配置，提供 `zh-CN` / `en-US` 双语 WebUI 元数据和配置项文案。
- 新增 `enable_admin_global_memory_tool` 配置和 `memory_store_global` LLM 工具。开启后管理员可指挥 AI 写入 `global` 作用域记忆，后续所有会话都会参与召回。

### 变更
- 记忆注入优先使用 AstrBot v4.24+ 的临时用户内容区，并标记为本轮临时内容，避免写入会话历史；旧版回退到最早的 `user` 上下文位置，不再拼接到当前 prompt 前。
- 记忆注入包裹说明强化为“长期记忆检索参考”，明确不是当前正在发生的事情，也不是用户指令；同时在顶部提醒 AI 必要时使用 `memory_recall(query)` 工具继续搜索更多记忆。

## v0.3.1 (2026-05-03)

### 新增
- **破坏性操作确认码**：`/memory clear`、`/memory rebuild`、`/memory rebuild --clear-cache` 需带 `--confirm <code>` 确认码，防止误操作。执行前展示影响范围和预计记录数

### 修复
- 新增快照上限（MAX_SESSION_SNAPSHOTS=20 / MAX_SNAPSHOT_CHARS=8000），防止长期会话内存膨胀
- `llm_generate` 参数 `provider_id` → `chat_provider_id`，兼容新版 API
- forget 不再区分"不存在"和"属于他人"，统一返回无权限，防止泄漏跨用户 URI 存在性
- UMO 解析改用 `split(":", 2)`，防止含多冒号的 UMO 被错误拆分
- `_is_visible_shared_personal` 过滤多 owner personal 记忆，确保仅 owner 可见
- 召回先 `fetch_k=top_k*3` 再过滤去重，提高召回覆盖
- `_flush_pending_writes` 失败记录进入重试队列，避免静默丢弃
- 重建迁移时校验目标知识库为空，防止误覆盖
- 崩溃恢复校验目标 KB 存在性及 ID 一致性，跳过已存在 URI
- SENSITIVE_PATTERNS 改为预编译 `re.compile`，sanitize 性能优化

## v0.3.0 (2026-05-03)

### 新增
- **群聊记忆作用域**：引入三层记忆作用域模型（`personal` / `group` / `conversation`），解决群聊场景下记忆归属问题
  - `personal`：用户个人记忆，按 `user_id` 隔离
  - `group`：群组共享记忆，按 `session_id` 隔离，群内所有成员可见
  - `conversation`：当前会话临时记忆，仅当前会话内召回
- **可见性模型**：`private`（仅记忆所有者可见）/ `group`（同群组内多人共享），多所有者记忆自动设为 `group` 可见
- **新元数据字段**：`memory_scope`、`owner_user_id`、`owner_user_ids`、`owner_session_id`、`visibility`、`speaker_id`、`subject`、`entities`、`topics`、`memory_content`
- **作用域感知召回**：群聊中自动合并 personal + group + conversation 三层记忆，私聊召回 personal，并可使用 conversation 保存当前私聊会话上下文
- **重建式升级**：运行时不再对旧 metadata 做兼容兜底；从旧版本升级后需执行 `/memory rebuild` 补齐 v0.3 作用域字段
- **记忆注入格式化**：按作用域分组展示，区分 personal/group/conversation 三类记忆
- **记忆提取增强**：LLM 提取 prompt 新增会话作用域信息、`scope`/`subject`/`subjects`/`entities`/`topics` 字段，支持群聊下多人记忆归属标注
- **Sender 追踪**：请求快照中记录 `sender_id`，对话历史按发送者标注
- **检索优化超时配置**：新增 `optimize_recall_query_timeout`，限制检索优化模型调用最长等待时间
- **列表扫描上限配置**：新增 `max_memory_list_scan`，限制群聊可见记忆列表的扫描量
- **删除扫描分页配置**：新增 `memory_delete_scan_page_size`，控制删除/清空记忆前同步收集 KB 文档记录的分页大小

### 变更
- `/memory list` 群聊中展示当前用户可见的所有记忆（含群组共享）
- 记忆内容格式化改用结构化 `memory:` 标签行，仅写入 domain、memory、recall_when、entities、topics 等语义检索字段
- 可见性值改为 `MemoryVisibility` 常量，减少裸字符串重复使用
- 重建/迁移确认码绑定源/目标 KB ID，缓存清理确认码绑定实际缓存指纹

## v0.2.2 (2026-04-03)

### 修复
- 迁移补丁覆盖范围扩大：除 `is_memory_record` 标记的记录外，也修补有 `uri` 但无标记的更早期旧记录
- `/memory forget` 支持普通用户删除自己的记忆，管理员可按 URI 直接删除所有用户的记忆
- LLM 工具 `memory_forget` 删除失败时区分"不存在"和"属于其他用户"两种情况
- 修复删除记忆时始终返回成功的问题，现返回实际删除数量
- 记忆存储时按 URI 去重：内容相同跳过写入，内容不同自动换新 URI

## v0.2.1 (2026-03-30)

### 修复
- 启动时自动修补旧记忆条目缺少 `chunk_index` 字段的问题：旧版插件写入向量数据库时未设置该字段，导致在 AstrBot 知识库界面执行检索时报 `KeyError: 'chunk_index'`。现通过 SQLite `json_set` 原地修补，无需重新嵌入向量。

## v0.2.0 (2026-03-27)

### 新增
- 记忆重建 (`/memory rebuild`)：原地重新嵌入所有记忆
- 记忆迁移 (`/memory rebuild --to <知识库名>`)：迁移记忆到目标知识库
- 缓存清理 (`/memory rebuild --clear-cache`)：手动清理重建缓存
- KV 持久化：重建中间数据通过 KV 数据库持久化，支持进程崩溃恢复
- 完整性校验：重建完成后自动对比预期与实际记忆数量
- 缓冲写入：重建期间新产生的记忆自动缓冲，完成后语义去重写入
- 分页提示自动适配 `--all` 模式和命令前缀

### 修复
- 命令前缀从 AstrBot 配置自动读取，不再硬编码 `/`
- 重建拉取兼容旧格式记忆（无 `is_memory_record` 字段的记录）
- 拉取 0 条但源 KB 有数据时自动中止，防止误删
- `try/finally` 兜底释放重建锁，防止异常路径永久卡锁
- 快照恢复保留未成功的记录，供下次继续恢复
- 迁移未提交时缓冲写入落回当前活跃知识库
- 失败路径正确展示异常终止信息，不再误报为完成
