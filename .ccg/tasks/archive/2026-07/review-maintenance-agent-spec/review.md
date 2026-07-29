# maintenance-agent-spec 审查

审查基线：AstrBot 4.26.8，commit `60c9e68d50dc9b9ed58503f21a0b77a8d0bd2159`。

## 结论

方向可行，但当前规格不可直接进入实施。必须先解决下列阻断项：

1. 自动清理按 `created_at` 判断，而现有废弃路径没有 `deprecated_at`。老记忆刚被废弃就会立即满足清理条件，绕过配置的 N 天宽限期。
2. 后台批次没有定义 tenant/owner/UMO/scope 边界。现有 MemoryManager 的隔离依赖 `AstrMessageEvent`，后台任务没有 event，可能跨用户读取对话、建立关联或执行写入。
3. `build_prompt()` 使用 `str.format()`，默认提示词和用户覆盖提示词中的 JSON 花括号会被当作格式字段，默认模板即可触发异常。

## High

- “无审核员则直接执行”允许单次 LLM 输出触发 merge/archive。Host 解析 JSON 不是安全边界；需要强类型 manifest、操作 allowlist、字段/URI/scope 校验、旧值摘要或版本前置条件、幂等 operation_id，以及 destructive 操作的强制复核/人工确认策略。
- 关联召回没有规定当前事件下的可见性复核、deprecated 过滤、单跳/条数/token 上限、去重和关系语义。`contradicts`/`supersedes` 不能与 `supports` 一样无条件注入。
- 新 organizer 与现有 TTL/LLM consolidation 流程重叠。必须明确替换、迁移或禁用旧流程，否则会并发处理同一记忆并产生重复摘要或竞态。
- 自建 60 秒轮询缺少时区、reload 去重、single-flight、超时、取消和进程崩溃语义。AstrBot 4.26.8 已提供 `context.cron_manager.add_basic_job()`；若仍自建循环，必须在 initialize/terminate 中显式管理任务。
- `memory_links` 只在 purge 中清理，现有 forget/clear/rebuild/migration 路径会留下悬空边。所有物理删除和合并路径都必须统一走 link-aware executor/repository。
- KV 会话状态与 DB 写入不是原子操作；崩溃后重放可能重复执行。需要持久化操作状态机、幂等键和 optimistic concurrency，而不是只靠 session_id。

## Medium

- 对话历史不应保留“直接读 AstrBot SQLite”的方案。4.26.8 已暴露 `context.conversation_manager`；规格应选定公开 API，并明确 UMO、conversation_id、persona 来源和时间/字符截断顺序。
- “最近执行管理命令的管理员”可能来自群聊；向该 UMO 推送详细争议内容会向群成员泄露记忆。自动回退应限于已验证的私聊管理员 UMO，群聊仅发通用提示或要求显式配置。
- `maintenance_window` 与任务级 `schedule` 的优先级、basic 模式启用但 `maintenance_tasks=[]` 的行为、总开关是否控制 purge、weekly/monthly 时间格式、系统时区/DST 均未定义。
- 缺少测试阶段。至少覆盖 purge 宽限期、tenant 隔离、prompt 花括号、manifest 拒绝未知字段、stale proposal、崩溃重放、reload 重复调度、link 级联清理、召回预算、global 人工确认和通知隐私。

## 建议的实施前置

先补一版 v0.2 规格，增加：work item 的租户模型、manifest JSON Schema/Pydantic 模型、安全 executor 与操作状态机、`deprecated_at` 迁移、现有 consolidation 的迁移策略、调度生命周期、关联召回预算/授权规则，以及逐阶段测试矩阵。完成这些后再拆实施阶段。
