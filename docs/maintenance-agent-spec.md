# 后台记忆整理系统 — 设计规格

> 版本: draft v0.2 | 状态: 实施中 | 分支: feat/maintenance-agent

## 概述

为插件增加后台记忆整理能力。通过定时运行的 LLM Agent 团队，自动完成记忆去重合并、关联发现、矛盾检测和质量精炼，保持记忆池长期健康。

### 设计原则

- **保守操作**：宁可不动也不误删。错误归档一条正确记忆是最严重的事故。
- **Agent 只建议，Host 执行**：Agent 输出结构化 JSON manifest，插件代码解析后才执行 DB 写入。LLM 幻觉不会直接破坏数据。
- **结构化输出**：所有 Agent 操作以 JSON manifest 形式返回，不依赖自由文本解析。
- **可配置提示词**：内置默认模板 + 变量注入，用户可完全覆盖或追加指引。

---

## 角色定义

### 全局设置中的程序

| 名称 | 类型 | 说明 |
|------|------|------|
| 自动清理 | 纯逻辑定时任务 | 物理删除 deprecated 超过 N 天的记忆（FAISS + SQLite + KB 文档记录） |

### Agent 团队（template_list 条目）

| 角色 | key | 职责 | 需要 LLM |
|------|-----|------|----------|
| 📋 整理师 | `organizer` | 去重合并相似记忆、精炼措辞、归档无价值条目 | ✅ |
| 🔍 分析师 | `analyst` | 拉取对话历史，发现记忆间隐含关联，检测矛盾 | ✅ |
| ✅ 审核员 | `reviewer` | 复核整理师和分析师的操作建议，approve/reject | ✅ |

---

## 数据模型

### 关联表（新增）

独立 SQLite 表，不塞 metadata：

```sql
CREATE TABLE IF NOT EXISTS memory_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_uri TEXT NOT NULL,
    target_uri TEXT NOT NULL,
    relation_type TEXT NOT NULL,  -- 'related' | 'contradicts' | 'supersedes' | 'supports' | 'context'
    reason TEXT DEFAULT '',
    confidence REAL DEFAULT 1.0,
    created_by TEXT DEFAULT 'analyst',  -- 哪个角色创建的
    created_at TEXT NOT NULL,
    UNIQUE(source_uri, target_uri, relation_type)
);

CREATE INDEX idx_links_source ON memory_links(source_uri);
CREATE INDEX idx_links_target ON memory_links(target_uri);
```

### 召回时关联记忆的使用

当一条记忆被召回时，查询其关联记忆中 confidence >= 阈值的条目，将关联记忆作为补充上下文一并注入（标记为 `[关联记忆]`）。

---

## 配置 Schema

### 全局设置（扁平 + collapsed）

```json
{
  "config_mode": {
    "type": "string", "default": "basic",
    "options": ["basic", "advanced"],
    "description": "配置模式",
    "hint": "基础模式只显示核心开关，高级模式显示全部配置"
  },
  "maintenance_enabled": {
    "type": "bool", "default": false,
    "description": "后台整理总开关"
  },
  "maintenance_model_id": {
    "type": "string", "_special": "select_provider", "default": "",
    "description": "整理模型",
    "hint": "用于后台整理 Agent 的 LLM 模型，建议使用廉价模型"
  },
  "maintenance_window": {
    "type": "string", "default": "02:00-06:00",
    "description": "整理时间窗口",
    "hint": "单时间点如 03:00，或范围如 02:00-06:00（范围内空闲时执行）",
    "condition": {"config_mode": "advanced"}
  },
  "auto_purge_enabled": {
    "type": "bool", "default": true,
    "description": "自动清理废弃记忆",
    "hint": "定期物理删除已标记废弃超期的记忆"
  },
  "auto_purge_after_days": {
    "type": "int", "default": 7,
    "description": "清理超期天数",
    "hint": "deprecated 标记超过此天数后物理删除",
    "condition": {"config_mode": "advanced"}
  },
  "context_max_rounds": {
    "type": "int", "default": 50,
    "description": "最大拉取对话轮数",
    "condition": {"config_mode": "advanced"}
  },
  "context_max_chars": {
    "type": "int", "default": 30000,
    "description": "最大拉取字符数",
    "condition": {"config_mode": "advanced"}
  },
  "context_max_age_days": {
    "type": "int", "default": 7,
    "description": "对话最大回溯天数",
    "condition": {"config_mode": "advanced"}
  },
  "persona_mode": {
    "type": "string", "default": "auto",
    "options": ["auto", "manual", "off"],
    "description": "人格加载模式",
    "hint": "auto=自动从主人格提取摘要, manual=使用下方自定义, off=不加载",
    "condition": {"config_mode": "advanced"}
  },
  "persona_summary": {
    "type": "text", "default": "",
    "description": "自定义人格摘要",
    "hint": "manual 模式下填写，帮助整理 Agent 理解对话语境",
    "condition": {"config_mode": "advanced"},
    "collapsed": true
  }
}
```

### 任务编排（template_list）

```json
{
  "maintenance_tasks": {
    "type": "template_list",
    "description": "整理团队编排",
    "hint": "添加后台整理角色，配置各自的执行时间和行为",
    "default": [],
    "condition": {"config_mode": "advanced"},
    "templates": {
      "organizer": {
        "description": "📋 整理师",
        "hint": "去重合并相似记忆、精炼措辞、归档无价值条目",
        "items": {
          "enabled": {"type": "bool", "default": true, "description": "启用"},
          "schedule": {"type": "string", "default": "03:00", "description": "执行时间", "hint": "如 03:00 或 02:00-06:00"},
          "batch_size": {"type": "int", "default": 30, "description": "每次处理记忆数"},
          "prompt_override": {"type": "text", "default": "", "description": "自定义提示词（完全覆盖）", "hint": "留空使用内置模板。变量: {persona_summary} {memories} {memory_count} {current_time}"},
          "prompt_extra": {"type": "text", "default": "", "description": "追加指引", "hint": "拼接到内置模板末尾的额外要求"}
        }
      },
      "analyst": {
        "description": "🔍 分析师",
        "hint": "拉取对话历史，发现记忆间隐含关联，检测矛盾",
        "items": {
          "enabled": {"type": "bool", "default": true, "description": "启用"},
          "schedule": {"type": "string", "default": "02:00", "description": "执行时间"},
          "max_new_links": {"type": "int", "default": 20, "description": "每次最多新建关联数"},
          "detect_contradiction": {"type": "bool", "default": true, "description": "同时检测矛盾"},
          "prompt_override": {"type": "text", "default": "", "description": "自定义提示词（完全覆盖）", "hint": "变量: {persona_summary} {memories} {conversation_history} {memory_count} {current_time}"},
          "prompt_extra": {"type": "text", "default": "", "description": "追加指引"}
        }
      },
      "reviewer": {
        "description": "✅ 审核员",
        "hint": "复核整理师和分析师的操作建议，防止误删误改",
        "items": {
          "enabled": {"type": "bool", "default": true, "description": "启用"},
          "schedule": {"type": "string", "default": "after", "description": "执行时机", "hint": "after=在其他角色完成后自动运行，也可指定时间如 04:00"},
          "model_id": {"type": "string", "_special": "select_provider", "default": "", "description": "审核模型", "hint": "留空使用全局整理模型"},
          "prompt_override": {"type": "text", "default": "", "description": "自定义提示词（完全覆盖）", "hint": "变量: {proposed_changes} {original_data} {persona_summary}"},
          "prompt_extra": {"type": "text", "default": "", "description": "追加指引"}
        }
      }
    }
  }
}
```

---

## 提示词系统

### 变量列表

| 变量 | 适用角色 | 内容 |
|------|----------|------|
| `{persona_summary}` | 全部 | bot 人格摘要 |
| `{memories}` | 整理师/分析师 | 格式化的记忆列表（URI + 内容 + 域 + 重要度 + 召回次数） |
| `{memories_json}` | 整理师/分析师 | 同上 JSON 格式 |
| `{conversation_history}` | 分析师 | 拉取的对话历史（受 max_rounds/max_chars/max_age 限制） |
| `{memory_count}` | 全部 | 当前活跃记忆总数 |
| `{memory_stats}` | 全部 | 按域/作用域的统计分布 |
| `{current_time}` | 全部 | 任务执行时间 |
| `{proposed_changes}` | 审核员 | 其他角色输出的操作 manifest |
| `{original_data}` | 审核员 | 分析时的原始输入数据 |
| `{admin_guides}` | 全部 | 管理员历史指引（从争议审查中累积的规矩） |

### 提示词选配逻辑

```python
def build_prompt(task_config, default_template, variables):
    if task_config.get("prompt_override"):
        template = task_config["prompt_override"]
    else:
        template = default_template
        if task_config.get("prompt_extra"):
            template += "\n\n## 额外指引\n" + task_config["prompt_extra"]
    return template.format(**variables)
```

### 内置默认模板（示例：分析师）

```
你是"{persona_summary}"这个 bot 的记忆分析师。

## 身份与原则
- 你是后台维护人员，不是聊天机器人本身
- 保守原则：不确定时保持现状，不归档不删除
- 你只输出结构化建议，不直接修改任何数据

## 当前状态
- 活跃记忆: {memory_count} 条
- 统计: {memory_stats}
- 执行时间: {current_time}

## 记忆池
{memories}

## 最近对话
{conversation_history}

## 任务
1. 找出记忆之间应该建立的关联（因果、场景、主题、时间）
2. 发现互相矛盾的记忆对（新的 supersede 旧的）
3. 识别对话中应该关联到已有记忆的新信息

## 输出格式（严格 JSON）
{
  "new_links": [
    {"source": "uri_a", "target": "uri_b", "relation": "related|supports|context", "reason": "..."}
  ],
  "contradictions": [
    {"old_uri": "...", "new_uri": "...", "reason": "..."}
  ],
  "notes": "可选的分析备注"
}
```

---

## 执行流程

### 调度

1. 插件启动时注册 asyncio 定时器，每 60 秒检查一次
2. 检查当前时间是否在各任务的 schedule 窗口内
3. 窗口内且距上次执行超过最小间隔 → 触发
4. `schedule: "after"` 的审核员在其他角色产出 manifest 后自动触发

### 执行管线

```
定时器触发
    │
    ▼
拉取输入数据（记忆列表 / 对话历史 / 人格摘要）
    │
    ▼
组装 prompt（内置模板 + 变量注入 / 用户覆盖）
    │
    ▼
调用 LLM（maintenance_model_id 或任务级 model_id）
    │
    ▼
解析 JSON manifest
    │
    ├── 有审核员 → 暂存 manifest → 审核员 approve 后执行
    │
    └── 无审核员 → 直接执行
    │
    ▼
Host 执行 DB 操作（merge / archive / insert link / mark superseded）
    │
    ▼
记录执行日志到 KV（时间、结果、耗时、操作数）
```

### 互审模式（会话制）

每次整理周期分配一个 `session_id`，所有角色在同一会话上下文中工作，
审核员的驳回理由能回传给原角色修正，而不是无上下文地重跑。

```
─── 整理周期 session_id = "maint-20260730-0200" ───

1. 整理师/分析师 → 输出 manifest（暂存，关联 session_id）
                       │
                       ▼
2. 审核员 ← 输入: manifest + 原始数据 + session 上下文
       │
       ▼ 输出: [{"id": N, "verdict": "approve|reject", "reason": "..."}]
       │
       ├── 全部 approve → Host 执行
       │
       └── 有 reject → 驳回理由回传原角色
                       │
                       ▼
3. 原角色 ← 输入: 自己的原始 manifest + 审核员驳回理由（同一 session 上下文）
       │
       ▼ 输出: 修正后的 manifest
       │
       ▼
4. 审核员 ← 二次审核修正项
       │
       ▼
   全部 approve → Host 执行
   仍有 reject → 丢弃该项，记录日志（最多 2 轮修正，防止死循环）
```

**session_id 的作用：**
- 审核员驳回后，原角色能看到自己当初的提案和驳回理由，有针对性地修正
- 同一周期内多个角色的操作共享上下文，审核员能看到全貌
- 执行日志按 session_id 归档，可追溯一次整理的完整决策链
- KV 中按 session_id 存储中间状态，进程重启后可恢复未完成的审核流程

### 人工升级（争议项）

当 Agent 团队内部无法达成一致时，升级给管理员人工审查：

**触发条件（任一）：**
- 同一操作被驳回 2 轮后仍有争议
- 审核员标记 `confidence < 0.5` 或 `controversial: true`

**硬性规则（不可配置）：**
> ❗ 所有涉及 global 作用域记忆的写入/修改/归档/删除，**必须经管理员人工确认**，
> 无论审核员是否 approve。Agent 团队对 global 记忆只有“建议权”，没有“执行权”。
> 此规则不可通过配置关闭。

**流程：**
```
审核员 reject + 标记 controversial
    │
    ▼
Host 将争议项写入待审队列（KV: maintenance_pending_review）
    │
    ▼
管理员通过命令查看: /memory review
    │
    ▼ 显示争议列表:
    │   #1 [整理师] 合并 facts://a1b2 + facts://c3d4
    │      整理师理由: "两条都在说用户喜欢咖啡"
    │      审核员理由: "一条是说喜欢拿铁，一条是说喜欢美式，不是同一件事"
    │
    ▼
管理员指令:
    /memory review approve 1       → 批准执行
    /memory review reject 1        → 废案丢弃
    /memory review guide 1 <指引>  → 立即重跑 Agent，结果返回管理员确认
```

**guide 流程（即时执行）：**
```
管理员: /memory review guide 1 这两条不该合并，拿铁和美式是不同偏好
    │
    ▼
Host 立即调用 Agent，输入: 原始数据 + 争议上下文 + 管理员指引
    │
    ▼
Agent 返回修正后的操作结果（展示给管理员）
    │
    ▼
管理员确认:
    /memory review confirm 1   → 写入 DB
    /memory review reject 1    → 废案
    /memory review guide 1 <新指引> → 再跑一次（最多 3 轮，超过强制废案）
```

**指引累积：**
无论最终 confirm 还是废案，管理员的 guide 内容都会存入 KV（`maintenance_admin_guides`），
作为 `{admin_guides}` 变量注入后续整理周期的 prompt——
相当于给 Agent 立长期规矩，同类情况不会再犯。

**管理员命令：**
```
/memory review              - 查看待审争议列表
/memory review approve <id> - 批准执行
/memory review reject <id>  - 废案丢弃
/memory review guide <id> <指引> - 立即重跑 Agent，结果返回确认
/memory review confirm <id> - 确认写入（guide 后使用）
/memory review clear        - 清空待审队列
```

### 待审通知（主动推送）

待审项不应静默堆积。配置定时推送，主动提醒管理员处理：

```json
"review_notify_enabled": {"type": "bool", "default": true, "description": "待审通知开关"},
"review_notify_umo": {
  "type": "string", "default": "",
  "description": "通知目标 UMO（可选）",
  "hint": "填写群聊或私聊的 unified_msg_origin。留空则自动发给最近一次执行管理命令的管理员"
},
"review_notify_schedule": {
  "type": "string", "default": "daily",
  "options": ["daily", "weekly", "monthly"],
  "description": "通知频率",
  "hint": "daily=每天固定时间, weekly=每周固定日, monthly=每月固定日"
},
"review_notify_time": {
  "type": "string", "default": "10:00",
  "description": "通知时间",
  "hint": "daily 为每天时刻，weekly 为周几+时刻如 mon:10:00，monthly 为几号+时刻如 1:10:00"
}
```

管理员身份直接使用 AstrBot 核心的 `admins_id` 配置（`event.is_admin()`），插件不重复配置。
主动推送通过 `context.send_message(umo, message_chain)` 发送。
管理员在目标会话（私聊或群聊）中与机器人交互一次，发送 `/sid` 指令获取当前会话的 UMO，填入配置即可。


推送内容示例：
```
📝 记忆整理待审通知
当前有 3 条争议操作等待审查：

#1 [整理师] 合并 facts://a1b2 + facts://c3d4
     审核员: "一条说拿铁一条说美式，不是同一件事"
#2 [分析师] 关联 preferences://x1 → facts://y2
     审核员: "关联牵强，置信度 0.3"
#3 [global] 修改 facts://global_rule1
     硬性规则: global 记忆必须人工确认

回复 /memory review 查看详情并处理
```

无待审项时不推送（不骚扰）。
## 对话历史访问

AstrBot 本地存储对话记录。访问方式待确认：
- 方案 A：通过 `context` API（如果有暴露接口）
- 方案 B：直接读取 AstrBot 的 SQLite 数据库（`data/` 目录）

拉取限制（三个维度取最严格）：
- `context_max_rounds`: 最大轮数
- `context_max_chars`: 最大总字符数
- `context_max_age_days`: 只拉最近 N 天

---

## 物理清理（自动清理程序）

非 Agent，纯逻辑定时任务：

```python
async def auto_purge(self):
    """物理删除 deprecated 超过 N 天的记忆"""
    cutoff = now - timedelta(days=purge_after_days)
    # 1. 查询 deprecated=True 且 created_at < cutoff 的记忆
    # 2. 从 FAISS 删除向量
    # 3. 从 SQLite documents 表删除
    # 4. 从 KB 文档记录删除
    # 5. 清理 memory_links 中涉及的关联
    # 6. 同步 KB 统计
```

---

## 文件结构规划

```
astrbot_plugin_simple_long_memory/
├── main.py                    # 现有：入口/命令/工具/钩子
├── memory_manager.py          # 现有：KB 操作/召回/重建
├── memory_protocol.py         # 现有：URI/UMO/格式化
├── prompts.py                 # 现有：提取/检索 prompt + 校验常量
├── maintenance/               # 新增：后台整理模块
│   ├── __init__.py
│   ├── scheduler.py           # 调度器：时间窗口检查、任务触发
│   ├── runner.py              # 执行管线：拉取数据→组装prompt→调LLM→解析manifest→执行
│   ├── agents/
│   │   ├── organizer.py       # 整理师：去重合并、质量精炼
│   │   ├── analyst.py         # 分析师：关联发现、矛盾检测
│   │   └── reviewer.py        # 审核员：复核操作建议
│   ├── prompts.py             # 内置默认模板 + 变量注入
│   ├── links.py               # 关联表 CRUD
│   └── purge.py               # 物理清理程序
└── docs/
    └── maintenance-agent-spec.md  # 本文档
```

---

## 实施阶段

| Phase | 内容 | 依赖 |
|-------|------|------|
| 1 | deprecated_at 迁移 + 关联表 + purge + prompt 模板修复 + cron 调度 | 无 |
| 2 | 执行管线框架 + manifest 校验 + 操作状态机 | Phase 1 |
| 3 | 整理师（去重合并 + 质量精炼），替代旧 consolidation | Phase 2 |
| 4 | 分析师（关联发现 + 矛盾检测）+ conversation_manager API | Phase 2 |
| 5 | 审核员 + 互审模式 + 人工升级 | Phase 3, 4 |
| 6 | 召回时关联记忆注入（可见性/预算/去重） | Phase 1 |
| 7 | 配置 schema + i18n + README | 全部 |

---

## v0.2 审查修正（基于 .ccg review）

### 阻断项修复

**1. deprecated_at 字段**
- `expire_stale_memories` 和 `mark_consolidated` 写入时补上 `deprecated_at: ISO timestamp`
- 迁移补丁：已有 `deprecated=1` 但无 `deprecated_at` 的记录，用 `created_at` 回填
- purge 按 `deprecated_at + N days` 判断，不用 `created_at`

**2. 租户模型（无 event 场景）**
- 后台任务不依赖 AstrMessageEvent，改为显式传入 owner_user_id / kb_id / scope
- 整理师/分析师按 owner 逐个处理，不跨用户
- 关联发现只在同一 owner 的记忆之间建立，不跨用户串联
- 对话历史拉取按 UMO 限定，不跨会话读取

**3. prompt 模板引擎**
- 改用 `string.Template`（`$var` 语法），不用 `str.format()`
- JSON 花括号不再被误解析
- 用户自定义 prompt 中用 `$memories` 而非 `{memories}`

### High 项修复

**安全 executor**
- manifest 用 Pydantic/dataclass 强类型校验，拒绝未知字段
- 操作 allowlist：只允许 merge / archive / update / new_link / supersede
- destructive 操作（archive/delete/merge）无审核员时也需人工确认
- additive 操作（new_link）可直接执行
- 每个操作带幂等 operation_id，崩溃重放不重复执行

**旧 consolidation 迁移**
- 新系统上线后禁用 `_maybe_consolidate_memories`
- 旧巩固功能迁移为整理师的一个子任务（低频老记忆压缩）
- 配置 `maintenance_enabled=true` 时自动关闭旧巩固，避免并发竞态

**调度生命周期**
- 使用 `context.cron_manager.add_basic_job()` 而非自建 60s 轮询
- initialize 注册 / terminate 注销，不泄漏任务
- single-flight：同一任务不并发执行

**link 级联清理**
- 所有物理删除路径（forget/clear/rebuild/purge）统一走 link-aware executor
- 删除记忆时同步删除 memory_links 中 source_uri 或 target_uri 涉及的边

**关联召回规则**
- 只召回未 deprecated 的关联记忆
- 单跳（不递归），最多 3 条关联记忆
- 总注入 token 预算上限（关联记忆 + 主记忆 合计不超过 max_length）
- contradicts/supersedes 关系不注入召回，仅用于后台分析

**通知隐私**
- 群聊 UMO 只发通用提示（“有 N 条待审”），不发具体内容
- 详细争议内容只发到私聊 UMO

**对话历史**
- 使用 `context.conversation_manager` API，不直接读 SQLite
- 按 UMO 限定拉取范围，明确截断顺序（时间 → 轮数 → 字符数）
