# 诊断目标

- 确认 AstrBot v4.26.7 中简单长期记忆插件持续报告 `Connection error.` 的直接原因。
- 区分模型可变维度能力、New API 路由约束、Windows 到 WSL 的连通性和 provider 生命周期问题。
- 不修改 Windows AstrBot 配置及插件运行代码。

# 结论

- Windows 可访问 `127.0.0.1:3000` 与 `localhost:3000`，New API 服务可达。
- 当前路由下 `qwen3-embedding-8b` 默认返回 768，显式 3072 被上游 schema 拒绝；4096 和 1024 可用。
- `gitee/qwen3-embedding-8b` 显式请求 3072 成功，返回向量长度为 3072。
- Windows AstrBot venv 中 5 路并发探测全部成功，并发不是连接错误原因。
- embedding provider 热重载后插件保留了旧 `KBHelper/vec_db` 引用；旧 provider 客户端已被终止，新控制台探测实例仍可成功，形成持续 `Connection error.`。
- 现有 FAISS 索引维度为 3072。保留索引应使用 `gitee/qwen3-embedding-8b`、`always`、3072，并在保存后完整重启 AstrBot 实例。
