# 自审结果

## 结论

- 实际异常是 FAISS 查询的无消息 `AssertionError`，触发点为查询向量维度与索引维度不一致。
- 记忆知识库索引维度为 3072，日志中的新向量实际维度为 1024。
- AstrBot v4.26.7 新增 `embedding_dimensions_mode`；默认 `auto` 对自定义 OpenAI 兼容地址不发送 `dimensions`，与当前 `127.0.0.1:3000` provider 配置形成不一致。
- `ProviderRequest.extra_user_content_parts` 在 v4.26.7 仍兼容，不是故障点。

## 建议

- 将 `openai_embedding` 的 `embedding_dimensions_mode` 设为 `always` 并重启，以继续使用现有 3072 维、106 条记录的索引。
- 若服务不支持 3072 维，则改用 1024 维配置后重建知识库；不能只改配置而保留旧索引。
- 后续可改进插件异常日志，至少输出异常类型并附 traceback，避免空消息掩盖根因。

## 验证

- 已读取 Windows `astrbot.log` 中同路径的完整 traceback。
- 已用 FAISS 官方读取接口确认现有索引 `d=3072`、`ntotal=106`。
- 已通过 `gh` 核对 AstrBot v4.26.7 官方提交 `635124be32` 的 embedding 维度发送策略变更。
