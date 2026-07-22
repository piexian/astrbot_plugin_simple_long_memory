# 自审结果

## Critical

无。

## Warning

- 当前运行配置仍显示 `qwen3-embedding-8b`、`auto`、3072；该组合在自定义 API Base 下不会发送 `dimensions`，实际得到 768，与现有 3072 维索引不匹配。
- 仅保存或热重载 provider 不会刷新插件持有的旧向量库 provider 引用，需要完整重启 AstrBot 实例。

## Info

- 本任务只执行日志、源码和最小 API 探测，没有修改插件代码或 Windows AstrBot 配置。
- 未运行代码测试；没有产品代码变更，验证采用 Windows 原生 venv 的串行及 5 路并发 embedding 请求。
