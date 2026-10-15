# 5. manager 的生命周期

## 5.1 初始化

manager 保存模型、容量和回调，初始时 `slots=[]`、`graphs_ready=False`、`_capture_template=None`。当前实现只允许 `max_inflight=1`。前文已讲过 slot 的 `prepare()`：复制本轮输入并更新定长布局；`pending` 表示前向完成但对应反向尚未结束。
