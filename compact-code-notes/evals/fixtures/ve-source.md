# 提供的源码快照

从 VE CUDA Graph 学习笔记中的 manager 代码提取，作为本次练习的源码依据，不代表远端当前版本。

```python
class VECudaGraphManager:
    def __init__(
        self,
        tower: torch.nn.Module,
        max_tokens: int,
        max_segments: int,
        max_inflight: int,
        zero_grad_buffer: Callable[[], None],
        no_sync: Callable,
        mode: str,
        capture_stream: torch.cuda.Stream | None = None,
        pool=None,
        defer_capture: bool = False,
    ) -> None:
        if mode not in {"forward", "forward_backward"}:
            raise ValueError(f"unsupported VE CUDA Graph mode: {mode}")
        if max_inflight != 1:
            raise ValueError("VE CUDA Graph requires max_inflight=1")
        self.tower = tower
        self.max_tokens = int(max_tokens)
        self.max_segments = int(max_segments)
        self.max_inflight = int(max_inflight)
        self.zero_grad_buffer = zero_grad_buffer
        self.no_sync = no_sync
        self.mode = mode
        self.capture_stream = capture_stream
        self.pool = pool
        self.defer_capture = defer_capture
        self.slots: list[_GraphSlot] = []
        self.graphs_ready = False
        self.grad_init_graph = None
        self._capture_template = None
```

```python
def forward(self, images: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
    if self.defer_capture and not self.graphs_ready:
        if self._capture_template is None:
            self._capture_template = (images.detach().clone(), grid_thw.detach().clone())
        return self.tower(images, grid_thw)["last_hidden_state"]
    if not self.slots:
        self._capture(images, grid_thw)
    if self.mode == "forward":
        return self.slots[0].forward_only(images, grid_thw)
    for slot in self.slots:
        if not slot.pending:
            return slot.forward_backward(images, grid_thw)
    raise RuntimeError(f"VE CUDA Graph has more than {self.max_inflight} outstanding forwards")
```

```python
def _capture(self, images: torch.Tensor, grid_thw: torch.Tensor) -> None:
    self.prepare_capture(images, grid_thw)
    self.capture_forward()
    if self.mode == "forward_backward":
        self.capture_backward()
    self.finish_capture()
```

本练习只提供上述实现。外层 trainer 的调用时机没有提供，不需要去查其他仓库。
