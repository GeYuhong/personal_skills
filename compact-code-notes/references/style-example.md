# 风格样例

这份样例提炼自用户 VE CUDA Graph `note.md` 已完成的第 1–5 节。学习它的组织和解释密度，不照抄项目结论。以下源码取自那份笔记的片段，仅用于展示写法；为实际项目写作时需要核对对应版本。

下面两个小节假定前文已经讲过 graph slot 和捕获过程：第一个展示如何简洁串起已知组件，第二个展示遇到不直观机制时如何保留必要深度。

---

## 5.3 `_capture()`：串起建图阶段

`_capture()` 接收用于建图的输入，按顺序完成准备、前向捕获、按需反向捕获和收尾。它由 manager 的直接捕获路径调用，最终让 manager 持有可用的 graph slot；函数本身不返回本轮训练特征。

```python
def _capture(self, images: torch.Tensor, grid_thw: torch.Tensor) -> None:
    self.prepare_capture(images, grid_thw)
    self.capture_forward()
    if self.mode == "forward_backward":
        self.capture_backward()
    self.finish_capture()
```

`prepare_capture()` 创建固定容量 wrapper 和 graph slot，准备缓冲区并执行 warmup。后续的 `capture_forward()` 和 `capture_backward()` 把捕获交给 slot，具体过程沿用前文的实现；仅在 `forward_backward` 模式下捕获反向。最后，manager 的 `finish_capture()` 调用各 slot 的收尾方法，再将 `graphs_ready` 设为 True。

直接捕获路径中的 `manager.forward()` 在 `_capture()` 返回后还会继续执行本次前向 replay。**建图完成与本轮正式前向是先后两个阶段**，捕获后的梯度清理不会取消随后 replay 产生的训练梯度。

## 3.4 `finish_capture()`：整理输出和梯度状态

graph slot 在捕获完成后需要整理流依赖、输出引用和梯度状态，再恢复准备阶段影响的统计量，为正式 replay 做准备。

```python
def finish_capture(self) -> None:
    torch.cuda.current_stream(self.images.device).wait_stream(self.stream)
    if self.capture_backward:
        self.output = self.output.detach()
    for parameter in self.parameters:
        parameter.grad = None
    self.zero_grad_buffer()
    for counter, value in zip(self.counters, self.saved_counters, strict=True):
        counter.copy_(value)
```

`wait_stream()` 在当前流和捕获流之间建立依赖，使当前流后续提交的工作等待捕获流此前已提交的工作完成。它不要求 CPU 在这里阻塞到 GPU 完成。捕获了反向图时，`detach()` 随后切断输出与捕获阶段 autograd 图的关系，但保留对同一块存储的引用。前向 replay 仍向这块存储写入结果，反向则由前文的 autograd bridge 调用已经捕获好的反向图。

清理 `.grad` 和清空梯度缓冲区处理的是两个层面：`parameter.grad = None` 解除属性引用，不会清零张量内容；捕获阶段保存的 `parameter_grads` 还持有那些梯度张量。`zero_grad_buffer()` 调用框架的清理逻辑，实际清理的缓冲区由回调实现决定，不能仅凭函数名推断全部行为。

最后，`copy_()` 把 warmup 前保存的专家计数恢复到原计数器中，避免准备过程污染训练统计。这里恢复的是备份值，并不一定是零；原地写入也保留了计数器的对象和存储。

---

两个小节都先说明作用，再给源码，随后解释关键关系。`_capture()` 只串起已讲组件；`finish_capture()` 则多花篇幅区分流依赖、autograd 关系、存储和数值。篇幅由理解难点决定，不给每个函数套同一组小标题。
