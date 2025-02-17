import torch
import taichi as ti

ti.init(arch=ti.cpu)

# 1. 定义 PyTorch 参数
x_torch = torch.tensor([1.0], requires_grad=True)
y_torch = torch.tensor([2.0], requires_grad=True)
params = [x_torch, y_torch]
optimizer = torch.optim.SGD(params, lr=0.1)

# 2. 创建 Taichi 场并同步数据
x = ti.field(ti.f32, shape=(), needs_grad=True)
y = ti.field(ti.f32, shape=(), needs_grad=True)
loss = ti.field(ti.f32, shape=(), needs_grad=True)

# 将 PyTorch 数据拷贝到 Taichi
@ti.kernel
def sync_data(x_torch: ti.types.ndarray(), y_torch: ti.types.ndarray()):
    x[None] = x_torch[0]
    y[None] = y_torch[0]

# 3. 前向计算（Taichi 内核）
@ti.kernel
def forward():
    loss[None] = (x[None] ** 2 + y[None] ** 2)  # 示例损失函数

sync_data(x_torch, y_torch)  # 同步数据到 Taichi
# 4. 自动微分配置
with ti.ad.Tape(loss=loss):
    forward()

# 5. 反向传播并获取梯度
#forward.grad()
print(x.grad[None],y.grad[None])
x_torch.grad = x.grad.to_torch()[None]
y_torch.grad = y.grad.to_torch()[None]

# 将梯度赋给 PyTorch 张量
#x_torch.grad = grad_x.clone().detach()
#y_torch.grad = grad_y.clone().detach()

# 6. 优化器更新
optimizer.step()
optimizer.zero_grad()

print("Updated x:", x_torch)  # 应接近 0.8 (1 - 0.1*2)
print("Updated y:", y_torch)  # 应接近 1.8 (2 - 0.1*2)