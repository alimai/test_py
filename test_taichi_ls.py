import taichi as ti
import math

ti.init(arch=ti.cpu)  # 初始化 Taichi，使用 CPU 后端

# 参数设置
N = 200  # 网格分辨率
band_width = 5  # 窄带宽度
dt = 0.1  # 时间步长
reinit_freq = 100  # 重新初始化频率

# 定义场
phi = ti.field(dtype=ti.f32, shape=(N, N))  # Level Set 场
phi_new = ti.field(dtype=ti.f32, shape=(N, N))  # 更新后的 Level Set 场
is_active = ti.field(dtype=ti.i32, shape=(N, N))  # 标记活动点

# 初始化 Level Set 场为圆形
@ti.kernel
def initialize():
    for i, j in ti.ndrange(N, N):  # 使用 ti.ndrange 遍历二维场
        x = (i - N / 2) / N * 2
        y = (j - N / 2) / N * 2
        phi[i, j] = ti.sqrt(x**2 + y**2) - 0.5  # 圆形界面

# 标记活动点
@ti.kernel
def mark_active_points():
    for i, j in ti.ndrange(N, N):
        if abs(phi[i, j]) <= band_width:
            is_active[i, j] = 1
        else:
            is_active[i, j] = 0

# 更新窄带内的 Level Set 值
@ti.kernel
def update_narrow_band():
    for i, j in ti.ndrange(N, N):
        if is_active[i, j]:
            # 简单演化：沿法线方向移动
            phi_new[i, j] = phi[i, j] - dt * 1.0  # 假设速度为 1.0
        else:
            phi_new[i, j] = phi[i, j]

# 交换场
@ti.kernel
def swap_fields():
    for i, j in ti.ndrange(N, N):
        phi[i, j] = phi_new[i, j]

# 重新初始化 Level Set 场
@ti.kernel
def reinitialize():
    for i, j in ti.ndrange(N, N):
        if is_active[i, j]:
            # 手动实现 copysign 逻辑
            sign = 1.0 if phi[i, j] >= 0 else -1.0
            phi[i, j] = sign * band_width


# 主函数
def main():
    initialize()  # 初始化 Level Set 场
    gui = ti.GUI("Sparse Field Level Set", res=(N, N))

    step = 0
    while gui.running:
        mark_active_points()  # 标记活动点
        update_narrow_band()  # 更新窄带内的 Level Set 值

        swap_fields()  # 交换场

        # 定期重新初始化
        if step % reinit_freq == 0:
            reinitialize()

        # 可视化
        gui.set_image(phi.to_numpy())
        gui.show()
        step += 1
        print(step)

if __name__ == "__main__":
    main()