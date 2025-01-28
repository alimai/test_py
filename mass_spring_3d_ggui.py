import taichi as ti
ti.init(arch=ti.cpu)  # 初始化Taichi，使用CPU架构

n = 128  # 网格大小
quad_size = 1.0 / n  # 每个网格的大小
dt = 4e-2 / n  # 时间步长
substeps = int(1 / 60 // dt)  # 每帧的子步数

gravity = ti.Vector([0, -9.8, 0])  # 重力加速度
spring_Y = 3e4  # 弹簧系数
dashpot_damping = 1e4  # 阻尼系数
drag_damping = 1  # 空气阻力系数

ball_radius = 0.3  # 球的半径
ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))  # 球的中心位置
ball_center[0] = [0, 0, 0]  # 初始化球的中心位置

x = ti.Vector.field(3, dtype=float, shape=(n, n))  # 质点位置
v = ti.Vector.field(3, dtype=float, shape=(n, n))  # 质点速度

num_triangles = (n - 1) * (n - 1) * 2  # 三角形数量
indices = ti.field(int, shape=num_triangles * 3)  # 三角形顶点索引
vertices = ti.Vector.field(3, dtype=float, shape=n * n)  # 顶点位置
colors = ti.Vector.field(3, dtype=float, shape=n * n)  # 顶点颜色

bending_springs = False  # 是否使用弯曲弹簧

@ti.kernel
def initialize_mass_points():
    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1  # 随机偏移量

    for i, j in x:
        x[i, j] = [
            i * quad_size - 0.5 + random_offset[0], 0.6,
            j * quad_size - 0.5 + random_offset[1]
        ]  # 初始化质点位置
        v[i, j] = [0, 0, 0]  # 初始化质点速度

@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = (i * (n - 1)) + j
        # 第一个三角形
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + (j + 1)
        # 第二个三角形
        indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
        indices[quad_id * 6 + 4] = i * n + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * n + j

    for i, j in ti.ndrange(n, n):
        if (i // 4 + j // 4) % 2 == 0:
            colors[i * n + j] = (0.22, 0.72, 0.52)  # 设置顶点颜色
        else:
            colors[i * n + j] = (1, 0.334, 0.52)  # 设置顶点颜色

initialize_mesh_indices()  # 初始化网格索引

spring_offsets = []
if bending_springs:
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i, j) != (0, 0):
                spring_offsets.append(ti.Vector([i, j]))  # 添加弯曲弹簧偏移量

else:
    for i in range(-2, 3):
        for j in range(-2, 3):
            if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
                spring_offsets.append(ti.Vector([i, j]))  # 添加普通弹簧偏移量

@ti.kernel
def substep():
    for i in ti.grouped(x):
        v[i] += gravity * dt  # 施加重力

    for i in ti.grouped(x):
        force = ti.Vector([0.0, 0.0, 0.0])
        for spring_offset in ti.static(spring_offsets):
            j = i + spring_offset
            if 0 <= j[0] < n and 0 <= j[1] < n:
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                d = x_ij.normalized()
                current_dist = x_ij.norm()
                original_dist = quad_size * float(i - j).norm()
                # 弹簧力
                force += -spring_Y * d * (current_dist / original_dist - 1)
                # 阻尼力
                force += -v_ij.dot(d) * d * dashpot_damping * quad_size

        v[i] += force * dt  # 更新速度

    for i in ti.grouped(x):
        v[i] *= ti.exp(-drag_damping * dt)  # 施加空气阻力
        offset_to_center = x[i] - ball_center[0]
        if offset_to_center.norm() <= ball_radius:
            # 速度投影
            normal = offset_to_center.normalized()
            v[i] -= min(v[i].dot(normal), 0) * normal
        x[i] += dt * v[i]  # 更新位置

@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]  # 更新顶点位置

window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024), vsync=True)  # 创建窗口
canvas = window.get_canvas()
canvas.set_background_color((0.5, 0.5, 0.5))  # 设置背景颜色
scene = window.get_scene()
camera = ti.ui.make_camera()

current_t = 0.0
initialize_mass_points()  # 初始化质点

while window.running:
    if current_t > 1.5:
        # 重置
        initialize_mass_points()
        current_t = 0

    for i in range(substeps):
        substep()  # 执行子步
        current_t += dt
    update_vertices()  # 更新顶点

    camera.position(0.0, 0.0, 3)  # 设置相机位置
    camera.lookat(0.0, 0.0, 0)  # 设置相机观察点
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))  # 设置点光源
    scene.ambient_light((0.5, 0.5, 0.5))  # 设置环境光
    scene.mesh(vertices, indices=indices, per_vertex_color=colors, two_sided=True)  # 绘制网格

    # 绘制一个较小的球以避免视觉穿透
    scene.particles(ball_center, radius=ball_radius * 0.95, color=(0.5, 0.42, 0.8))
    canvas.scene(scene)
    window.show()

# TODO: 增加自碰撞处理
