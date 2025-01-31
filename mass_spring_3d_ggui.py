import taichi as ti
ti.init(arch=ti.cpu)  # 初始化Taichi，使用CPU架构

ellipse_short = 0.3  # 椭圆的短轴
ellipse_long = 0.5  # 椭圆的长轴
ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))  # 球的中心位置
ball_center[0] = [0, 0, 0]  # 初始化球的中心位置

n_x = 32  # 质点数目
cloth_size_x = ellipse_short * 2  # 布料大小
quad_size = cloth_size_x / (n_x - 1)  # 每个网格的大小

n_y = 2  # 质点数目
cloth_size_y = quad_size * (n_y - 1)  # 布料大小
dt = 3e-4  # 时间步长

gravity = ti.Vector([0, 0, -9.8])  # 重力加速度
spring_Y = 3e3  # 弹簧系数
dashpot_damping = 3e4  # 阻尼系数
drag_damping = 1e3  # 空气阻力系数

x = ti.Vector.field(3, dtype=float, shape=(n_x, n_y))  # 质点位置
v = ti.Vector.field(3, dtype=float, shape=(n_x, n_y))  # 质点速度

#用于绘制的数据
num_triangles = (n_x - 1) * (n_y - 1) * 2  # 三角形数量
indices = ti.field(int, shape=num_triangles * 3)  # 三角形顶点索引
vertices = ti.Vector.field(3, dtype=float, shape=n_x * n_y)  # 顶点位置
colors = ti.Vector.field(3, dtype=float, shape=n_x * n_y)  # 顶点颜色

bending_springs = True  # 是否使用弯曲弹簧
spring_offsets = []#弹簧偏移量

@ti.kernel
def initialize_mass_points():
    for i, j in x:# 初始化质点位置
        random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5, ti.random()]) * 0.02  # 随机偏移量
        x[i, j] = [
            i * quad_size - cloth_size_x * 0.5,
            j * quad_size - cloth_size_y * 0.5,
            0.0
        ] 
        x[i, j][0] *= abs(x[i, j][0]/ellipse_short)**0.6 * ellipse_short /(abs(x[i, j][0])+1e-5)#减少顶部质点密度
        x[i, j][2] = ellipse_long * 1.01 * (1-(x[i, j][0]/(ellipse_short*1.01))**2)**0.5#定义椭圆
        if i!=0 and i!=n_x-1:#固定两端
            x[i, j] += random_offset  # 添加随机偏移量
        v[i, j] = [0, 0, 0]  # 初始化质点速度

@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(n_x - 1, n_y - 1):
        quad_id = (j * (n_x - 1)) + i
        # 第一个三角形
        indices[quad_id * 6 + 0] = j * n_x + i
        indices[quad_id * 6 + 1] = (j + 1) * n_x + i
        indices[quad_id * 6 + 2] = j * n_x + (i + 1)
        # 第二个三角形
        indices[quad_id * 6 + 3] = (j + 1) * n_x + i + 1
        indices[quad_id * 6 + 4] = j * n_x + (i + 1)
        indices[quad_id * 6 + 5] = (j + 1) * n_x + i

    for i, j in ti.ndrange(n_x, n_y):
        if (j // 4 + i // 4) % 2 == 0:
            colors[j * n_x + i] = (0.22, 0.72, 0.52)  # 设置顶点颜色
        else:
            colors[j * n_x + i] = (0, 0.334, 0.52)  # 设置顶点颜色

def add_spring_offsets():
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
    for n in ti.grouped(x):
        v[n] += gravity * dt  # 施加重力

    for n in ti.grouped(x):
        force = ti.Vector([0.0, 0.0, 0.0])
        for spring_offset in ti.static(spring_offsets):
            m = n + spring_offset
            if 0 <= m[0] < n_x and 0 <= m[1] < n_y:
                x_ij = x[n] - x[m]
                v_ij = v[n] - v[m]
                d = x_ij.normalized()
                current_dist = x_ij.norm()
                original_dist = quad_size * float(n - m).norm()
                # 弹簧力
                force += -spring_Y * d * (current_dist / original_dist - 1)
                # 阻尼力
                force += -v_ij.dot(d) * d * dashpot_damping * quad_size
        dv = force * dt
        for i_c in ti.static(range(3)):
            tmpValue = v[n][i_c]
            if (dv[i_c]+tmpValue) * tmpValue < 0:
                dv[i_c]=-tmpValue
        v[n] += dv#force * dt  # 更新速度
        for i_c in ti.static(range(3)):
            tmpValue = abs(v[n][i_c])
            if tmpValue > 0.5:
                v[n][i_c] /= tmpValue * 2

    for n in ti.grouped(x):
        v[n] *= ti.exp(-drag_damping * dt)  # 施加空气阻力
        offset_to_center = x[n] - ball_center[0]
        offset_to_center[1] = 0#当作圆柱处理
        # 碰撞检测
        #if offset_to_center.norm() <= ellipse_short:
        if (ellipse_long*x[n][0])**2+(ellipse_short*x[n][2])**2 < (ellipse_short*ellipse_long)**2:
            if abs(n[0]*quad_size-cloth_size_x * 0.5-ball_center[0][0])<=quad_size*0.5:#固定中心点
                v[n] = [0, 0, 0]
            else:
                normal = offset_to_center.normalized()# 速度投影
                v[n] -= min(v[n].dot(normal), 0) * normal
        if n[0]!=0 and n[0]!=n_x-1:#固定两端
            x[n] += dt * v[n]  # 更新位置

@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n_x, n_y):
        vertices[j * n_x + i] = x[i, j]  # 更新顶点位置

if __name__ == '__main__':  # 主函数
    window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024), vsync=True)  # 创建窗口
    canvas = window.get_canvas()
    canvas.set_background_color((0.5, 0.5, 0.5))  # 设置背景颜色
    scene = window.get_scene()
    camera = ti.ui.make_camera()

    initialize_mesh_indices()  # 初始化网格索引
    add_spring_offsets()
    initialize_mass_points()  # 初始化质点

    current_t = 0.0
    substeps = 1#int(1 / 60 // dt)  # 每帧的子步数
    first_half = ti.Vector.field(3, dtype=float, shape=n_x)
    while window.running:
        if current_t > 1.0:
            # 重置
            initialize_mass_points()
            current_t = 0

        for i in range(substeps):
            substep()  # 执行子步
            current_t += dt
        update_vertices()  # 更新顶点

        camera.position(0.0, -2.0, 0.0)  # 设置相机位置
        camera.lookat(0.0, 0.0, 0.0)  # 设置相机观察点
        camera.up(0, 0, 1)
        scene.set_camera(camera)

        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))  # 设置点光源
        scene.ambient_light((0.5, 0.5, 0.5))  # 设置环境光
        #scene.mesh(vertices, indices=indices, per_vertex_color=colors, two_sided=True)  # 绘制网格
        # 绘制一个较小的球以避免视觉穿透
        scene.particles(ball_center, radius=ellipse_short * 0.95, color=(0.5, 0.5, 0.5))

        for i in range(n_x):
            first_half[i] = vertices[i]
        scene.particles(first_half, radius=0.02, color=(0.5, 0.42, 0.8))

        canvas.scene(scene)
        window.show()

# TODO: 增加自碰撞处理
