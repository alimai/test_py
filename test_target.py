import numpy as np
import taichi as ti
ti.init(arch=ti.cpu)  # 初始化Taichi，使用CPU架构

ellipse_long = 0.5  # mm,椭圆的长轴
ellipse_short = 0.3  # mm,椭圆的短轴
ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))  # 椭圆的中心位置
ball_center[0] = [0, 0, 0]  # 初始化

n_x = 32  # 控制点行数
n_y = 2  # 控制点列数
tooth_size = 0.02#牙齿大小
dt = 3e-4  # 时间步长

spring_YP = 3e5  # 弹簧系数--长度相关
spring_YN = 3e7  # 弹簧系数--长度相关
dashpot_damping = 3e3  # 阻尼系数--速度差相关
drag_damping = 1e3  # 空气阻力系数
field_damping = 1e5

x = ti.Vector.field(3, dtype=float, shape=(n_x, n_y))  # 质点位置
v = ti.Vector.field(3, dtype=float, shape=(n_x, n_y))  # 质点速度

bending_springs = True  # 是否使用弯曲弹簧
spring_offsets = [] #弹簧偏移量---算子计算范围




r_level0 = 0.75 #网格数
r_level1 = 1.1#网格数，+1.1>1.0防止数值误差

block1 = ti.root.pointer(ti.ijk, (8, 4, 8))
block2 = block1.pointer(ti.ijk, (8, 4, 8))
pixel = block2.bitmasked(ti.ijk, (8, 4, 8))

field1 = ti.field(ti.f32)
field2 = ti.field(ti.f32)
pixel.place(field1, field2)

bg_n = pixel.shape
bg_size_x = ellipse_long * 2 * 1.2
bg_quad_size = bg_size_x / bg_n[0]
bg_size_y = bg_quad_size * bg_n[1]
bg_size_z = bg_quad_size * bg_n[2]
field_offset = []

@ti.kernel
def init_field_data()->int:
    bg_n_act = 0
    n_layers = int(0.1 / bg_quad_size)
    target_radius_long = ellipse_long / bg_quad_size
    target_radius_short = ellipse_short / bg_quad_size
    target_center = [bg_n[0]/2-0.5, bg_n[1]/2-0.5, bg_n[2]/2-0.5]
    target_radius_focal = ti.sqrt(target_radius_long**2 - target_radius_short**2)
    target_focal_top = [target_center[0], target_center[1], target_center[2]+target_radius_focal]
    target_focal_bottom = [target_center[0], target_center[1], target_center[2]-target_radius_focal]
    for i, j, k in ti.ndrange(bg_n[0], bg_n[1], bg_n[2]):
        dist_focal = ti.sqrt((i-target_focal_top[0])**2+(k-target_focal_top[2])**2)\
        +ti.sqrt((i-target_focal_bottom[0])**2+(k-target_focal_bottom[2])**2)
        dist_bias = dist_focal-target_radius_long*2
        if abs(dist_bias) <= r_level0+(n_layers-1)*r_level1:
            for n in range(n_layers):
                if abs(dist_bias) <= r_level0+n*r_level1:
                    field1[i, j, k] = n
                    field2[i, j, k] = -(r_level0+n_layers*r_level1+0.1)
                    bg_n_act += 1
                    break
                    # if dist_bias > 0:
                    #     field1[i, j, k] = n
                    #     field2[i, j, k] = -(r_level0+n_layers*r_level1+0.1)
                    #     bg_n_act += 1
                    #     break
                    # else:
                    #     field1[i, j, k] = -n
                    #     field2[i, j, k] = -(r_level0+n_layers*r_level1+0.1)
                    #     bg_n_act += 1
                    #     break
    return bg_n_act

bg_n_act = init_field_data()
field1_index = ti.Vector.field(3, dtype=float, shape=bg_n_act) 

@ti.kernel
def transe_field_data():
    bg_n_tmp=0
    ti.loop_config(serialize=True)
    for i, j, k in pixel:
        if j == ti.ceil(bg_n[1]/2) and k > bg_n[2]/2-0.5:#只绘制上半部分
            field1_index[bg_n_tmp] = [i*bg_quad_size-bg_size_x*0.5,
                                        j*bg_quad_size-bg_size_y*0.5,
                                        k*bg_quad_size-bg_size_z*0.5]
        bg_n_tmp += 1
    assert(bg_n_act == bg_n_tmp)

def add_field_offsets():
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if (i, j, k) != (0, 0, 0) and abs(i) + abs(j) <= 2:
                        field_offset.append(ti.Vector([i, j, k]))





@ti.kernel
def initialize_mass_points():
    cloth_size_x = ellipse_short * 2  # 布料大小
    cloth_size_y = cloth_size_x * n_y / n_x  # 布料大小        
    quad_size = cloth_size_x / n_x
    for i, j in x:# 初始化质点位置
        random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5, ti.random()]) * 0.02  # 随机偏移量
        x[i, j] = [
            i * quad_size - cloth_size_x * 0.5 + 0.5 * quad_size,
            j * quad_size - cloth_size_y * 0.5 + 0.5 * quad_size,
            0.0
        ] 
        x[i, j][0] *= abs(x[i, j][0]/ellipse_short)**0.6 * ellipse_short /(abs(x[i, j][0])+1e-5)#减少顶部质点密度
        x[i, j][2] = ellipse_long * 1.01 * (1-(x[i, j][0]/(ellipse_short*1.01))**2)**0.5#定义椭圆
        if i!=0 and i!=n_x-1:#固定两端
            x[i, j] += random_offset  # 添加随机偏移量
        v[i, j] = [0, 0, 0]  # 初始化质点速度

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
    #gravity = ti.Vector([0, 0, -9.8])  # 重力加速度
    # for n in ti.grouped(x):
    #     v[n] += gravity * dt  # 施加重力

    for n in ti.grouped(x):#core
        force = ti.Vector([0.0, 0.0, 0.0])
        for spring_offset in ti.static(spring_offsets):
            m = n + spring_offset
            if 0 <= m[0] < n_x and 0 <= m[1] < n_y:
                bias_x = x[n] - x[m]
                bias_v = v[n] - v[m]
                direct_mn = bias_x.normalized()
                current_dist = bias_x.norm()
                original_dist = tooth_size * spring_offset.norm()
                #弹簧力
                if current_dist > original_dist:
                    force += -spring_YP * direct_mn * (current_dist / original_dist - 1)
                else:
                    force += -spring_YN * direct_mn * (current_dist / original_dist - 1)
                #阻尼力
                force += -dashpot_damping * direct_mn * bias_v.dot(direct_mn) * tooth_size
        # 场力
        pos=ti.Vector([(x[n][0]+bg_size_x*0.5)/bg_quad_size, 
                       (x[n][1]+bg_size_y*0.5)/bg_quad_size,
                       (x[n][2]+bg_size_z*0.5)/bg_quad_size])
        for i in ti.static(range(3)):
            if abs(pos[i]-int(pos[i]))<1e-3: pos[i] += 1e-3#防止pos[i]为整数
            pos[i] = min(max(1e-3, pos[i]), bg_n[i]-1-1e-3)#限制边界
        pos_down = ti.floor(pos)
        for i in ti.static(range(3)):
            pos_up = pos_down
            pos_up[i] = ti.ceil(pos[i])
            direct_ud = (pos_up - pos_down).normalized()
            field_up = field1[ti.cast(pos_up, ti.i32)]
            field_down = field1[ti.cast(pos_down, ti.i32)]
            force += -(field_up-field_down)*direct_ud*field_damping
        if pos_down[1] > bg_n[1]*0.5+0.5:
            force += ti.Vector([0.0, -1.0, 0.0])*field_damping
        elif pos_down[1] < bg_n[1]*0.5-0.5:
            force += ti.Vector([0.0, 1.0, 0.0])*field_damping
        
        v[n] += force * dt  # 更新速度
        for i_c in ti.static(range(3)):
            tmpValue = abs(v[n][i_c])
            if tmpValue > 0.5:#限制最大速度
                v[n][i_c] /= tmpValue * 2

    for n in ti.grouped(x):
        v[n] *= ti.exp(-drag_damping * dt)  # 施加空气阻力
        # # 碰撞检测
        # offset_to_center = x[n] - ball_center[0]
        # offset_to_center[1] = 0#当作圆柱处理
        # #if offset_to_center.norm() <= ellipse_short:
        # if (ellipse_long*x[n][0])**2+(ellipse_short*x[n][2])**2 < (ellipse_short*ellipse_long)**2:
        #         normal = offset_to_center.normalized()# 速度投影
        #         v[n] -= min(v[n].dot(normal), 0) * normal
        if n[0]==0 or n[0]==n_x-1:#固定两端
            v[n][2] = 0
        x[n] += dt * v[n]  # 更新位置



if __name__ == '__main__':  # 主函数
    window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024), vsync=True)  # 创建窗口
    canvas = window.get_canvas()
    canvas.set_background_color((0.5, 0.5, 0.5))  # 设置背景颜色
    scene = window.get_scene()
    camera = ti.ui.make_camera()
    
    add_field_offsets()
    transe_field_data() # for display 

    add_spring_offsets()
    initialize_mass_points()  # 初始化质点

    current_t = 0.0
    substeps = 10#int(1 / 60 // dt)  # 每帧的子步数
    first_half = ti.Vector.field(3, dtype=float, shape=n_x)
    first_half = ti.Vector.field(3, dtype=float, shape=n_x)

    # bg_n_act = init_field_data()
    # field1_index = ti.Vector.field(3, dtype=float, shape=bg_n_act) 
    # transe_field_data(field1_index)
    # bg_n_tmp=0   
    # for i, j, k in pixel:
    #     field1_index[bg_n_tmp] = [i*bg_quad_size, j*bg_quad_size, k*bg_quad_size]
    #     bg_n_tmp += 1
    
    total_time = 1.0
    while window.running:
        if current_t > total_time:
            # 重置
            initialize_mass_points()
            current_t = 0

        for i in range(substeps):
            substep()  # 执行子步
            current_t += dt

        if current_t < total_time*0.5:
            camera.position(0.0, 2.0, 0.0)  # 设置相机位置
        else:
            camera.position(2.0 * np.sin((current_t-total_time*0.5)*np.pi*10),
                            2.0 * np.cos((current_t-total_time*0.5)*np.pi*10),
                            0.0)  # 设置相机位置
        camera.lookat(0.0, 0.0, 0.0)  # 设置相机观察点
        camera.up(0, 0, 1)
        scene.set_camera(camera)

        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))  # 设置点光源
        scene.ambient_light((0.5, 0.5, 0.5))  # 设置环境光
        #scene.mesh(vertices, indices=indices, per_vertex_color=colors, two_sided=True)  # 绘制网格
        # 绘制一个较小的球以避免视觉穿透
        #scene.particles(ball_center, radius=ellipse_short * 0.95, color=(0.5, 0.5, 0.5))

        for i in range(n_x):
            first_half[i] = x[i, 0]
        scene.particles(first_half, radius=tooth_size, color=(0.5, 0.42, 0.8))
        scene.particles(field1_index, radius=0.001, color=(0.5, 0.5, 0.5))

        canvas.scene(scene)
        window.show()

# TODO: 增加自碰撞处理
