import numpy as np
import time
import torch
import taichi as ti
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot库

TEST_MODE = False#True#
ti.init(arch=ti.cpu, debug=TEST_MODE)#  # 初始化Taichi，使用CPU架构

#<<<<<初始量>>>>>
dt = 1e-4  # 时间步长
learning_rate = 1e-1  # 学习率
alpha = 1e-3  # 学习率衰减

spring_YP_base = 1e6  #1.2e6 # 引力系数--长度相关
spring_YN_base = 3e3  # 斥力系数--长度相关
dashpot_damping_base = 1e1  # 阻尼系数--速度差相关
drag_damping_base = 1.0  # 空气阻力系数
field_damping_base = 1e4

#分层数据结构
scalar = lambda: ti.field(dtype=ti.f32)  # 标量字段，用于place放入taichi分层数据中
vec = lambda: ti.Vector.field(3, dtype=ti.f32)  # 向量字段，用于place放入taichi分层数据中


#<<<<<场量>>>>>
ellipse_long = 0.6  # mm,椭圆的长轴
ellipse_short = 0.35  # mm,椭圆的短轴
r_level0 = 0.75 #网格数
r_level1 = 1.1 #网格数，+1.1>1.0防止数值误差
field_damping = ti.field(dtype=ti.f32, shape=())#scalar()

block1 = ti.root.pointer(ti.ijk, (8, 4, 8))
block2 = block1.pointer(ti.ijk, (8, 4, 8))
voxels = block2.bitmasked(ti.ijk, (8, 4, 8))

field1 = scalar()#ti.field(dtype=ti.f32)
field2 = scalar()#ti.field(dtype=ti.f32)
voxels.place(field1, field2)#, field_damping)

bg_n = voxels.shape
bg_size_x = ellipse_long * 2 * 1.2
bg_quad_size = bg_size_x / bg_n[0]
bg_size = ti.Vector([bg_size_x, bg_quad_size * bg_n[1], bg_quad_size * bg_n[2]])
field_offset =[]#不能在核函数内初始化
field_max = ti.field(dtype=ti.i32, shape=())

#<<<<<牙齿量>>>>>
n_x = 15  # 控制点行数
n_y = 3  # 控制点列数
tooth_size = 0.01#牙齿大小基准


spring_offsets =[] #弹簧偏移量---算子计算范围#不能在核函数内初始化
r = ti.field(dtype=ti.f32, shape=(n_x, n_y))  # 牙齿大小

#单层数据结构
# x = ti.Vector.field(3, dtype=float, shape=(n_x, n_y))  # 质点位置
# v = ti.Vector.field(3, dtype=float, shape=(n_x, n_y))  # 质点速度

spring_YP= scalar()  # 引力系数--长度相关
spring_YN = scalar()  # 斥力系数--长度相关
dashpot_damping = scalar()  # 阻尼系数--速度差相关
drag_damping = scalar()  # 空气阻力系数

max_steps = 256#512#1024
lay1 = ti.root.dense(ti.k, max_steps)
lay1.place(spring_YP, spring_YN, dashpot_damping, drag_damping)
lay2 = lay1.dense(ti.ij, (n_x, n_y))
x = vec()
v = vec()
f = vec()
l = scalar() #location in field
lay2.place(x, v, f, l)

batch_size = 5
spring_YP_grad = ti.field(dtype=ti.f32, shape=(max_steps))
spring_YN_grad = ti.field(dtype=ti.f32, shape=(max_steps))
dashpot_damping_grad = ti.field(dtype=ti.f32, shape=(max_steps))
drag_damping_grad = ti.field(dtype=ti.f32, shape=(256))

loss = scalar()
ti.root.place(loss)
ti.root.lazy_grad()
#lay1.lazy_grad()
#lay2.lazy_grad()
grad_max = ti.field(dtype=ti.f32, shape=())

spring_YP_th = torch.tensor([spring_YP_base]*max_steps, requires_grad=True)
spring_YN_th = torch.tensor([spring_YN_base]*max_steps, requires_grad=True)
dashpot_damping_th = torch.tensor([dashpot_damping_base]*max_steps, requires_grad=True)
drag_damping_th = torch.tensor([drag_damping_base]*max_steps, requires_grad=True)
params = [spring_YP_th, spring_YN_th, dashpot_damping_th]#, drag_damping_th]
optimizer = torch.optim.SGD(params, lr=learning_rate)#.Adam

@ti.kernel
def init_field_data()->int:
    field_max[None] = 0
    bg_n_act = 0
    n_layers = int(0.1 / bg_quad_size)
    target_radius_long = ellipse_long / bg_quad_size
    target_radius_short = ellipse_short / bg_quad_size
    target_center =[bg_n[0]/2-0.5, bg_n[1]/2-0.5, bg_n[2]/2-0.5]
    target_radius_focal = ti.sqrt(target_radius_long**2 - target_radius_short**2)
    target_focal_top =[target_center[0], target_center[1], target_center[2]+target_radius_focal]
    target_focal_bottom =[target_center[0], target_center[1], target_center[2]-target_radius_focal]
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
                    if n > field_max[None]: field_max[None]=n
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
    #for i, j, k in voxels:#无法串行化
    for i, j, k in ti.ndrange(bg_n[0], bg_n[1], bg_n[2]):
        if j == ti.ceil(bg_n[1]/2) and k > bg_n[2]/2-0.5:#只绘制上半部分
            if ti.is_active(voxels,[i,j,k]): 
                field1_index[bg_n_tmp] =ti.Vector([i,j,k])*bg_quad_size - bg_size * 0.5
                # field1_index[bg_n_tmp] =[i*bg_quad_size-bg_size_x*0.5,
                #                             j*bg_quad_size-bg_size_y*0.5,
                #                             k*bg_quad_size-bg_size_z*0.5]
                bg_n_tmp += 1
    #assert(bg_n_act == bg_n_tmp)#全部绘制时相等

def add_field_offsets():
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if (i, j, k) != (0, 0, 0) and abs(i) + abs(j) <= 2:
                        field_offset.append(ti.Vector([i, j, k]))




def output_spring_para():
    s_para = np.array([spring_YP.to_numpy(), spring_YN.to_numpy(), dashpot_damping.to_numpy(), drag_damping.to_numpy()])
    np.save('spring_para.npy', s_para)
def load_spring_para():
    #return False
    global spring_YP_th, spring_YN_th, dashpot_damping_th, drag_damping_th
    try:
        s_para = np.load('spring_para.npy')
    except FileNotFoundError:
        return False
    if(len(s_para) > 0):
        spring_YP.from_numpy(s_para[0])
        spring_YN.from_numpy(s_para[1])
        dashpot_damping.from_numpy(s_para[2])
        drag_damping.from_numpy(s_para[3])

        spring_YP_th = torch.from_numpy(s_para[0])
        spring_YN_th = torch.from_numpy(s_para[1])
        dashpot_damping_th = torch.from_numpy(s_para[2])
        drag_damping_th = torch.from_numpy(s_para[3])
        return True
    else:
        return False
@ti.kernel
def initialize_spring_para2():
    field_damping[None] = field_damping_base
    for t in range(max_steps):
        spring_YP[t]= spring_YP_base  
        spring_YN[t] = spring_YN_base  
        dashpot_damping[t] = dashpot_damping_base  
        drag_damping[t] = drag_damping_base

@ti.func
def re_update_grad_core(grad_max_cur: ti.f32):
    grad_max_used = grad_max_cur
    if(grad_max_used < grad_max[None]):
        grad_max_used = grad_max[None]

    for t in range(max_steps):
        #if t>=max_steps-2:
        spring_YP.grad[t] *= spring_YP[t] / grad_max_used
        spring_YN.grad[t] *= spring_YN[t]  / grad_max_used
        dashpot_damping.grad[t] *= dashpot_damping[t]  / grad_max_used
        drag_damping.grad[t] *= drag_damping[t]  / grad_max_used
@ti.kernel
def re_update_grad(iter: ti.i32)->ti.f32:
    grad_sum = ti.Vector([0.0, 0.0,0.0,0.0])
    for t in range(max_steps):
        spring_YP.grad[t] *= spring_YP[t]
        spring_YN.grad[t] *= spring_YN[t]
        dashpot_damping.grad[t] *= dashpot_damping[t]
        drag_damping.grad[t] *= drag_damping[t]        

        grad_sum[0] += abs(spring_YP.grad[t])
        grad_sum[1] += abs(spring_YN.grad[t])
        grad_sum[2] += abs(dashpot_damping.grad[t])
        grad_sum[3] += abs(drag_damping.grad[t])

    ti.sync()
    grad_sum_total = grad_sum.sum()
    #for i in range(grad_sum.n):
    #   print(elem)
    #   #grad_sum_total += grad_sum[i]
    #print("sug_grad_total: ", sug_grad_total)

    #if not np.isnan(grad_sum_total):---不能判断nan值
    if not ti.math.isnan(grad_sum_total):
        grad_max_cur = 0.0
        for t in range(max_steps):
            ti.atomic_max(grad_max_cur,abs(spring_YP.grad[t]))
            ti.atomic_max(grad_max_cur,abs(spring_YN.grad[t]))
            ti.atomic_max(grad_max_cur,abs(dashpot_damping.grad[t]))
            ti.atomic_max(grad_max_cur,abs(drag_damping.grad[t]))

        ti.sync()
        if iter <= 100:
            grad_max[None] = max(grad_max[None], grad_max_cur)
        re_update_grad_core(grad_max_cur)

    return grad_sum_total

@ti.kernel
def update_spring_para2(iter: ti.i32):
    # for t in range(max_steps):
    #     #if t>=max_steps-2:
    #     spring_YP[t] += -learning_rate * spring_YP.grad[t]
    #     spring_YN[t] += -learning_rate * spring_YN.grad[t]
    #     dashpot_damping[t] += -learning_rate * dashpot_damping.grad[t]
    #     drag_damping[t] += -learning_rate * drag_damping.grad[t]

    if iter%batch_size == 0:
        for t in range(max_steps):
            spring_YP_grad[t] = 0.0
            spring_YN_grad[t] = 0.0
            dashpot_damping_grad[t] = 0.0
            drag_damping_grad[t] = 0.0
    ti.sync()
    for t in range(max_steps):
        spring_YP_grad[t] += (spring_YP.grad[t])
        spring_YN_grad[t] += (spring_YN.grad[t])
        dashpot_damping_grad[t] += (dashpot_damping.grad[t])
        drag_damping_grad[t] += (drag_damping.grad[t])
    ti.sync()
    if (iter+1)%batch_size == 0:
        for t in range(max_steps):
            spring_YP[t] += -learning_rate * spring_YP_grad[t]# / batch_size#spring_YP.grad[t]
            spring_YN[t] += -learning_rate * spring_YN_grad[t]# / batch_size#spring_YN.grad[t]
            dashpot_damping[t] += -learning_rate * dashpot_damping_grad[t]# / batch_size#dashpot_damping.grad[t]
            drag_damping[t] += -learning_rate * drag_damping_grad[t]# / batch_size#drag_damping.grad[t]

#非ti.kernel#函数
def update_spring_para_th():
    optimizer.zero_grad()
    spring_YP_th.grad = spring_YP.grad.to_torch()
    spring_YN_th.grad = spring_YN.grad.to_torch()
    dashpot_damping_th.grad = dashpot_damping.grad.to_torch()
    drag_damping_th.grad = drag_damping.grad.to_torch()

    #更新参数
    optimizer.step()

    #print(spring_YP_th)
    spring_YP.from_torch(spring_YP_th)
    spring_YN.from_torch(spring_YN_th)
    dashpot_damping.from_torch(dashpot_damping_th)
    drag_damping.from_torch(drag_damping_th)

@ti.kernel
def initialize_mass_points(t: ti.i32):
    size_x = ellipse_short * 2  # 分布范围     
    quad_size = size_x / (n_x+1) # +1使X分布不对称
    size_y = n_y * quad_size#size_x * n_y / n_x  # 分布范围   
    index_center_x = 7.5
    for i, j in ti.ndrange(n_x, n_y):# 初始化质点位置
        random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5, ti.random() - 0.5]) * 0.03 #ti.Vector([0.01,0.01,0.01]) # 随机偏移量
        x[i, j, t] =[
            i * quad_size - size_x * 0.5 + 0.5 * quad_size,
            j * quad_size - size_y * 0.5 + 0.5 * quad_size,
            0.0
        ] 
        x[i, j, t][0] *= abs(x[i, j, t][0]/ellipse_short)**0.6 * ellipse_short /(abs(x[i, j, t][0])+1e-5)#减少顶部质点密度
        x[i, j, t][2] = ellipse_long * 1.01 * (1-(x[i, j, t][0]/(ellipse_short*1.01))**2)**0.5#定义椭圆
        if i!=0 and i!=n_x-1:#固定两端
            x[i, j, t] += random_offset  # 添加随机偏移量
        v[i, j, t] =[0, 0, 0]  # 初始化质点速度
        r[i, j] = tooth_size #初始化半径
        if abs(i - index_center_x)  > 2:
            r[i,j] += tooth_size * 0.5
        if abs(i - index_center_x)  > 3:
            r[i,j] += tooth_size * 0.5
        if abs(i - index_center_x)  > 5:
            r[i,j] += tooth_size * 0.5

def add_spring_offsets():
    for i in range(-1, 2):
            j=0#for j in range(-1, 2):#暂不考虑Y方向
            if (i, j) != (0, 0) :
                spring_offsets.append(ti.Vector([i, j]))  # 添加一阶弹簧偏移量
    # for i in range(-2, 3):
    #         j=0#for j in range(-2, 3):#暂不考虑Y方向
    #         if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
    #             spring_offsets.append(ti.Vector([i, j]))  # 添加二阶弹簧偏移量



@ti.func
def linear_interpolation(p: ti.template(), gradient: ti.template())->ti.f32:
    base = ti.floor(p, ti.i32)
    offset = p - base

    field_inter = 0.0
    gradient = ti.Vector([0.0,0.0,0.0])
    for i, j, k in ti.ndrange(2, 2, 2):
        neighbor = base + ti.Vector([i, j, k])
        value = field1[neighbor]
        if abs(value) > field_max[None]: value = field_max[None]
        # 计算标量权重
        weight = (1 - offset.x + (offset.x * 2 - 1) * i) * \
                 (1 - offset.y + (offset.y * 2 - 1) * j) * \
                 (1 - offset.z + (offset.z * 2 - 1) * k)
        field_inter += value * weight    
    
        # 计算梯度权重---容易梯度爆炸！！！
        # 原因应为ti.floor支持不好，某些时候插值结果只依赖base
        # 不包含其它参数
        # weight_x = (2 * i - 1) * \
        #          (1 - offset.y + (offset.y * 2 - 1) * j) * \
        #          (1 - offset.z + (offset.z * 2 - 1) * k)
        # weight_y = (1 - offset.x + (offset.x * 2 - 1) * i) * \
        #          (2 * j - 1) * \
        #          (1 - offset.z + (offset.z * 2 - 1) * k)
        # weight_z = (1 - offset.x + (offset.x * 2 - 1) * i) * \
        #          (1 - offset.y + (offset.y * 2 - 1) * j) * \
        #          (2 * k - 1)
        # gradient += value * ti.Vector([weight_x, weight_y, weight_z]) 

        #offset2 = p - neighbor
        weight_x = (2 * i - 1)# * (1 - offset2.y) * (1 - offset2.z)
        weight_y = (2 * j - 1)# * (1 - offset2.x) * (1 - offset2.z)
        weight_z = (2 * k - 1)# * (1 - offset2.x) * (1 - offset2.y) 
        gradient += value * ti.Vector([weight_x, weight_y, weight_z])
    
    return field_inter



@ti.kernel
def substep(t: ti.i32):     
    #gravity = ti.Vector([0, 0, -9.8])  # 重力加速度
    # for n in ti.grouped(x):
    #     v[n] += gravity * dt  # 施加重力
    #     

    for i, j in ti.ndrange(n_x, n_y):#for n in ti.grouped(v):#core
        index = ti.Vector([i, j, t])
        n = ti.Vector([i, j, t-1])
        force = ti.Vector([0.0, 0.0, 0.0])
        for offset_orig in ti.static(spring_offsets):
            spring_offset = ti.Vector([offset_orig[0], offset_orig[1], 0])
            m = n + spring_offset
            if 0 <= m[0] < n_x and 0 <= m[1] < n_y:        
                force_cur = ti.Vector([0.0, 0.0, 0.0])
                bias_x = x[n] - x[m]
                current_dist = bias_x.norm()
                original_dist = spring_offset.norm() * (r[n[0],n[1]] + r[m[0], m[1]]) #tooth_size

                direct_mn = bias_x.normalized() 
                m_mirror = n - spring_offset    
                if 0 <= m_mirror[0] < n_x and 0 <= m_mirror[1] < n_y: #重定义方向
                    bias_fnl = x[m_mirror]-x[m]
                    #bias_fnl[1] = bias_x[1]#y方向维持---直接对向量的某个分量赋值时其他分量的梯度信息可能会丢失。
                    bias_fnl = ti.Vector([bias_fnl[0], bias_x[1], bias_fnl[2]])#y方向维持
                    direct_mn = bias_fnl.normalized()
                    
                #弹簧力
                if current_dist > original_dist:
                    #force += -spring_YP * direct_mn * (current_dist / original_dist - 1)#**2
                    force_cur += -spring_YP[t-1] * direct_mn * (current_dist - original_dist)#**2
                else:
                    #force += spring_YN * direct_mn * (1 - current_dist / original_dist)#**0.5
                    force_cur += spring_YN[t-1] * direct_mn * (original_dist - current_dist)#**0.5
                #阻尼力
                #bias_v = v[n] - v[m]
                #force_cur += -dashpot_damping[t-1] * direct_mn * bias_v.dot(direct_mn) * (r[n[0],n[1]] + r[m[0], m[1]])#tooth_size
                force += force_cur

        # 场力
        pos = (x[n]+bg_size * 0.5)/bg_quad_size
        for ii in ti.static(range(3)):
            if abs(pos[ii]-int(pos[ii]))<1e-3: pos[ii] += 1e-3#防止pos[ii]为整数
            pos[ii] = min(max(1e-3, pos[ii]), bg_n[ii]-1-1e-3)#限制边界
        grad_field = ti.Vector([0.0,0.0,0.0])        
        field_inter = linear_interpolation(pos, grad_field)
        if force.norm() > 0 and grad_field.norm() > 0:
            direct_f1=force.normalized()
            direct_f2 = ti.Vector([0.0,1.0,0.0])
            direct_f3 = (direct_f1.cross(direct_f2)).normalized()
            if direct_f3.dot(grad_field) < 0:
                direct_f3 *= -1.0
            grad_field = grad_field.norm() * direct_f3
        #print(field_inter, grad_field, -grad_field*field_inter*field_damping[None], force)
        force += -grad_field*field_inter*field_damping[None]
        l[index] = field_inter
        f[index] = force

    ti.sync()
    for i, j in ti.ndrange(n_x, n_y):       
        index = ti.Vector([i, j, t])
        n = ti.Vector([i, j, t-1])
        if n[0]!=0 and n[0]!=n_x-1:#固定两端
            #v[n] = force * dt# 更新速度
            #v[index] += force * dt  # 更新速度
            v[index] = (v[n] + f[index] * dt) / (1.0+drag_damping[t-1])  # 更新速度并施加空气阻力
            #v[n] += (ti.random() - 0.5)*0.1 # 添加随机扰动
            # # 碰撞检测
            # offset_to_center = x[n] - ball_center[0]
            # offset_to_center[1] = 0#当作圆柱处理
            # #if offset_to_center.norm() <= ellipse_short:
            # if (ellipse_long*x[n][0])**2+(ellipse_short*x[n][2])**2 < (ellipse_short*ellipse_long)**2:
            #         normal = offset_to_center.normalized()# 速度投影
            #         v[n] += -min(v[n].dot(normal), 0) * normal
        else:
            v[index] = 0.0

    #更新位置
    ti.sync()
    for i, j in ti.ndrange(n_x, n_y):#for n in ti.grouped(x):#core        
        index = ti.Vector([i, j, t])
        n = ti.Vector([i, j, t-1])
        x[index] = (x[n] + dt * v[index]) 


@ti.kernel
def calcute_loss_dist(j: ti.i32): 
    for t in range(max_steps):
        avg_bias = 0.0       
        list_dist = ti.Vector([0.0] * (n_x-1))
        for i in ti.static(range(n_x-1)):
            list_dist[i] = (x[i, j, t] - x[i+1, j, t]).norm() - (r[i,j]+r[i+1,j])
            avg_bias += list_dist[i]
        avg_bias /= (n_x-1)
        loss_step = 0.0
        for i in ti.static(range(n_x-1)):
            loss_step += abs(list_dist[i]-avg_bias)
        loss[None] += loss_step*t**2*1e1

@ti.kernel
def calcute_loss_x(j: ti.i32):
    for t in range(max_steps):
        for i in ti.static(range(n_x)):
            loss[None] += l[i,j,t]*t**2*1e-2
@ti.kernel
def calcute_loss_v(j: ti.i32):
    for t in range(max_steps):
        for i in ti.static(range(n_x)):
            loss[None] += v[i,j,t].norm()*t**2*1e-3 


def compute_loss():
    j = n_y//2
    loss[None] = 0.0
    calcute_loss_dist(j)
    #print(loss[None])
    calcute_loss_x(j) 
    #print(loss[None])
    calcute_loss_v(j) 
 
point = ti.Vector.field(3, dtype=float, shape=1) # for display 
def run_windows(window, n, keep = False):
    if window is None:
        window = ti.ui.Window("Teeth target Simulation", (1024, 1024), vsync=True)  # 创建窗口
    canvas = window.get_canvas()
    canvas.set_background_color((0.3, 0.3, 0.3))  # 设置背景颜色
    scene = window.get_scene()
    camera = ti.ui.make_camera()

    if n < max_steps*0.5:
        camera.position(0.0, 2.0, 0.0)  # 设置相机位置
    else:
        camera.position(2.0 * np.sin((n-max_steps*0.5) / max_steps *np.pi*4),
                        2.0 * np.cos((n-max_steps*0.5) / max_steps *np.pi*4),
                        0.0)  # 设置相机位置
    camera.position(0.0, 2.0, 0.0)  # 设置相机位置
    camera.lookat(0.0, 0.0, 0.0)  # 设置相机观察点
    camera.up(0, 0, 1)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))  # 设置点光源
    scene.ambient_light((0.5, 0.5, 0.5))  # 设置环境光
    for i in range(4):
        point[0] =[0.0, 0.0, 0.0]
        color =[0.0, 0.0, 0.0]
        if i < 3:
            point[0][i] = 0.05
            color[i] = 1.0
        scene.particles(point, radius=0.01 if i!=3 else 0.02, color=tuple(color))
    for i in range(n_x):
        point[0] = x[i, 1, n]
        scene.particles(point, radius=r[i,1]+0.02, color=(0.5, 0.42, 0.8))

    scene.particles(field1_index, radius=0.001, color=(0.5, 0.5, 0.5))
    #scene.mesh(vertices, indices=indices, per_vertex_color=colors, two_sided=True)  # 绘制网格

    canvas.scene(scene)
    window.show()
    if keep:
        input()

if __name__ == '__main__':  # 主函数 

    max_iter = 100# 最大迭代次数 
    transe_field_data() # for display

    window = None      
    disp_by_step = False#True#
    if not TEST_MODE and disp_by_step:
        window = ti.ui.Window("Teeth target Simulation", (1024, 1024), vsync=True)  # 创建窗口

    add_field_offsets()
    add_spring_offsets()
    initialize_spring_para2()        
    #load_spring_para()
    print(spring_YP[0], spring_YN[0], dashpot_damping[0], drag_damping[0])
    print(spring_YP[1], spring_YN[1], dashpot_damping[1], drag_damping[1])
    print(spring_YP[max_steps//2], spring_YN[max_steps//2], dashpot_damping[max_steps//2], drag_damping[max_steps//2])
    print(spring_YP[max_steps-1], spring_YN[max_steps-1], dashpot_damping[max_steps-1], drag_damping[max_steps-1])

    
    spring_YPs=[]
    losses =[]  # 损失列表
    for iter in range(max_iter):#while window.running:
        #print('\nIter=', iter)
        loss[None] = 0.0
        initialize_mass_points(0)
        with ti.ad.Tape(loss=loss, validation=TEST_MODE): # 使用自动微分
            for n in range(1, max_steps):            
                substep(n)  # 执行子步
                if not TEST_MODE and disp_by_step:
                    if iter % (max_iter//10) == 0: #display 
                        if n % 10 == 1:#if n % (max_steps-1) == 0: 
                            run_windows(window, n)
            compute_loss()
  
        
        learning_rate *= (1.0 - alpha)
        grad_sum_total = re_update_grad(iter)
        if not np.isnan(grad_sum_total):
            update_spring_para2(iter)#update_spring_para_th()#
        else:
            print(loss[None], grad_sum_total)
            continue#break#       
        losses.append(loss[None])  # 添加损失到列表
        spring_YPs.append(spring_YP[max_steps//2])         
        
        if iter % (max_iter//100) == 0:
            print('\nX=', iter, ', Y=', loss[None], ", Z=", grad_sum_total)
        # print(spring_YP.grad[0], spring_YN.grad[0], dashpot_damping.grad[0], drag_damping.grad[0])
        # print(spring_YP.grad[1], spring_YN.grad[1], dashpot_damping.grad[1], drag_damping.grad[1])
        # print(spring_YP.grad[max_steps//2], spring_YN.grad[max_steps//2], dashpot_damping.grad[max_steps//2])#, drag_damping.grad[max_steps//2])
        # print(spring_YP.grad[0], spring_YP.grad[1], spring_YP.grad[max_steps-2], spring_YP.grad[max_steps-1])
        # print(spring_YP[0], spring_YN[0], dashpot_damping[0], drag_damping[0])
        # print(spring_YP[1], spring_YN[1], dashpot_damping[1], drag_damping[1])
        #print(spring_YP[max_steps//2], spring_YN[max_steps//2], dashpot_damping[max_steps//2], drag_damping[max_steps//2])
        #print(spring_YP[0], spring_YP[1], spring_YP[max_steps-2], spring_YP[max_steps-1])
        #break

    
    print(spring_YP[0], spring_YN[0], dashpot_damping[0], drag_damping[0])
    print(spring_YP[1], spring_YN[1], dashpot_damping[1], drag_damping[1])
    print(spring_YP[max_steps//2], spring_YN[max_steps//2], dashpot_damping[max_steps//2], drag_damping[max_steps//2])
    print(spring_YP[max_steps-1], spring_YN[max_steps-1], dashpot_damping[max_steps-1], drag_damping[max_steps-1])
    
    #output_spring_para()
    #run_windows(window, max_steps-1, keep = True)
    spring_YPs_2=[]
    for t in range(max_steps-1):        
        spring_YPs_2.append(spring_YP[t])
    fig,axs = plt.subplots(3)
    axs[0].plot(losses)  # 绘制损失曲线
    axs[1].plot(spring_YPs)  # 绘制损失曲线
    axs[2].plot(spring_YPs_2)  # 绘制损失曲线
    plt.tight_layout()  # 紧凑布局
    plt.show()  # 显示图像


# YODO：可将每层每个牙齿的弹簧参数用自动微分优化
# TODO: 应用各种优化算法，如deepmind的蒙特卡洛树搜索，deepseek的GRPO等
# TODO: 增加自碰撞处理
