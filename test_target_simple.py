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

#分层数据结构
scalar = lambda: ti.field(dtype=ti.f32)  # 标量字段，用于place放入taichi分层数据中
vec = lambda: ti.Vector.field(3, dtype=ti.f32)  # 向量字段，用于place放入taichi分层数据中


#<<<<<场量>>>>>
ellipse_long = 0.6  # mm,椭圆的长轴
ellipse_short = 0.35  # mm,椭圆的短轴


#<<<<<牙齿量>>>>>
n_x = 15  # 控制点行数
n_y = 3  # 控制点列数
tooth_size = 0.01#牙齿大小基准


spring_offsets =[] #弹簧偏移量---算子计算范围#不能在核函数内初始化
r = ti.field(dtype=ti.f32, shape=(n_x, n_y))  # 牙齿大小

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


@ti.kernel
def initialize_spring_para2():
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
            if(grad_max[None] > loss[None]):
                grad_max[None] = loss[None]
        re_update_grad_core(grad_max_cur)

    return grad_sum_total

@ti.kernel
def update_spring_para2(iter: ti.i32):
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


@ti.kernel
def initialize_mass_points(t: ti.i32):
    size_x = ellipse_short * 2  # 分布范围     
    quad_size = size_x / (n_x+1) # +1使X分布不对称
    size_y = n_y * quad_size#size_x * n_y / n_x  # 分布范围   
    index_center_x = 7.5
    for i, j in ti.ndrange(n_x, n_y):# 初始化质点位置
        random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5, ti.random()]) * 0.03 #ti.Vector([0.01,0.01,0.01]) # 随机偏移量
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
        for i in range(-2, 3):
                j=0#for j in range(-2, 3):#暂不考虑Y方向
                if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
                    spring_offsets.append(ti.Vector([i, j]))  # 添加普通弹簧偏移量





@ti.kernel
def substep(t: ti.i32):         

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
                current_dist = bias_x.norm() - (r[n[0],n[1]] + r[m[0], m[1]])*0.5
                original_dist = spring_offset.norm() * (r[n[0],n[1]] + r[m[0], m[1]])*0.5 #tooth_size

                direct_mn = bias_x.normalized() 
                    
                #弹簧力
                if current_dist > original_dist:
                    #force += -spring_YP * direct_mn * (current_dist / original_dist - 1)#**2
                    force_cur += -spring_YP[t-1] * direct_mn * (current_dist - original_dist)#**2
                else:
                    #force += spring_YN * direct_mn * (1 - current_dist / original_dist)#**0.5
                    force_cur += spring_YN[t-1] * direct_mn * (original_dist - current_dist)#**0.5
                force += force_cur
    
        force += ti.Vector([0, 0, 9000.8])  # 重力加速度
        f[index] = force

    ti.sync()
    for i, j in ti.ndrange(n_x, n_y):       
        index = ti.Vector([i, j, t])
        n = ti.Vector([i, j, t-1])
        if n[0]!=0 and n[0]!=n_x-1:#固定两端
            #v[n] = force * dt# 更新速度
            #v[index] += force * dt  # 更新速度
            v[index] = (v[n] + f[index] * dt) / (1.0+drag_damping[t-1])  # 更新速度并施加空气阻力
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


def compute_loss():
    j = n_y//2
    loss[None] = 0.0
    calcute_loss_dist(j)
    #print(loss[None])
    #print(loss[None])
 

if __name__ == '__main__':  # 主函数 

    max_iter = 100# 最大迭代次数 

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
        
        if iter % (max_iter//50) == 0:
            print('\nX=', iter, ', Y=', loss[None], ", Z=", grad_sum_total)

    
    print(spring_YP[0], spring_YN[0], dashpot_damping[0], drag_damping[0])
    print(spring_YP[1], spring_YN[1], dashpot_damping[1], drag_damping[1])
    print(spring_YP[max_steps//2], spring_YN[max_steps//2], dashpot_damping[max_steps//2], drag_damping[max_steps//2])
    print(spring_YP[max_steps-1], spring_YN[max_steps-1], dashpot_damping[max_steps-1], drag_damping[max_steps-1])
    
    spring_YPs_2=[]
    # for t in range(max_steps-1):        
    #     spring_YPs_2.append(spring_YP[t])
    for n in range(n_x):     
        spring_YPs_2.append(x[n,1,max_steps-1][2])
    fig,axs = plt.subplots(3)
    axs[0].plot(losses)  # 绘制损失曲线
    axs[1].plot(spring_YPs)  # 绘制损失曲线
    axs[2].plot(spring_YPs_2)  # 绘制损失曲线
    plt.tight_layout()  # 紧凑布局
    plt.show()  # 显示图像


# TODO: 应用各种优化算法，如deepmind的蒙特卡洛树搜索，deepseek的GRPO等
# TODO: 增加自碰撞处理
