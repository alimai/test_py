
import time
import numpy as np
import torch
import taichi as ti
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot库

# 设备配置
device = torch.device('cpu')

# 基本参数
dt = 1e-4  # 时间步长
learning_rate = 1e-1  # 学习率
alpha = 1e-3  # 学习率衰减

# 物理参数
spring_YP_base = 1e6#1e6/(1e6+1)#  
spring_YN_base = 3e3#3e3/(3e3+1)#
dashpot_damping_base = 1e1#1e1/(1e1+1)#  
drag_damping_base = 1.0  

# 几何参数
ellipse_long = 0.6
ellipse_short = 0.35
n_x = 15
n_y = 3
tooth_size = 0.01
max_steps = 256
batch_size = 5

class MassSpringSystem:
    def __init__(self):
         # 创建参数张量
        self.spring_YP = torch.full((max_steps,), spring_YP_base, device=device, requires_grad=True)
        self.spring_YN = torch.full((max_steps,), spring_YN_base, device=device, requires_grad=True)
        self.dashpot_damping = torch.full((max_steps,), dashpot_damping_base, device=device, requires_grad=True)
        self.drag_damping = torch.full((max_steps,), drag_damping_base, device=device, requires_grad=True)
        
        
        # 创建状态张量，这些不需要梯度
        self.x = torch.zeros((max_steps, n_x, n_y, 3), device=device)#, requires_grad=True)
        self.v = torch.zeros((max_steps, n_x, n_y, 3), device=device)#, requires_grad=True)
        self.f = torch.zeros((max_steps, n_x, n_y, 3), device=device)#, requires_grad=True)
    
        # 创建半径张量
        self.r = torch.full((n_x, n_y), tooth_size, device=device)
        
        # 创建弹簧偏移列表
        self.spring_offsets = self._create_spring_offsets()
        
    def _create_spring_offsets(self):
        offsets = []
        for i in range(-1, 2):
                j=0#for j in range(-1, 2):#暂不考虑Y方向
                if (i, j) != (0, 0) :
                    offsets.append(torch.tensor([i, j]))  # 添加一阶弹簧偏移量
        return offsets
    
    def gen_start_pos(self):
        size_x = ellipse_short * 2
        quad_size = size_x / (n_x + 1)# +1使X分布不对称
        size_y = n_y * quad_size
        
        # 创建网格点
        i_coords = torch.arange(n_x, device=device)
        j_coords = torch.arange(n_y, device=device)
        i, j = torch.meshgrid(i_coords, j_coords, indexing='ij')
        
        # # 计算初始位置
        # x = (i * quad_size - size_x * 0.5 + 0.5 * quad_size).unsqueeze(-1)
        # y = (j * quad_size - size_y * 0.5 + 0.5 * quad_size).unsqueeze(-1)
        # z = torch.zeros_like(x)
        # # 组合坐标
        # pos = torch.cat([x, y, z], dim=-1)
        # 等价于上边注释代码
        pos = torch.empty((n_x,n_y,3), device=device)
        pos[...,0]=i * quad_size - size_x * 0.5 + 0.5 * quad_size
        pos[...,1]=j * quad_size - size_y * 0.5 + 0.5 * quad_size
        pos[...,2]=torch.zeros_like(pos[..., 0], device=device)
        
        # 调整椭圆形状#操作每个元素(此处为3维坐标)第0/2个数据
        pos[..., 0] *= torch.abs(pos[..., 0]/ellipse_short).pow(0.6) * ellipse_short / (torch.abs(pos[..., 0])+1e-5)
        pos[..., 2] = ellipse_long * 1.01 * (1-(pos[..., 0]/(ellipse_short*1.01)).pow(2)).sqrt()
        
        # 添加随机偏移(除了边界点)
        mask = torch.ones_like(pos)
        #操作第0/-1列数据：mask[:, 0] = mask[:, -1] = 0
        #操作第0/-1行数据
        mask[0, :] = mask[-1, :] = 0
        # 添加随机偏移
        random_offset = (torch.rand(n_x, n_y, 3, device=device) - 0.5) * 0.03
        pos += random_offset * mask

        return pos

    def initialize_mass_points(self):
        # 重新初始化状态张量
        self.x = torch.zeros((max_steps, n_x, n_y, 3), device=device)
        self.v = torch.zeros((max_steps, n_x, n_y, 3), device=device)
        self.f = torch.zeros((max_steps, n_x, n_y, 3), device=device)    
        self.x[0] = self.gen_start_pos()

    def compute_forces(self, t):
        forces = torch.zeros_like(self.x[t])
        current_pos = self.x[t-1]
        
        # 计算弹力
        for offset in self.spring_offsets:
            i_shift = offset[0].item()
            j_shift = offset[1].item()
            
            # 计算相邻点的距离和方向
            rolled_pos = torch.roll(current_pos, shifts=(i_shift, j_shift), dims=(0, 1))
            bias_x = current_pos - rolled_pos
            dist = torch.norm(bias_x, dim=-1, keepdim=True)
            direction = bias_x / (dist + 1e-7)

            #修正力传递方向
            rolled_pos_mirror = torch.roll(current_pos, shifts=(-i_shift, -j_shift), dims=(0, 1))   
            bias_x_center = rolled_pos_mirror - rolled_pos 
            bias_x_center[...,1] = bias_x [...,1]
            dist_center = torch.norm(bias_x_center, dim=-1, keepdim=True) 
            direction_center = bias_x_center / (dist_center + 1e-7)  
            bias_x_mirror = current_pos - rolled_pos_mirror 
            bias_x_mirror[...,1] = bias_x [...,1]
            dist_mirror = torch.norm(bias_x_mirror, dim=-1, keepdim=True) 
            direction_mirror = bias_x_mirror / (dist_mirror + 1e-7) 
            direction_normal = (direction+direction_mirror) * 0.5
            ratio_normal = torch.norm(torch.cross(direction, direction_center, dim = -1))

            # 计算原始距离
            r_rolled = torch.roll(self.r, shifts=i_shift, dims=0)  # 先在 i 方向滚动
            r_rolled = torch.roll(r_rolled, shifts=j_shift, dims=1)  # 再在 j 方向滚动
            original_dist = (torch.norm(offset.float()) * (self.r + r_rolled) * 0.5).unsqueeze(-1)
        
            # 计算弹力
            stretch_mask = dist > original_dist
            #compress_mask = ~stretch_mask
            
            forces += torch.where(stretch_mask,
                                -abs(self.spring_YP[t-1]) * (direction_center+direction_normal*ratio_normal) * (dist - original_dist),#/(1-abs(self.spring_YP[t-1]))
                                abs(self.spring_YN[t-1]) * (direction_center+direction_normal*ratio_normal) * (original_dist - dist))#/(1-abs(self.spring_YN[t-1]))
        
        # 添加重力
        forces[..., 2] += 9000.8
        
        return forces
    
    def step(self, t):
        # 计算力
        self.f[t] = self.compute_forces(t)
        
        # 更新速度
        mask = torch.ones_like(self.v[t])
        mask[0, :] = mask[-1, :] = 0
        
        self.v[t] = mask * (self.v[t-1] + self.f[t] * dt) / (1.0 + self.drag_damping[t-1])

        # 更新位置
        self.x[t] = self.x[t-1] + dt * self.v[t]

    def compute_loss(self):
        j = n_y // 2
        loss = torch.tensor(0.0, device=device)
        
        for t in range(max_steps):
            biass = self.x[t, 1:, j] - self.x[t, :-1, j]
            dists = torch.norm(biass, dim=1)
            target_dists = self.r[1:, j] + self.r[:-1, j]
            rel_dists = torch.abs(dists - target_dists)
            
            # avg_dist = torch.mean(rel_dists)
            # step_loss = torch.sum(torch.abs(rel_dists - avg_dist))
            # loss += step_loss * (t ** 2) * 1e1
            #rel_dists的方差
            loss += torch.var(rel_dists) * (t**2) * 1e2            
            loss += torch.sum(torch.norm(self.v[t,:,j], dim=1))*(t**2)*1e-1 
        return loss

def output_spring_para(system):
    s_para = np.array([system.spring_YP.detach().numpy(), system.spring_YN.detach().numpy(), \
                       system.dashpot_damping.detach().numpy(), system.drag_damping.detach().numpy()])
    np.save('spring_para_pt.npy', s_para)
def load_spring_para(system):
    #return False
    try:
        s_para = np.load('spring_para_pt.npy')
    except FileNotFoundError:
        return False
    if(len(s_para) > 0):
        system.spring_YP.data = torch.from_numpy(s_para[0])#.requires_grad_(True)
        system.spring_YN.data = torch.from_numpy(s_para[1])#.requires_grad_(True)
        system.dashpot_damping.data = torch.from_numpy(s_para[2])#.requires_grad_(True)
        system.drag_damping.data = torch.from_numpy(s_para[3])#.requires_grad_(True)
        return True
    else:
        return False
    
ti.init(arch=ti.cpu)
point = ti.Vector.field(3, dtype=float, shape=1) # for display 
points = ti.Vector.field(3, dtype=float, shape=n_x) # for display 
r_points = ti.field(dtype=float, shape=n_x) # for display 
def run_windows(window, n, system, keep = False):
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
    # points.from_torch(system.x[:,1,n])
    # r_points.from_torch(system.r[:,1])
    for i in range(n_x):
        point[0] = system.x[n, i, 1]#.item()
        scene.particles(point, radius=system.r[i,1].item()+0.02, color=(0.5, 0.42, 0.8))

    #scene.particles(field1_index, radius=0.001, color=(0.5, 0.5, 0.5))
    #scene.mesh(vertices, indices=indices, per_vertex_color=colors, two_sided=True)  # 绘制网格

    canvas.scene(scene)
    window.show()
    if keep:
        input()

def main():   
    max_iter = 100# 最大迭代次数 
    disp_by_step = True#False#
    window = None   
    if disp_by_step:
        window = ti.ui.Window("Teeth target Simulation", (1024, 1024), vsync=True)  # 创建窗口

    system = MassSpringSystem()
    optimizer = torch.optim.SGD([
        system.spring_YP,
        system.spring_YN,
        system.dashpot_damping,
        system.drag_damping,
    ], lr=learning_rate)

    losses = []
    spring_YPs = []    
    #load_spring_para(system)
    
    for iter in range(max_iter):
        optimizer.zero_grad()        
        system.initialize_mass_points()

        # 前向传播
        #with torch.set_grad_enabled(True):
        for t in range(1, max_steps):
            system.step(t)
            if disp_by_step:
                if iter % (max_iter//10) == 0: #display 
                    if t % 10 == 1:#if n % (max_steps-1) == 0: 
                        run_windows(window, t, system)
        
        # 计算损失并反向传播
        loss = system.compute_loss()
        loss.backward(retain_graph=True)
        for t in range(0, max_steps):
           system.spring_YP.grad[t] *= system.spring_YP[t].item()**2/loss.item()
           system.spring_YN.grad[t] *= system.spring_YN[t].item()**2/loss.item()
           #system.dashpot_damping.grad[t] *= system.dashpot_damping[t].item()
           system.drag_damping.grad[t] *= system.drag_damping[t].item()**2/loss.item()

        # 记录数据
        losses.append(loss.item())
        spring_YPs.append(system.spring_YP[max_steps//2].item())       
        
        if iter % (max_iter//50) == 0:
            print(f'\nIter={iter}, Loss={loss.item()}')
            print(f'spring_YP={system.spring_YP[max_steps//2].item()}')
            print(f'spring_YN={system.spring_YN[max_steps//2].item()}')
            # print(f'dashpot_damping={system.dashpot_damping[max_steps//2].item()}')
            print(f'drag_damping={system.drag_damping[max_steps//2].item():.4e}')

        # 更新参数
        optimizer.step()
        
    pos_final = [] 
    for t in range(max_steps):
        #pos_final.append(system.x[max_steps-1,n,1,2].item())
        pos_final.append(system.spring_YP[t].item())

    # 绘图
    fig, axs = plt.subplots(3)
    axs[0].plot(losses)
    axs[1].plot(spring_YPs)
    axs[2].plot(pos_final)
    plt.show()
    #output_spring_para(system)

if __name__ == '__main__':
    main()