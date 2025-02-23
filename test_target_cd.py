import torch
import numpy as np
import matplotlib.pyplot as plt

# 设备配置
device = torch.device('cpu')

# 基本参数
dt = 1e-4  # 时间步长
learning_rate = 1e-1  # 学习率
alpha = 1e-3  # 学习率衰减

# 物理参数
spring_YP_base = 1e6/(1e6+1)#1e6  
spring_YN_base = 3e3/(3e3+1)#3e3  
dashpot_damping_base = 1e1/(1e1+1)#1e1  
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
         # 创建参数张量，使用nn.Parameter包装以确保正确的梯度计算
        self.spring_YP = torch.nn.Parameter(torch.full((max_steps,), spring_YP_base, device=device))
        self.spring_YN = torch.nn.Parameter(torch.full((max_steps,), spring_YN_base, device=device))
        self.dashpot_damping = torch.nn.Parameter(torch.full((max_steps,), dashpot_damping_base, device=device))
        self.drag_damping = torch.nn.Parameter(torch.full((max_steps,), drag_damping_base, device=device))
        
        
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
        for i in range(-2, 3):
            if i != 0 and abs(i) <= 2:
                offsets.append(torch.tensor([i, 0], device=device))
        return offsets
    
    def initialize_mass_points(self):
        size_x = ellipse_short * 2
        quad_size = size_x / (n_x + 1)
        size_y = n_y * quad_size
        
        # 创建网格点
        i_coords = torch.arange(n_x, device=device)
        j_coords = torch.arange(n_y, device=device)
        i, j = torch.meshgrid(i_coords, j_coords, indexing='ij')
        
        # 计算初始位置
        x = (i * quad_size - size_x * 0.5 + 0.5 * quad_size).unsqueeze(-1)
        y = (j * quad_size - size_y * 0.5 + 0.5 * quad_size).unsqueeze(-1)
        z = torch.zeros_like(x)
        
        # 添加随机偏移
        random_offset = (torch.rand(n_x, n_y, 3, device=device) - 0.5) * 0.03
        
        # 组合坐标
        pos = torch.cat([x, y, z], dim=-1)
        
        # 调整椭圆形状#操作每个元素(此处为3维坐标)第0/2个数据
        pos[..., 0] *= torch.abs(pos[..., 0]/ellipse_short).pow(0.6) * ellipse_short / (torch.abs(pos[..., 0])+1e-5)
        pos[..., 2] = ellipse_long * 1.01 * (1-(pos[..., 0]/(ellipse_short*1.01)).pow(2)).sqrt()
        
        # 添加随机偏移(除了边界点)
        mask = torch.ones_like(pos)
        #操作第0/-1行数据
        mask[0, :] = mask[-1, :] = 0
        #操作第0/-1列数据：mask[:, 0] = mask[:, -1] = 0
        pos += random_offset * mask

        # 重新初始化状态张量
        self.x = torch.zeros((max_steps, n_x, n_y, 3), device=device)
        self.v = torch.zeros((max_steps, n_x, n_y, 3), device=device)
        self.f = torch.zeros((max_steps, n_x, n_y, 3), device=device)    
        self.x[0] = pos.clone()
        #print(self.x[0])

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
            
            # 计算原始距离
            r_rolled = torch.roll(self.r, shifts=i_shift, dims=0)  # 先在 i 方向滚动
            r_rolled = torch.roll(r_rolled, shifts=j_shift, dims=1)  # 再在 j 方向滚动
            original_dist = (torch.norm(offset.float()) * (self.r + r_rolled) * 0.5).unsqueeze(-1)
        
            # 计算弹力
            stretch_mask = dist > original_dist
            compress_mask = ~stretch_mask
            
            forces += torch.where(stretch_mask,
                                -abs(self.spring_YP[t-1])/(1-abs(self.spring_YP[t-1])) * direction * (dist - original_dist),
                                abs(self.spring_YN[t-1])/(1-abs(self.spring_YN[t-1])) * direction * (original_dist - dist))
        
        # 添加重力
        forces[..., 2] += -9.8
        
        return forces
    
    def step(self, t):
        # 计算力
        self.f[t] = self.compute_forces(t)
        
        # 更新速度
        mask = torch.ones_like(self.v[t])
        mask[0, :] = mask[-1, :] = 0
        
        self.v[t] = mask * (self.v[t-1] + self.f[t] * dt) / (1.0 + drag_damping_base)

        # 更新位置
        self.x[t] = self.x[t-1] + dt * self.v[t]

    def compute_loss(self):
        j = n_y // 2
        loss = torch.tensor(0.0, device=device)
        
        for t in range(max_steps):
            dists = torch.norm(self.x[t, 1:, j] - self.x[t, :-1, j], dim=1)
            target_dists = self.r[1:, j] + self.r[:-1, j]
            rel_dists = dists - target_dists
            
            avg_dist = torch.mean(rel_dists)
            step_loss = torch.sum(torch.abs(rel_dists - avg_dist))
            loss += step_loss * (t ** 2) * 1e1
            
        return loss
    
def main():
    system = MassSpringSystem()
    optimizer = torch.optim.SGD([
        system.spring_YP,
        system.spring_YN,
        system.dashpot_damping,
        system.drag_damping,
    ], lr=learning_rate)
    
    losses = []
    spring_YPs = []
    
    max_iter = 10# 最大迭代次数 
    for iter in range(max_iter):
        optimizer.zero_grad()        
        system.initialize_mass_points()

        # 前向传播
        #with torch.set_grad_enabled(True):
        for t in range(1, max_steps):
            system.step(t)
        
        # 计算损失并反向传播
        loss = system.compute_loss()
        loss.backward(retain_graph=True)
        # for t in range(0, max_steps):
        #    system.spring_YP.grad[t] *= system.spring_YP[t].item()#**2/loss.item()
        #    system.spring_YN.grad[t] *= system.spring_YN[t].item()#**2/loss.item()
        #    #system.dashpot_damping.grad[t] *= system.dashpot_damping[t].item()
        #    #system.drag_damping.grad[t] *= system.drag_damping[t].item()

        # 记录数据
        losses.append(loss.item())
        spring_YPs.append(system.spring_YP[max_steps//2].item())       
        
        if iter % (max_iter//10) == 0:
            print(f'\nIter={iter}, Loss={loss.item()}')
            print(f'spring_YP={system.spring_YP[max_steps//2].item():.4e}')
            print(f'spring_YN={system.spring_YN[max_steps//2].item():.4e}')
            # print(f'dashpot_damping={system.dashpot_damping[max_steps//2].item():.4e}')
            # print(f'drag_damping={system.drag_damping[max_steps//2].item():.4e}')

        # 更新参数
        optimizer.step()
        

    pos_final = [] 
    for n in range(0, 15):
        pos_final.append(system.x[max_steps-1][n][1][2].item())

    # 绘图
    fig, axs = plt.subplots(3)
    axs[0].plot(losses)
    axs[1].plot(spring_YPs)
    axs[2].plot(pos_final)
    plt.show()

if __name__ == '__main__':
    main()