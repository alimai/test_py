import os
import time
import taichi as ti
import numpy as np

bias_diagonal = np.sqrt(2)#1.0#
r_level0 = 0.75
r_level1 = r_level0 + 1.0
ti.init(arch=ti.cpu)#, cpu_max_num_threads=1)

block1 = ti.root.pointer(ti.ij, (8,8))
block2 = block1.pointer(ti.ij, (8,8))
block3 = block2.pointer(ti.ij, (8,8))
pixel = block3.bitmasked(ti.ij, (2,2))
#pixel = block2.dense(ti.ij, (2,2))

N_x = pixel.shape[0]
N_y = pixel.shape[1]

x1 = ti.field(ti.f32)
x2 = ti.field(ti.f32)
pixel.place(x1, x2)

@ti.kernel
def activate():
    x1[2,3] = 1.0
    x1[200,4000] = 2.0
    #ti.deactivate(block2, [0,0])

@ti.kernel
def print_active():
    #ti.activate(block2, [0,0])
    for i, j in block2:
        print("Active block", i, j)
    #for i, j in pixel:
    #    print('field x[{}, {}] = {}'.format(i, j, x1[i, j]))


@ti.kernel
def init_data():
    for i in range(pixel.shape[0]):
        for j in range(pixel.shape[1]):
            target_radius = 50
            dist_bias = ti.sqrt((i-500)**2+(j-500)**2)
            if(abs(dist_bias-target_radius) <= r_level1):
                if(abs(dist_bias-target_radius) <= r_level0):
                    x1[i, j] = 0.0
                elif(dist_bias-target_radius > r_level0):
                    x1[i, j] = 1.0
                else:
                    x1[i, j] = -1.0
                x2[i, j] = -(r_level1+0.1)

            dist_bias = ti.sqrt((i-300)**2+(j-300)**2)
            if(abs(dist_bias-target_radius) <= r_level1):
                if(abs(dist_bias-target_radius) <= r_level0):
                    x1[i, j] = 0.0
                elif(dist_bias-target_radius > r_level0):
                    x1[i, j] = 1.0
                else:
                    x1[i, j] = -1.0
                x2[i, j] = -(r_level1+0.1)
                    
            dist_bias = ti.sqrt((i-800)**2+(j-800)**2)
            if(abs(dist_bias-target_radius) <= r_level1):
                if(abs(dist_bias-target_radius) <= r_level0):
                    x1[i, j] = 0.0
                elif(dist_bias-target_radius > r_level0):
                    x1[i, j] = 1.0
                else:
                    x1[i, j] = -1.0
                x2[i, j] = -(r_level1+0.1)

            if (i == 600 and j <= 400) or (j == 400 and i <= 600):
                x1[i, j] = 0.0
                x2[i, j] = -(r_level1+0.1)
            if (i == 601 and j <= 401) or (j == 401 and i <= 601):
                x1[i, j] = 1.0
                x2[i, j] = -(r_level1+0.1)
            if (i == 599 and j <= 399) or (j == 399 and i <= 599):
                x1[i, j] = -1.0
                x2[i, j] = -(r_level1+0.1)

            # if (i == 400 and j >= 600) or (j == 600 and i >= 400):
            #     x1[i, j] = 0.0
            #     x2[i, j] = -(r_level1+0.1)
            # if (i == 401 and j >= 601) or (j == 601 and i >= 401):
            #     x1[i, j] = -1.0
            #     x2[i, j] = -(r_level1+0.1)
            # if (i == 399 and j >= 599) or (j == 599 and i >= 399):
            #     x1[i, j] = 1.0
            #     x2[i, j] = -(r_level1+0.1)

@ti.func
def update_neighbours_L0(i_nb, j_nb, value_center_x2, bias = 1.0):  
    if abs(x1[i_nb, j_nb]) > r_level0:
        if(abs(x2[i_nb, j_nb]) > r_level1):
            if(x1[i_nb, j_nb] < 0):
                x2[i_nb, j_nb] = value_center_x2 - bias
            else:
                x2[i_nb, j_nb] = value_center_x2 + bias
        else:
            bias_last = x2[i_nb, j_nb] - value_center_x2#有方向
            value_new = value_center_x2 + bias*bias_last/ti.sqrt(bias**2+bias_last**2)
            if abs(value_new) < abs(x2[i_nb, j_nb]):
                x2[i_nb, j_nb] = value_new

@ti.func
def update_neighbours_L1(i_nb, j_nb, i_center, j_center, bias = 1.0):
    value_center_x2 = x2[i_center, j_center]
    if not ti.is_active(pixel, [i_nb, j_nb]): 
        if x1[i_center, j_center] > 0:
            x2[i_nb, j_nb] = value_center_x2 + bias
            x1[i_nb, j_nb] = r_level1+0.1
        else:
            x2[i_nb, j_nb] = value_center_x2 - bias
            x1[i_nb, j_nb] = -r_level1-0.1
    elif abs(x2[i_nb, j_nb]) > r_level0:
        # value_new = value_center_x2 + bias
        # if x1[i_center, j_center] < 0:
        #     value_new = value_center_x2 - bias
        # if abs(value_new) < abs(x2[i_nb, j_nb]):
        #     x2[i_nb, j_nb] = value_new
        update_neighbours_L0(i_nb, j_nb, value_center_x2)

@ti.kernel
def process_core(rate: float, step: int):    

    for i, j in pixel:
        if abs(x1[i, j]) <= r_level0:
            x2[i, j] = x1[i, j] - rate            

    ti.sync()
    for i, j in pixel:
        if step % 2 == 0:
            if abs(x1[i, j]) <= r_level0 and i > 0:
                update_neighbours_L0(i-1, j, x2[i, j]) 
        else:
            if abs(x1[i, j]) <= r_level0 and j > 0:
                update_neighbours_L0(i, j-1, x2[i, j])
    ti.sync()
    for i, j in pixel:
        if step % 2 == 0:
            if abs(x1[i, j]) <= r_level0 and i < N_x-1:
                update_neighbours_L0(i+1, j, x2[i, j]) 
        else:
            if abs(x1[i, j]) <= r_level0 and j < N_y-1:
                update_neighbours_L0(i, j+1, x2[i, j])
    ti.sync()
    for i, j in pixel:
        if step % 2 == 0:
            if abs(x1[i, j]) <= r_level0 and j > 0:
                update_neighbours_L0(i, j-1, x2[i, j])
        else:
            if abs(x1[i, j]) <= r_level0 and i > 0:
                update_neighbours_L0(i-1, j, x2[i, j])
    ti.sync()
    for i, j in pixel:
        if step % 2 == 0:
            if abs(x1[i, j]) <= r_level0 and j < N_y-1:
                update_neighbours_L0(i, j+1, x2[i, j])
        else:
            if abs(x1[i, j]) <= r_level0 and i < N_x-1:
                update_neighbours_L0(i+1, j, x2[i, j])
  
    ti.sync()
    for i, j in ti.ndrange(N_x, N_y):
        ti.loop_config(serialize=True)  
        if ti.is_active(pixel, [i, j]):
            if (abs(x2[i, j]) <= r_level0) and (abs(x1[i, j]) > r_level0):  
                if i > 0:
                    update_neighbours_L1(i-1, j, i, j)         
                if i < N_x-1:
                    update_neighbours_L1(i+1, j, i, j)        
                if j > 0:
                    update_neighbours_L1(i, j-1, i, j)      
                if j < N_y-1:
                    update_neighbours_L1(i, j+1, i, j)

    ti.sync()
    for i, j in pixel:
        x1[i, j] = x2[i, j]
        x2[i, j] = -(r_level1+0.1)
        if abs(x1[i, j]) > r_level1:
            x1[i, j] = -(r_level1+0.1)
            ti.deactivate(pixel, [i,j])

@ti.kernel
def deactivate_unvalid_block():   
    for m,n in ti.ndrange(block3.shape[0], block3.shape[1]):
        if ti.is_active(block3, [m, n]):
            status_block = False
            for i_local, j_local in ti.ndrange(2, 2):  # pixel 层级的局部坐标
                i_global = m * 2 + i_local  # 将局部坐标转换为全局坐标
                j_global = n * 2 + j_local
                if ti.is_active(pixel, [i_global, j_global]):
                    status_block = True
            if not status_block:
                ti.deactivate(block3, [m, n])            
    ti.sync() 
    for m,n in ti.ndrange(block2.shape[0], block2.shape[1]):
        if ti.is_active(block2, [m, n]):
            status_block = False
            for i_local, j_local in ti.ndrange(8,8):  # block3 层级的局部坐标
                i_global = m * 8 + i_local  # 将局部坐标转换为全局坐标
                j_global = n * 8 + j_local
                if ti.is_active(block3, [i_global, j_global]):
                    status_block = True
                    break
            if not status_block:
                ti.deactivate(block2, [m, n])
    ti.sync()
    for m,n in ti.ndrange(block1.shape[0], block1.shape[1]):
        if ti.is_active(block1, [m, n]):
            status_block = False
            for i_local, j_local in ti.ndrange(8,8):  # block2 层级的局部坐标
                i_global = m * 8 + i_local  # 将局部坐标转换为全局坐标
                j_global = n * 8 + j_local
                if ti.is_active(block2, [i_global, j_global]):
                    status_block = True
                    break
            if not status_block:
                ti.deactivate(block1, [m, n])

init_data()    
#activate()
#print_active()

gui = ti.GUI("Sparse Field", res=(N_x, N_y))

step = 0
start_time = time.time()
while gui.running:#step < 1000:#
    if (step % 20 == 0):#True:#
        gui.set_image(x1.to_numpy())
        gui.show()
        deactivate_unvalid_block()
    if step<2e5:
        process_core(0.3, step)
    step += 1
    #print(step)
end_time = time.time()
print("Time cost: ", end_time-start_time)
