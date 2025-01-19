
import time
import taichi as ti
import numpy as np

bias_diagonal = np.sqrt(2)#1.0#
r_level0 = 0.75
r_level1 = r_level0 + 1.1#+1.1>1.0防止数值误差
ti.init(arch=ti.gpu)#, cpu_max_num_threads=1)

size_root_element = 8
size_block1_element = 8
size_block2_element = 8
size_block3_element = 2
block1 = ti.root.pointer(ti.ij, (size_root_element,size_root_element))
block2 = block1.pointer(ti.ij, (size_block1_element,size_block1_element))
block3 = block2.pointer(ti.ij, (size_block2_element,size_block2_element))
pixel = block3.bitmasked(ti.ij, (size_block3_element,size_block3_element))
#pixel = block2.dense(ti.ij, (2,2))

N_x = pixel.shape[0]
N_y = pixel.shape[1]
print(N_x,N_y)

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
def init_data2():
    for i, j in ti.ndrange(N_x, N_y):
        if i > 100 and i < 900 and j > 100 and j< 900:
            if (i == 400 and j <= 400) or  (j == 400 and i <= 400):
                x1[i, j] = 0.0
                x2[i, j] = -(r_level1+0.1)
            if (i == 401 and j <= 401) or (j == 401 and i <= 401):
                x1[i, j] = 1.0
                x2[i, j] = -(r_level1+0.1)
            if (i == 399 and j <= 399) or (j == 399 and i <= 399):
                x1[i, j] = -1.0
                x2[i, j] = -(r_level1+0.1)

            if (i == 800 and j <= 800) or (j == 800 and i <= 800):
                x1[i, j] = 0.0
                x2[i, j] = -(r_level1+0.1)
            if  (i == 799 and j <= 799) or (j == 799 and i <= 799):
                x1[i, j] = 1.0
                x2[i, j] = -(r_level1+0.1)
            if (i == 801 and j <= 801) or (j == 801 and i <= 801):
                x1[i, j] = -1.0
                x2[i, j] = -(r_level1+0.1)

@ti.kernel
def init_data():
    #下边两行在gpu模式下会导致结果出错？？？
    # for i in range(pixel.shape[0]):
    #     for j in range(pixel.shape[1]):
    for i, j in ti.ndrange(N_x, N_y):
            target_radius = 50
            target_center = 500
            dist_bias = ti.sqrt((i-target_center)**2+(j-target_center)**2)
            if(abs(dist_bias-target_radius) <= r_level1):
                if(abs(dist_bias-target_radius) <= r_level0):
                    x1[i, j] = 0.0
                elif(dist_bias-target_radius > r_level0):
                    x1[i, j] = 1.0
                else:
                    x1[i, j] = -1.0
                x2[i, j] = -(r_level1+0.1)

            target_center = 300  
            dist_bias = ti.sqrt((i-target_center)**2+(j-target_center)**2)
            if(abs(dist_bias-target_radius) <= r_level1):
                if(abs(dist_bias-target_radius) <= r_level0):
                    x1[i, j] = 0.0
                elif(dist_bias-target_radius > r_level0):
                    x1[i, j] = 1.0
                else:
                    x1[i, j] = -1.0
                x2[i, j] = -(r_level1+0.1)

            target_center = 800
            dist_bias = ti.sqrt((i-target_center)**2+(j-target_center)**2)
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
        update_neighbours_L0(i_nb, j_nb, value_center_x2)
        # value_new = value_center_x2 + bias
        # if x1[i_center, j_center] < 0:
        #     value_new = value_center_x2 - bias
        # if abs(value_new) < abs(x2[i_nb, j_nb]):
        #     x2[i_nb, j_nb] = value_new

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
    #下述三行在gpu模式下可能出错（因为有新的active？）
    # for i, j in ti.ndrange(N_x, N_y):
    #    ti.loop_config(serialize=True)  
    #    if ti.is_active(pixel, [i, j]):
    for i, j in pixel:
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
            ti.deactivate(pixel, [i,j])

@ti.kernel
def deactivate_unvalid_block():   
    for m,n in ti.ndrange(block3.shape[0], block3.shape[1]):
        if ti.is_active(block3, [m, n]):
            status_block = False
            for i_local, j_local in ti.ndrange(size_block3_element, size_block3_element):  # pixel 层级的局部坐标
                i_global = m * size_block3_element + i_local  # 将局部坐标转换为全局坐标
                j_global = n * size_block3_element + j_local
                if ti.is_active(pixel, [i_global, j_global]):
                    status_block = True
            if not status_block:
                ti.deactivate(block3, [m, n])            
    ti.sync() 
    for m,n in ti.ndrange(block2.shape[0], block2.shape[1]):
        if ti.is_active(block2, [m, n]):
            status_block = False
            for i_local, j_local in ti.ndrange(size_block2_element,size_block2_element):  # block3 层级的局部坐标
                i_global = m * size_block2_element + i_local  # 将局部坐标转换为全局坐标
                j_global = n * size_block2_element + j_local
                if ti.is_active(block3, [i_global, j_global]):
                    status_block = True
                    break
            if not status_block:
                ti.deactivate(block2, [m, n])
    ti.sync()
    for m,n in ti.ndrange(block1.shape[0], block1.shape[1]):
        if ti.is_active(block1, [m, n]):
            status_block = False
            for i_local, j_local in ti.ndrange(size_block1_element,size_block1_element):  # block2 层级的局部坐标
                i_global = m * size_block1_element + i_local  # 将局部坐标转换为全局坐标
                j_global = n * size_block1_element + j_local
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
        #input("input:")
    process_core(0.3, step)
    step += 1
end_time = time.time()
ti.reset()
print("Time cost: ", end_time-start_time)
