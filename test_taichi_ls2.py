import os
import taichi as ti
import numpy as np

bias_diagonal = 1.0#np.sqrt(2)#
r_level0 = 0.75
r_level1 = r_level0+1.0
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
            #dist_bias = ti.sqrt(i*i+j*j)-800
            #if(abs(dist_bias) < 1.5):
            if (i == 600 and j <= 400) or (j == 400 and i <= 600):
                x1[i, j] = 0.0
            if (i == 601 and j <= 401) or (j == 401 and i <= 601):
                x1[i, j] = 1.0
            if (i == 599 and j <= 399) or (j == 399 and i <= 599):
                x1[i, j] = -1.0
@ti.func
def update_neighbours_core(x2_nb, x1_nb, x2_center, bias = 1.0):
    if(x1_nb < 0):
        value_new = x2_center - bias
        if x2_nb < value_new:
            x2_nb = value_new
    else:
        value_new = x2_center + bias
        if x2_nb > value_new:
            x2_nb = value_new
    return x2_nb

@ti.func
def update_neighbours_core3(i_nb, j_nb, i_center, j_center, bias = 1.0):
    if not ti.is_active(pixel, [i_nb, j_nb]): 
        if x1[i_center, j_center] > 0:
            x2[i_nb, j_nb] = x2[i_center, j_center] + bias
            x1[i_nb, j_nb] = r_level1+0.1#激活时需要赋一个有意义的值
        else:
            x2[i_nb, j_nb] = x2[i_center, j_center] - bias
            x1[i_nb, j_nb] = -r_level1-0.1#激活时需要赋一个有意义的值
    elif abs(x1[i_nb, j_nb]) > r_level0:# and abs(x2[i_nb, j_nb]) > r_level0:
        if(x1[i_nb, j_nb] < 0):
            value_new = x2[i_center, j_center] - bias
            if(abs(x2[i_nb, j_nb]) > r_level1):
                x2[i_nb, j_nb] = value_new
            else:
                x2[i_nb, j_nb] = max(x2[i_nb, j_nb], value_new)
        else:
            value_new = x2[i_center, j_center] + bias
            if(abs(x2[i_nb, j_nb]) > r_level1):
                x2[i_nb, j_nb] = value_new
            else:
                bias_last = x2[i_nb, j_nb] - x2[i_center, j_center]
                value_new = x2[i_center, j_center] + bias*bias_last/ti.sqrt(bias**2+bias_last**2)
                x2[i_nb, j_nb] = min(x2[i_nb, j_nb], value_new)

@ti.func
def update_neighbours(i,j):
    if i > 0:
        # if not ti.is_active(pixel, [i-1, j]): 
        #     if x1[i, j] > 0:
        #         x2[i-1, j] = x2[i, j] + 1.0
        #     else:
        #         x2[i-1, j] = x2[i, j] - 1.0
        # elif abs(x1[i-1, j]) > r_level:
        #     x2[i-1, j] = update_neighbours_core(x2[i-1, j], x1[i-1, j], x2[i, j])
        update_neighbours_core3(i-1, j, i, j)
    if i < N_x-1:
        update_neighbours_core3(i+1, j, i, j)
    if j > 0:
        update_neighbours_core3(i, j-1, i, j)
    if j < N_y-1:
        update_neighbours_core3(i, j+1, i, j)
    # if i>0 and j>0:
    #     update_neighbours_core3(i-1, j-1, i, j, bias_diagonal)
    # if i>0 and j<N_y-1:
    #     update_neighbours_core3(i-1, j+1, i, j, bias_diagonal)
    # if i<N_x-1 and j>0:
    #     update_neighbours_core3(i+1, j-1, i, j, bias_diagonal)
    # if i<N_x-1 and j<N_y-1:
    #     update_neighbours_core3(i+1, j+1, i, j, bias_diagonal)

@ti.kernel
def process_core(rate: float):
    for i, j in pixel:
        if abs(x1[i, j]) <= r_level0:
            x2[i, j] = x1[i, j] - rate    
    for i, j in pixel:
        if abs(x1[i, j]) <= r_level0 and i > 0:
            update_neighbours_core3(i-1, j, i, j)  
    for i, j in pixel:
        if abs(x1[i, j]) <= r_level0 and i < N_x-1:
            update_neighbours_core3(i+1, j, i, j)  
    for i, j in pixel:
        if abs(x1[i, j]) <= r_level0 and j > 0:
            update_neighbours_core3(i, j-1, i, j)
    for i, j in pixel:
        if abs(x1[i, j]) <= r_level0 and j < N_y-1:
            update_neighbours_core3(i, j+1, i, j)
    for i, j in pixel:
        if (abs(x2[i, j]) <= r_level0) and (abs(x1[i, j]) > r_level0):
            update_neighbours(i,j)
    for i, j in pixel:
        x1[i, j] = x2[i, j]
        x2[i, j] = -(r_level1+0.1)
        if abs(x1[i, j]) > r_level1:
            x1[i, j] = -(r_level1+0.1)
            ti.deactivate(pixel, [i,j])



init_data()    
#activate()
#print_active()

gui = ti.GUI("Sparse Field", res=(N_x, N_y))

step = 0
while gui.running:
    process_core(0.3)
    gui.set_image(x1.to_numpy())
    gui.show()
    step += 1
    #print(step)
