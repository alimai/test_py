import taichi as ti
ti.init(arch=ti.cpu)

block0 = ti.root.pointer(ti.ij, (8,8))
block1 = block0.pointer(ti.ij, (8,8))
block2 = block1.pointer(ti.ij, (8,8))
block3 = block2.pointer(ti.ij, (8,8))
pixel = block3.bitmasked(ti.ij, (8,8))
#pixel = block2.dense(ti.ij, (2,2))
#https://github.com/alimai/test_py.git

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
    for i, j in pixel:
        print('field x[{}, {}] = {}'.format(i, j, x1[i, j]))

activate()
print_active()