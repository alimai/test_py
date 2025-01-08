# python/taichi/examples/simulation/fractal.py

import taichi as ti

ti.init(arch=ti.cpu)

n = 100
pixels = ti.field(dtype=float, shape=(n, n, n))


@ti.func
def complex_sqr(z):
    return ti.Vector([z[0]**2 - z[1]**2, z[1] * z[0] * 2])


@ti.kernel
def paint(t: float):
    for i, j, k in pixels:  # Parallelized over all pixels
        c = ti.Vector([-0.8, ti.cos(t) * 0.2])
        z = ti.Vector([i / n - 1, j / n - 0.5]) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
        pixels[i, j, k] = 1 - iterations * 0.02


gui = ti.GUI("Julia Set", res=(n , n), background_color=10)

for i in range(1000000):
    print(i)
    paint(i * 0.03)
  
    # 提取 2D 切片（例如 z=32）
    slice_z = 32
    field_np = pixels.to_numpy()  
    slice_2d = field_np[:,:, slice_z]

    gui.set_image(slice_2d)
    gui.show()