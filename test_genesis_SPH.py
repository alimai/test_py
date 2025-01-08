import genesis as gs

########################## 初始化 ##########################
gs.init(backend=gs.cpu)

########################## 创建场景 ##########################

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt       = 4e-3,
        substeps = 10,
    ),
    sph_options=gs.options.SPHOptions(
        lower_bound   = (-0.5, -0.5, 0.0),
        upper_bound   = (0.5, 0.5, 1),
        particle_size = 0.01,
    ),
    vis_options=gs.options.VisOptions(
        visualize_sph_boundary = True,
    ),
    show_viewer = True,
)

########################## 实体 ##########################
plane = scene.add_entity(
    morph=gs.morphs.Plane(),
)

liquid = scene.add_entity(
    # 粘性液体
    # material=gs.materials.SPH.Liquid(mu=0.02, gamma=0.02),
    material=gs.materials.SPH.Liquid(),
    morph=gs.morphs.Box(
        pos  = (0.0, 0.0, 0.65),
        size = (0.4, 0.4, 0.4),
    ),
    surface=gs.surfaces.Default(
        color    = (0.4, 0.8, 1.0),
        vis_mode = 'particle',
    ),
)

########################## 构建 ##########################
scene.build()

horizon = 100
for i in range(horizon):
    scene.step()

# 获取粒子位置
particles = liquid.get_particles()