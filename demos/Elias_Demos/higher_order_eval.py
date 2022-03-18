import numpy as np
np.savez("test3", np.ones(5))

from phi.jax.flow import *
from functools import partial

math.set_global_precision(64)

def tgv_F(vis, t):
    return math.exp(-2 * vis * t)


def tgv_velocity(x, vis=0, t=0):
    sin, cos = math.sin(x), math.cos(x)
    return math.stack([cos.vector['x'] * sin.vector['y'] * tgv_F(vis, t),
                       - sin.vector['x'] * cos.vector['y'] * tgv_F(vis, t)], dim=channel('vector'))


def tgv_pressure(x, vis=0, t=0):
    return -1/4 * (math.sum(math.cos(2 * x), 'vector')) * (tgv_F(vis, t) ** 2)


# def rk_timestep(velocity, pressure, vis, dt):
#
#     def adv_diff_press(v, p, dt_scale):
#         _dt = dt + dt_scale
#         v = advect.semi_lagrangian(v, v, _dt)
#         v = diffuse.explicit(v, vis, _dt)
#         v -= field.spatial_gradient_kamp(p, type=StaggeredGrid)
#
#     def pressure_treatment(v, p, dt_scale):
#         _dt = dt + dt_scale
#         v, delta_p = fluid.make_incompressible(v)
#         p += delta_p / _dt
#         return v, p
#
#     v_1, p_1 = velocity, pressure
#
#     r_1 = adv_diff_press(v_1, p_1, )
#
#
#     return velocity, pressure


dt = 1 / 16 * 2 * math.pi / 264
vis = 0.1

tges = 1
xy_nums = [5, 10]
t_num = int(math.ceil(tges / dt))
gridtype = CenteredGrid


def std_timestep(velocity, pressure):

    def adv_diff_press(v, p):
        v = advect.semi_lagrangian(v, v, dt)
        v = diffuse.explicit(v, vis, dt)
        v -= field.spatial_gradient(p, type=gridtype) * dt
        return v

    def pressure_treatment(v, p):
        v, delta_p = fluid.make_incompressible(v)
        p += delta_p / dt
        return v, p

    v_old = adv_diff_press(velocity, pressure)
    v_new, p_new = pressure_treatment(v_old, pressure)

    return v_new, p_new


std_timestep_jit = math.jit_compile(std_timestep)

def anal_sol(t):
    return DOMAIN.vector_grid(partial(tgv_velocity, vis=vis, t=t), type=gridtype)

errors_by_res = []
for xy_num in xy_nums:

    print(f"xy_num: {xy_num}")

    DOMAIN = physics._boundaries.Domain(x=xy_num, y=xy_num, boundaries=physics._boundaries.PERIODIC,
                    bounds=Box[0:2*math.pi, 0:2*math.pi])

    pressure = DOMAIN.scalar_grid(partial(tgv_pressure, vis=vis, t=0))
    velocity = DOMAIN.vector_grid(partial(tgv_velocity, vis=vis, t=0), type=gridtype)
    xvel, yvel = velocity.vector[0], velocity.vector[1]


    errors = []
    # for _ in view(play=False, framerate=100, namespace=globals()).range():

    for i in range(t_num):
        if i % 100 == 0:
            print(f"timestep: {i}")
            t = tensor(i * dt)
            error = math.sum(math.abs(anal_sol(i * dt).values - velocity.values)) / xy_num ** 2
            error_point = math.stack([t, error], channel('vector'))

            errors.append(error_point)
        velocity, pressure = std_timestep_jit(velocity, pressure)
        xvel, yvel = velocity.vector[0], velocity.vector[1]


    errors_by_res.append(math.stack(errors, channel("t")))

    print()

np.savez("test", test=math.stack(errors_by_res, channel("res")).numpy(('res', 'vector', 't')))


print("done")
