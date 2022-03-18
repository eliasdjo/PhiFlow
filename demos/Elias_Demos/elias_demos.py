""" Simulate Burgers' Equation
Simple advection-diffusion equation.
"""
# import matplotlib
# import numpy as np
# from phi.field import ConstantField, Grid, Field, spatial_gradient, spatial_gradient_laiz, \
#     laplace, laplace_laiz, solve_linear, jit_compile_linear
# from phi.physics._boundaries import Domain, STICKY as CLOSED, PERIODIC
# from phi.flow import *
#
# import phi
from phi.jax.flow import *
# matplotlib.use("TkAgg")

phi.verify()

def bell(x):
    sigma = 1
    my = 0
    bell_value = 1 / (sigma * math.sqrt(np.pi)) * math.exp(-1/2 * ((math.vec_abs(x, 'vector') - my) / sigma))
    zeros = math.zeros(bell_value.shape)
    return math.stack([bell_value, zeros], channel('vector'))


DOMAIN = physics._boundaries.Domain(x=20, y=20, boundaries=physics._boundaries.PERIODIC,
                bounds=Box[-6:6, -6:6])

centered_vel = DOMAIN.vector_grid(bell, type=StaggeredGrid)
velocity = DOMAIN.vector_grid(bell, type=StaggeredGrid)
velocity2 = DOMAIN.vector_grid(bell, type=StaggeredGrid)
xvel, yvel = velocity.vector[0], velocity.vector[1]
xvel2, yvel2 = velocity2.vector[0], velocity2.vector[1]

pressure = DOMAIN.scalar_grid(0)


# test = advect.std(velocity, velocity, 1)

# mk_inc = math.jit_compile(fluid.make_incompressible)
# mk_inc_kamp = math.jit_compile(fluid.make_incompressible_kamp)
# velocity, pressure = mk_inc(velocity)
# velocity2, pressure2 = mk_inc_kamp(velocity2)
# xvel, yvel = velocity.vector[0], velocity.vector[1]
# xvel2, yvel2 = velocity2.vector[0], velocity2.vector[1]


def timestep(velocity, pressure):
    velocity = advect.semi_lagrangian(velocity, velocity, 0.1)
    velocity = diffuse.explicit(velocity, 0.1, 0.1)
    velocity, pressure = fluid.make_incompressible(velocity)
    return velocity, pressure

def timestep_high_o(velocity, pressure):
    velocity = advect.laiz(velocity, velocity, 0.1)
    velocity = diffuse.laiz(velocity, 0.1, 0.1)
    velocity, pressure = fluid.make_incompressible_kamp(velocity)
    return velocity, pressure


timestep_jit = math.jit_compile(timestep)
timestep_high_o_jit = math.jit_compile(timestep_high_o)

for _ in view(play=False, framerate=100, namespace=globals()).range():
# for i in range(10000):
#     print(f"timestep: {i}")
    # velocity, pressure = timestep(velocity, pressure)
    velocity, pressure = timestep_jit(velocity, pressure)
    # velocity, pressure = timestep_high_o(velocity, pressure)
    velocity2, pressure2 = timestep_high_o_jit(velocity2, pressure2)

    # velocity = advect.semi_lagrangian(velocity, velocity, 0.1)
    # velocity2 = advect.laiz(velocity2, velocity2, 0.1)
    # velocity = diffuse.laiz(velocity, velocity, 0.01)
    # velocity2 = diffuse.explicit(velocity, velocity, 0.01)

    xvel, yvel = velocity.vector[0], velocity.vector[1]
    xvel2, yvel2 = velocity2.vector[0], velocity2.vector[1]

print("done")


