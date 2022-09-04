import phi.field
from phi.jax.flow import *
from functools import partial


visc = 0.1
xy_num = 30

def tgv_F(vis, t):
    return math.exp(-2 * vis * t)


def tgv_velocity(x, vis=0, t=0):
    sin, cos = math.sin(x), math.cos(x)
    return math.stack([cos.vector['x'] * sin.vector['y'] * tgv_F(vis, t),
                       - sin.vector['x'] * cos.vector['y'] * tgv_F(vis, t)], dim=channel('vector'))

def tgv_velocity2(x, vis=0, t=0):
    sin, cos = math.sin(x), math.cos(x)
    return math.stack([cos.vector['x'] * sin.vector['y'] * i for i in range(200)], dim=channel('vector'))


def tgv_pressure(x, vis=0, t=0):
    return -1 / 4 * (math.sum(math.cos(2 * x), 'vector')) * (tgv_F(vis, t) ** 2)


DOMAIN = dict(extrapolation=extrapolation.PERIODIC,
              bounds=Box['x,y', 0:2 * math.pi, 0:2 * math.pi], scheme=False, x=xy_num, y=xy_num)

DOMAIN2 = dict(extrapolation=extrapolation.PERIODIC, scheme=False, x=xy_num, y=xy_num)


def anal_sol(t):
    return StaggeredGrid(partial(tgv_velocity, vis=visc, t=t), **DOMAIN)


pressure = CenteredGrid(partial(tgv_pressure, vis=visc, t=0), **DOMAIN)
velocity = StaggeredGrid(partial(tgv_velocity, vis=visc, t=0), **DOMAIN)
velocity2 = CenteredGrid(partial(tgv_velocity2, vis=visc, t=0), **DOMAIN)

print("done")

