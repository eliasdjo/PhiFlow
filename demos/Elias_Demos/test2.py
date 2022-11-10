from phi.jax.flow import *
from functools import partial
visc = 0.1



xy_num = 30
visc = 0.1

def tgv_F(vis, t):
    return math.exp(-2 * vis * t)
def tgv_velocity(x, vis=0, t=0):
    sin, cos = math.sin(x), math.cos(x)
    return math.stack([cos.vector['x'] * sin.vector['y'] * tgv_F(vis, t),
                       - sin.vector['x'] * cos.vector['y'] * tgv_F(vis, t)], dim=channel('vector'))
def tgv_pressure(x, vis=0, t=0):
    return -1 / 4 * (math.sum(math.cos(2 * x), 'vector')) * (tgv_F(vis, t) ** 2)

DOMAIN = dict(extrapolation=extrapolation.PERIODIC,
              bounds=Box['x,y,z', 0:2 * math.pi, 0:2 * math.pi, 0:2 * math.pi], scheme=False, x=xy_num, y=xy_num, z=xy_num)
v = CenteredGrid(partial(tgv_pressure, vis=visc, t=0), **DOMAIN)

mask_tensor = math.zeros(v.values.shape) + math.scatter(math.zeros(v.values.shape.only('x')), tensor([0], instance('points')), tensor([1], instance('points')))
mask_tensor = math.where(mask_tensor, 0, 1)
mask = v.with_values(mask_tensor.flip('x'))
plot(mask)
show()
print("done") #TODO ED pytest upgrade
