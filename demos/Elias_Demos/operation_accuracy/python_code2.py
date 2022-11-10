from phi.jax.flow import *

def tgv_F(vis, t):
    return math.exp(-2 * vis * t)


def tgv_velocity(x, vis=0, t=0):
    sin, cos = math.sin(x), math.cos(x)
    return math.stack([cos.vector['x'] * sin.vector['y'] * tgv_F(vis, t),
                       cos.vector['x'] * 0], dim=channel('vector'))


DOMAIN = dict(bounds=Box['x,y', 0*math.pi : 2*math.pi, 0.25*math.pi : 1.75*math.pi], x=10, y=10, extrapolation=extrapolation.combine_sides(
    x=extrapolation.PERIODIC,
    y=extrapolation.combine_by_direction(extrapolation.REFLECT, extrapolation.SYMMETRIC)))

velocity = StaggeredGrid(tgv_velocity, **DOMAIN)
vis.plot(velocity.vector['x'], velocity.vector['y'], title=f'vel')
vis.show()

grad = field.spatial_gradient(velocity.vector[0], stack_dim=math.channel('vector'), scheme=Scheme(6, Solve('CG', 1e-5, 1e-5)))
vis.plot(grad.vector['x'], grad.vector['y'], title=f'grad')
vis.show()



DOMAIN = dict(bounds=Box['x,y', 0*math.pi : 2*math.pi, 0.25*math.pi : 1.75*math.pi], x=10, y=10, extrapolation=extrapolation.PERIODIC)
velocity = StaggeredGrid(tgv_velocity, **DOMAIN)

grad = field.spatial_gradient(velocity.vector[1], stack_dim=math.channel('vector'), scheme=Scheme(6, Solve('CG', 1e-5, 1e-5)))
vis.plot(grad.vector['x'], grad.vector['y'], title=f'grad p')
vis.show()

