from phi.flow import *

def tgv_F(vis, t):
    return math.exp(-2 * vis * t)

def tgv_velocity(x, vis=0, t=0):
    sin, cos = math.sin(x), math.cos(x)
    return math.stack([cos.vector['x'] * sin.vector['y'] * tgv_F(vis, t),
                       - sin.vector['x'] * cos.vector['y'] * tgv_F(vis, t)], dim=channel(vector='x,y'))

def testf(x):
    return math.stack([x.vector['y'] * 10 + x.vector['x'], -(x.vector['y'] * 10 + x.vector['x'])], dim=channel(vector='x,y'))


DOMAIN = dict(x=50, y=32,
              extrapolation=extrapolation.combine_sides(x=extrapolation.combine_by_direction(extrapolation.REFLECT, extrapolation.SYMMETRIC),
                                                        y=extrapolation.PERIODIC))

extrapol = extrapolation.combine_sides(x=extrapolation.combine_by_direction(extrapolation.REFLECT, extrapolation.SYMMETRIC),
                                                        y=extrapolation.PERIODIC)
extrapol2 = extrapolation.combine_sides(x=extrapolation.SYMMETRIC, y=extrapolation.PERIODIC)
extrapol3 = extrapolation.combine_by_direction(extrapolation.SYMMETRIC, extrapolation.PERIODIC)
test_ext = extrapolation.REFLECT
gradient_extrapol = extrapolation.combine_sides(x=extrapolation.combine_by_direction(extrapolation.ANTIREFLECT, extrapolation.ANTISYMMETRIC),
                                                        y=extrapolation.PERIODIC)

DOMAIN = dict(x=10, y=10, extrapolation=extrapol, bounds=Box['x,y', 0:10, 0:10])
# DOMAIN = dict(x=25, y=25, extrapolation=extrapol, bounds=Box['x,y', 0:2 * math.pi, 0:2 * math.pi])

velocity = StaggeredGrid(testf, **DOMAIN)
pressure = velocity[0]

# plot(velocity.vector[0])
# show()
# plot(velocity.vector[1])
# show()

# padded = field.pad(velocity, {'x': (2, 2), 'y': (2, 2)})
#
# plot(padded.vector[0])
# show()
# plot(padded.vector[1])
# show()

# unstacked = unstack(velocity, math.channel('vector'))

# grad = field.spatial_gradient(velocity, gradient_extrapol, stack_dim=math.channel('gradient'), scheme=Scheme(6, Solve('CG', 1e-12, 1e-12)))
# plot(grad.vector[0].gradient[0])
# show()
# plot(grad.vector[0].gradient[1])
# show()
# plot(grad.vector[1].gradient[0])
# show()
# plot(grad.vector[1].gradient[1])
# show()

# lap = field.divergence(velocity, scheme=Scheme(6, Solve('CG', 1e-12, 1e-12)))

grad = field.spatial_gradient(pressure, gradient_extrapol, type=StaggeredGrid, stack_dim=math.channel('gradient'), scheme=Scheme(6, Solve('CG', 1e-12, 1e-12)))




# test = velocity.vector[0]
# pressure = CenteredGrid(0, **DOMAIN)

print('done')

