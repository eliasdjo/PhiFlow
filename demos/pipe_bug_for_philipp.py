""" Streamline Profile
Simulates a viscous fluid flowing through a horizontal pipe.
"""
from phi.jax.flow import *

DOMAIN = dict(bounds=Box['x,y', 0:0.5, 0:0.5], x=20, y=20, extrapolation=extrapolation.combine_sides(
                x=extrapolation.PERIODIC,
                y=extrapolation.combine_by_direction(extrapolation.ANTIREFLECT, extrapolation.ANTISYMMETRIC)))

velocity = StaggeredGrid(0, **DOMAIN)
velocity, pressure = fluid.make_incompressible(velocity, scheme=Scheme(4), solve=math.Solve('CG', 1e-5, 1e-5))
velocity, pressure = fluid.make_incompressible(velocity, scheme=Scheme(4), solve=math.Solve('CG', 1e-5, 1e-5))
velocity, pressure = fluid.make_incompressible(velocity, scheme=Scheme(4), solve=math.Solve('CG', 1e-5, 1e-5))

# Done