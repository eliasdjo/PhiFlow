""" Streamline Profile
Simulates a viscous fluid flowing through a horizontal pipe.
"""
from phi.flow import *

DT = 1.
INFLOW_BC = extrapolation.combine_by_direction(normal=1, tangential=0)
velocity = StaggeredGrid(0, extrapolation.combine_sides(x=(extrapolation.PERIODIC, extrapolation.PERIODIC), y=0), x=50, y=32) # not working
# velocity = StaggeredGrid(0, extrapolation.combine_sides(x=(INFLOW_BC, extrapolation.BOUNDARY), y=0), x=50, y=32) # working
velocity = advect.semi_lagrangian(velocity, velocity, DT)