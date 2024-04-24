import math
import os
import numpy as np
from phi.jax.flow import *

def draw_benchm_comp(names):
    y_positions = [1, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5, 0.4531, 0.2813, 0.1719, 0.1016,
                   0.0703, 0.0625, 0.0547, 0]
    y_values = [-1.0000000, -0.6644227, -0.5808359, -0.5169277, -0.4723329, -0.3372212, -0.1886747, -0.0570178,
                0.0620561, 0.1081999, 0.2803696, 0.3885691, 0.3004561, 0.2228955, 0.2023300, 0.1812881, 0]
    paper_vals = math.vec(y=tensor(y_positions, spatial('y')), value=tensor(y_values, spatial('y')))

    sim_vals = []
    for name in names:
        data = np.load(f"data/{name}/data.npz")
        t_num = data['t_num'].item()
        vel_data = field.read(f"data/{name}/vel_{t_num}.npz")

        sim_val = vel_data.x[int(vel_data.x.size / 2) + 1].vector['x']
        sim_val = vec(y=sim_vals.points.vector['y'], value=sim_vals.vector['x'].values)
        sim_vals.append(sim_val)

    vis.plot(tensor([paper_vals, *sim_vals], channel(res='paper, simulation')), title=f'Horizontal velocity through the vertical centerline')
    vis.show()
    print('done')