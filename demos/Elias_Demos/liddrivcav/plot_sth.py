import math
import os
import numpy as np
from phi.jax.flow import *

def draw_benchm_comp(names, mode=['x','y']):

    if mode[0] == 'x':
        paper_positions = [1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5, 0.4531, 0.2813, 0.1719, 0.1016,
                       0.0703, 0.0625, 0.0547, 0]
        paper_values = [-1.0000000, -0.6644227, -0.5808359, -0.5169277, -0.4723329, -0.3372212, -0.1886747, -0.0570178,
                    0.0620561, 0.1081999, 0.2803696, 0.3885691, 0.3004561, 0.2228955, 0.2023300, 0.1812881, 0]
        paper_vals = math.vec(y=tensor(paper_positions, spatial('y')), value=tensor(paper_values, spatial('y')))
    elif mode[0] == 'y':
        paper_positions = [0.0000, 0.0312, 0.0391, 0.0469, 0.0547, 0.0937, 0.1406, 0.1953, 0.5000, 0.7656, 0.7734, 0.8437, 0.9062,
                       0.9219, 0.9297, 0.9375, 1]
        paper_values = [0, -0.2279225, -0.2936869, -0.3553213, -0.4103754, -0.5264392, -0.4264545, -0.3202137,
                    0.0257995, 0.3253592, 0.3339924, 0.3769189, 0.3330442, 0.3099097, 0.2962703, 0.2807056, 0]
        paper_vals = math.vec(x=tensor(paper_positions, spatial('x')), value=tensor(paper_values, spatial('x')))

    sim_vals = []
    status = []
    for name in names:
        data = np.load(f"data/{name}/data.npz")
        t_num = data['t_num'].item()
        mssd = data['max_steady_state_diff'].item()
        freq = data['freq']
        dt = data['dt']

        nan_encountered = math.is_nan(mssd)
        vel_data = field.read(f"data/{name}/vel_{t_num + (-1 * freq if nan_encountered else 0)}.npz")
        if nan_encountered:
            vel_data_ = field.read(f"data/{name}/vel_{max(0, t_num-2*freq)}.npz")
            mssd = math.max(math.abs(vel_data.values - vel_data_.values)) / (dt * math.max(math.abs(vel_data_.values)))
            mssd = "circa: " + str(mssd.numpy()/freq)
        status.append(f'conv.: {not nan_encountered}  mssd: {mssd}')

        vel_data = field.read(f"data/{name}/vel_{t_num + (-1 * freq if nan_encountered else 0)}.npz")
        sim_val = vel_data.__getattr__(mode[0])[int(vel_data.__getattr__(mode[0]).size / 2)].vector[mode[0]]
        sim_val = vec(**{mode[1]: sim_val.points.vector[mode[1]]}, value=sim_val.vector[mode[0]].values)
        sim_vals.append(sim_val)

    hor_ver = ['horizontal', 'vertical'] if mode[0] == 'x' else ['vertical', 'horizontal']
    vis.plot(tensor([paper_vals, *sim_vals], channel(res=["paper", *[n + " " + m for n, m in zip(names, status)]])), title=f'{hor_ver[0]} velocity through the {hor_ver[1]} centerline')
    vis.show()
    print('done')

def plot_div(names, mode=math.max):
    sim_vals = []
    status = []
    for name in names:
        data = np.load(f"data/{name}/data.npz")
        t_num = data['t_num'].item()
        mssd = data['max_steady_state_diff'].item()
        freq = data['freq']
        freq = 10000
        dt = data['dt']

        nan_encountered = math.is_nan(mssd)
        vel_data = [field.read(f"data/{name}/vel_{t}.npz") for t in range(0, t_num+1, freq)]

        if 'low' in name: ordn = 2
        if 'mid' in name: ordn = 4
        if 'high' in name: ordn = 6
        vel_data = [mode(math.abs(field.divergence(vel, order=ordn).values)) for vel in vel_data]
        sim_val = vec(timestep=tensor(np.arange(0, t_num+1, freq), spatial('timestep')), divergence=tensor(vel_data, spatial('timestep')))
        sim_vals.append(sim_val)
    vis.plot(tensor(sim_vals, channel(res=["paper", *[n + " " + m for n, m in zip(names, status)]])),
             title=f'{mode.__name__} divergence over the timesteps')
    vis.show()
    print('done')


# for mode in [['x', 'y'], ['y', 'x']]:
for mode in [['y', 'x']]:
    for ord in ['low', 'mid']:
        draw_benchm_comp([f"phi3.0_final_{ord}_{res}" for res in [31, 61, 121]] + ["phi3.0_final_high_31"], mode)


# # for mode in [['x', 'y'], ['y', 'x']]:
# for mode in [['y', 'x']]:
#     draw_benchm_comp(["new_try_low_31", "new_try_low_61", "new_try_low_121"], mode)
#     draw_benchm_comp(["new_try_mid_31", "new_try_mid_61", "new_try_mid_121"], mode)
#     draw_benchm_comp(["new_try_high_31", "new_try_high_61", "new_try_high_121"], mode)
#
#     draw_benchm_comp(["new_try_low_31", "new_try_mid_31", "new_try_high_31"], mode)
#     draw_benchm_comp(["new_try_low_61", "new_try_mid_61", "new_try_high_61"], mode)
#     draw_benchm_comp(["new_try_low_121", "new_try_mid_121", "new_try_high_121"], mode)

# for mode in [['y', 'x']]:
#     # draw_benchm_comp(["new_try_low_31", "new_try_low_61", "new_try_low_121"], mode)
#     # draw_benchm_comp(["new_try_mid_31", "new_try_mid_61", "new_try_mid_121"], mode)
#     # draw_benchm_comp(["new_try_high_31", "new_try_high_61", "new_try_high_121"], mode)
#
#     draw_benchm_comp([f"timestep_invest_low_121_dt_{dt}" for dt in [0.003, 0.001, 0.0003, 0.0001]], mode)
#     draw_benchm_comp([f"timestep_invest_high_121_dt_{dt}" for dt in [0.003, 0.001, 0.0003, 0.0001]], mode)

# for mode in [math.max, math.mean]:
#     plot_div(["new_try_low_31", "new_try_low_61", "new_try_low_31"], mode)
#     plot_div(["new_try_mid_31", "new_try_mid_61", "new_try_mid_121"], mode)
#     plot_div(["new_try_high_31", "new_try_mid_61"], mode)

# plot_div(["new_try_low_121", "new_try_mid_121", "new_try_high_121"], math.max)

print("done")