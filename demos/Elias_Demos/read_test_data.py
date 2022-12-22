import numpy as np
from phi.flow import *
import matplotlib.pyplot as plt

# prefix = "operation_accuracy/data_old3/"
# names = [
#     "gradient_snd_comp", "gradient_snd_comp",
#     "gradient_fst_comp", "gradient_fst_comp",
#     "laplacian", "laplacian_staggered",
#     "divergence", "divergence_staggered",
#     "diffusion", "diffusion_staggered",
#     "advection", "advection_staggered"
# ]

#
prefix = "tgv_accuracy/errors/"
# names = [
#     "phi_flow", "phi_flow_stagg",
#     "low_order", "low_order_stagg",
#     "mid_order", "mid_order_stagg",
#     "high_order", "high_order_stagg"
#          ]

names = ["high_order_stagg"]

# prefix = "homogenic_isotropic_turbulence/data/"
# names = ["compare_p_solve_small_and_big_stencil",
#          "compare_p_solve_small_and_big_stencil_to_high_res_sr_4",
#          "compare_p_solve_small_and_big_stencil_to_high_res_sr_4_2"]
# names = ["big_stencil_against_high_res",
#          "small_stencil_against_high_res"]


# prefix = "homogenic_isotropic_turbulence/errors/"
# names = ["compare_p_solve_small_and_big_stencil",
#          "compare_p_solve_small_and_big_stencil_to_high_res_sr_4",
#          "compare_p_solve_small_and_big_stencil_to_high_res_sr_4_2"]
# names = ["small_stencil_excerpt"]

names = [prefix + name for name in names]

for name in names:
    file = np.load(name + ".npz")
    data = file['data']
    xy_nums = file['xy_nums']

    plt.clf()

    for i, lines in enumerate(data):
        plt.plot(lines[0], lines[1], label=f"xy_nums: {xy_nums[i]}")

    plt.yscale('log')

    plt.title(name)

    plt.legend()
    plt.show()

print("done")
