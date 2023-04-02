import numpy as np
from phi.flow import *
import matplotlib.pyplot as plt

prefix = "operation_accuracy/data/"
names = ["laplacian", "laplacian_staggered", "diffusion", "diffusion_staggered",
         "gradient_fst_comp", "gradient_staggered_fst_comp", "gradient_snd_comp", "gradient_staggered_snd_comp",
         "advection", "advection_staggered"]

# prefix = "tgv_accuracy/data/"
# names = ["phi_flow_rel_err", "std_snd_ord_rel_err",
#          "high_ord_pt_jax_single_small_stencil_rel_err"]
# names = ["high_ord_pt_jax_single_small_stencil_rel_err"]

# prefix = "tgv_accuracy/errors/"
# names = ["high_order", "high_order_stagg"]

# prefix = "homogenic_isotropic_turbulence/data/"
# names = ["compare_p_solve_small_and_big_stencil",
#          "compare_p_solve_small_and_big_stencil_to_high_res_sr_4",
#          "compare_p_solve_small_and_big_stencil_to_high_res_sr_4_2"]
# names = ["big_stencil_against_high_res",
#          "small_stencil_against_high_res"]

# prefix = "homogenic_isotropic_turbulence/errors/"
# names = ["small_stencil_excerpt"]
# names = ["small_stencil_excerpt_"]


names = [prefix + name for name in names]

for name in names:
    file = np.load(name + ".npz")
    data = file['data']
    xy_nums = file['xy_nums']

    final_errors = [res[1][-1] for res in data]
    plt.plot(xy_nums, final_errors, label=f"name: {name}")

plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.legend()
plt.show()

print("done")
