from demos.Elias_Demos.fd_coefficient_calc3 import get_coefficients
from phi.jax.flow import *

def get_stencils(order, implicit_order=0, one_sided=False, left_border_one_sided=False, staggered=False,
                 output_boundary_valid=False, input_boundary_valid=False):
    extend = int(math.ceil((order - implicit_order) / 2))
    rhs_extend = int(math.ceil(implicit_order / 2))

    shifts = [*range(-extend, extend + 1)]
    rhs_shifts = [*range(-rhs_extend, rhs_extend + 1)] if implicit_order else []

    v_ns_b0, rhs_v_ns_b0 = [], []
    max_extend = max(extend, rhs_extend)
    if one_sided:
        for i in range(1, max_extend + 1):
            off = max(0, extend - max_extend + i)  # non defining boundary
            off_rhs = max(0, rhs_extend - max_extend + i)
            n_shifts = [*range(-extend + off, extend + 1 + off + off_rhs)]
            rhs_n_shifts = [*range(-rhs_extend + off_rhs, rhs_extend + 1)] if implicit_order else []

            if staggered:
                n_shifts = [n + 1 for n in n_shifts]
                coefficient_shifts = [n - 0.5 for n in n_shifts]
                if input_boundary_valid:
                    del coefficient_shifts[-1]
                    coefficient_shifts.insert(0, coefficient_shifts[0] - 0.5)
                    del n_shifts[-1]
                    n_shifts.insert(0, n_shifts[0] - 1)
                n_values, n_values_rhs = get_coefficients(coefficient_shifts, 1, rhs_n_shifts)
            else:
                coefficient_shifts = n_shifts.copy()
                if input_boundary_valid:
                    del coefficient_shifts[-1]
                    coefficient_shifts.insert(0, coefficient_shifts[0] - 0.5)
                    del n_shifts[-1]
                    n_shifts.insert(0, n_shifts[0] - 1)
                n_values, n_values_rhs = get_coefficients(coefficient_shifts, 1, rhs_n_shifts)

            if left_border_one_sided:
                n_values = [-v for v in reversed(n_values)]
                if staggered:
                    n_shifts = [-s + 1 for s in reversed(n_shifts)]
                else:
                    n_shifts = [-s for s in reversed(n_shifts)]
                n_values_rhs = [v for v in reversed(n_values_rhs)]
                rhs_n_shifts = [-s for s in reversed(rhs_n_shifts)]

            v_ns_b0.insert(0, [n_values, n_shifts])
            rhs_v_ns_b0.insert(0, [n_values_rhs, rhs_n_shifts])

        if staggered and not output_boundary_valid:
            del v_ns_b0[0]
            del rhs_v_ns_b0[0]

            if len(v_ns_b0) == 0:
                v_ns_b0 = [[[], []]]

            if len(rhs_v_ns_b0) == 0:
                rhs_v_ns_b0 = [[[], []]]

    else:
        if staggered:
            del shifts[0]
            values, rhs_values = get_coefficients([s - 0.5 for s in shifts], 1, rhs_shifts)
        else:
            values, rhs_values = get_coefficients(shifts, 1, rhs_shifts)

        return values, shifts, rhs_values, rhs_shifts

    return [v_ns_b0, rhs_v_ns_b0]

a = [
        [
            [get_stencils(6, implicit_order=2, one_sided=True, left_border_one_sided=left_side,
                          staggered=True, output_boundary_valid=out_valid,
                          input_boundary_valid=in_valid)
             for in_valid in [False, True]]
            for out_valid in [False, True]]
        for left_side in [False, True]]


a = tensor(a, batch('left_side', 'out_valid', 'in_valid', 'left_right', 'position', 'koeff_shifts', 'values'))