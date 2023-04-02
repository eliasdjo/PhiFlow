from phi.math import zeros, ones, concat, channel, tensor, dual, factorial
from phi import math
import numpy as np
# from fd_coefficient_calc3 import get_coefficients as get_coefficients2



def taylor_coeff2(offset, n, deriv):
    coeff = (offset) ** abs(n-deriv) / factorial(n-deriv)
    res = math.where(n-deriv >= 0, coeff, 0)
    return res


def get_coefficients(offsets, derivative, lhs_offsets=[], boundary_condition=None):
    handle_zero = 0 in lhs_offsets
    if handle_zero:
        lhs_offsets = lhs_offsets.copy()
        zero_index = lhs_offsets.index(0)
        lhs_offsets.remove(0)

    bc = boundary_condition is not None
    if bc:
        bc_offset, bc_deriv, bc_value = boundary_condition

    node_number = len(offsets + lhs_offsets) + bc

    one = concat([zeros(channel(x=derivative)), ones(channel(x=1)), zeros(channel(x=node_number - derivative - 1))], 'x')
    # ToDo ed switch zero(...) -> phiml.math.expand(0, ...)

    arange = tensor(np.arange(node_number), channel('x'))
    coeff = taylor_coeff2(tensor(offsets, dual('x')), arange, 0)
    coeff_lhs = taylor_coeff2(tensor(lhs_offsets, dual('x')), arange, derivative)
    mat = math.concat([coeff, coeff_lhs], '~x')

    if bc:
        coeff_bc = taylor_coeff2(tensor([bc_offset], dual('x')), arange, bc_deriv)
        mat = math.concat([mat, coeff_bc], '~x')

    np_mat = mat.numpy('x, ~x')
    np_b = one.numpy('x')
    coeff = np.linalg.solve(np_mat, np_b)
    ret = list(coeff)
    values, lhs_values = ret[:len(offsets)], ret[len(offsets):len(offsets + lhs_offsets)]
    lhs_values = [-v for v in lhs_values]

    bc_offset = 0
    if bc:
        bc_offset = ret[-1] * bc_value

    if handle_zero:
        lhs_values.insert(zero_index, 1)

    return values, lhs_values, bc_offset


# get_coefficients([-2, -1, 0, 1, 2], 1, [-1, 0, 1])
# print(get_coefficients([-2, -1, 0, 1, 2], 1, [-1, 0, 1]) == get_coefficients2([-2, -1, 0, 1, 2], 1, [-1, 0, 1]))
#
# print(get_coefficients([-2, -1, 0, 1, 2], 1, [-2, -1, 0, 1, 2]) == get_coefficients2([-2, -1, 0, 1, 2], 1, [-2, -1, 0, 1, 2]))
# print(get_coefficients([-2, -1, 0, 1, 2], 1, [0]) == get_coefficients2([-2, -1, 0, 1, 2], 1, [0]))
# print(get_coefficients([-2, -1, 0, 1, 2], 1, [-1, 0, 1]) == get_coefficients2([-2, -1, 0, 1, 2], 1, [-1, 0, 1]))
# print(get_coefficients([0, 1, 2, 3, 4], 1) == get_coefficients2([0, 1, 2, 3, 4], 1))
# print(get_coefficients([-1, 0, 1, 2, 3, 4], 1) == get_coefficients2([-1, 0, 1, 2, 3, 4], 1))
# print(get_coefficients([-2, -1, 0, 1, 2, 3, 4], 1) == get_coefficients2([-2, -1, 0, 1, 2, 3, 4], 1))
#
# print(get_coefficients([-2, -1, 1, 2], 0) == get_coefficients2([-2, -1, 1, 2], 0))
# print(get_coefficients([-2, -1, 0, 1, 2], 0, [0]) == get_coefficients2([-2, -1, 0, 1, 2], 0, [0]))
# print(get_coefficients([-2, -1, 0, 1, 2], 0) == get_coefficients2([-2, -1, 0, 1, 2], 0))
# print(get_coefficients([0, 1, 2, 3, 4], 0) == get_coefficients2([0, 1, 2, 3, 4], 0))
# print(get_coefficients([-1, 1, 2, 3, 4], 0) == get_coefficients2([-1, 1, 2, 3, 4], 0))
# print(get_coefficients([-2, -1, 1, 2, 3, 4], 0) == get_coefficients2([-2, -1, 1, 2, 3, 4], 0))
#
# print(get_coefficients([-2, -1, 0, 1, 2], 2, [-2, -1, 0, 1, 2]) == get_coefficients2([-2, -1, 0, 1, 2], 2, [-2, -1, 0, 1, 2]))
# print(get_coefficients([-2, -1, 0, 1, 2], 2, [0]) == get_coefficients2([-2, -1, 0, 1, 2], 2, [0]))
# print(get_coefficients([-2, -1, 0, 1, 2], 2, [-1, 0, 1]) == get_coefficients2([-2, -1, 0, 1, 2], 2, [-1, 0, 1]))
# print(get_coefficients([0, 1, 2, 3, 4], 2) == get_coefficients2([0, 1, 2, 3, 4], 2))
# print(get_coefficients([-1, 0, 1, 2, 3], 2) == get_coefficients2([-1, 0, 1, 2, 3], 2))
# print(get_coefficients([-2, -1, 0, 1, 2, 3, 4], 2) == get_coefficients2([-2, -1, 0, 1, 2, 3, 4], 2))
