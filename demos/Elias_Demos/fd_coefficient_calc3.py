from phi.math import zeros, ones, concat, channel, tensor, dual
from phi import math
import numpy as np

def lhs_matrix(offsets, derivative, lhs_offsets):

    def taylor_coeff(offset, n):
        def factorial(x):
            x_packed = math.pack_dims(x, x.shape.names, channel('dim'))
            x_list = list(x_packed)
            x_list = [np.math.factorial(elm) if elm >= 0 else math.inf for elm in x_list]
            x_packed = tensor(x_list, x_packed.shape)
            return math.unpack_dim(x_packed, 'dim', x.shape)

        coeff = (offset ** n) / factorial(n)
        lhs_coeff = -((offset*1.0) ** abs(n-derivative)) / factorial(n-derivative)
        mask = zeros(coeff.shape) + concat([ones(dual(x=len(offsets))), zeros(dual(x=len(lhs_offsets)))], '~x')
        return math.where(mask, coeff, lhs_coeff)

    all_offsets = tensor(offsets + lhs_offsets, dual('x'))
    arange = tensor(np.arange(len(offsets + lhs_offsets)), channel('x'))
    A = taylor_coeff(all_offsets, arange)
    return A

def get_coefficients(offsets, derivative, lhs_offsets=[]):
    handle_zero = 0 in lhs_offsets
    if handle_zero:
        lhs_offsets = lhs_offsets.copy()
        zero_index = lhs_offsets.index(0)
        lhs_offsets.remove(0)
    one = concat([zeros(channel(x=derivative)), ones(channel(x=1)), zeros(channel(x=len(offsets + lhs_offsets) - derivative - 1))], 'x')
    mat = lhs_matrix(offsets, derivative, lhs_offsets)
    np_mat = mat.numpy('x, ~x')
    np_b = one.numpy('x')
    coeff = np.linalg.solve(np_mat, np_b)
    ret = list(coeff)
    values, lhs_values = ret[:len(ret)-len(lhs_offsets)], ret[len(ret)-len(lhs_offsets):]
    if handle_zero:
        lhs_values.insert(zero_index, 1)
    return values, lhs_values

# print(get_coefficients([-2, -1, 0, 1, 2], 1, [1, -1]))
# print(get_coefficients([-2, -1, 0, 1, 2, 3, 4], 2))
# print(get_coefficients([-0.5, 0.5], 1))
