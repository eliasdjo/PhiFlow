import math
import os
from functools import partial

from phi.jax.flow import *
# from phi.flow import *

math.set_global_precision(64)


def tgv_F(vis, t):
    return math.exp(-2 * vis * t)


def tgv_velocity(x, vis=0, t=0):
    sin, cos = math.sin(x), math.cos(x)
    return math.stack([cos.vector['x'] * sin.vector['y'] * tgv_F(vis, t),
                       - sin.vector['x'] * cos.vector['y'] * tgv_F(vis, t)], dim=channel('vector'))


def tgv_pressure(x, vis=0, t=0):
    return -1 / 4 * (math.sum(math.cos(2 * x), 'vector')) * (tgv_F(vis, t) ** 2)


def tgv_velocity_laplacian(x):
    sin, cos = math.sin(x), math.cos(x)
    return math.stack([- 2 * cos.vector['x'] * sin.vector['y'],
                       2 * sin.vector['x'] * cos.vector['y']], dim=channel('vector'))

def tgv_velocity_laplacian_fst_comp(x):
    sin, cos = math.sin(x), math.cos(x)
    return - 2 * cos.vector['x'] * sin.vector['y']



def tgv_pressure_laplacian(x):
    cos_2x = math.cos(2 * x)
    return cos_2x.vector['x'] + cos_2x.vector['y']


def tgv_pressure_gradient(x):
    sin, cos = math.sin(x), math.cos(x)
    return math.stack([sin.vector['x'] * cos.vector['x'],
                       sin.vector['y'] * cos.vector['y']], dim=channel('vector'))


def tgv_velocity_diffuse(x):
    return math.stack(math.unstack(1 * tgv_velocity_laplacian(x), "vector"),
                      dim=channel('vector'))


def tgv_velocity_gradient_fst_comp(x):
    sin, cos = math.sin(x), math.cos(x)
    return math.stack([- sin.vector['x'] * sin.vector['y'],
                       cos.vector['x'] * cos.vector['y']], dim=channel('vector'))


def tgv_velocity_gradient_snd_comp(x):
    sin, cos = math.sin(x), math.cos(x)
    return math.stack([- cos.vector['x'] * cos.vector['y'],
                       sin.vector['x'] * sin.vector['y']], dim=channel('vector'))


def tgv_velocity_gradient_both_comp(x):
    sin, cos = math.sin(x), math.cos(x)
    return math.stack([math.stack([- sin.vector['x'] * sin.vector['y'],
                                   cos.vector['x'] * cos.vector['y']], dim=channel('gradient')),
                       math.stack([- cos.vector['x'] * cos.vector['y'],
                                   sin.vector['x'] * sin.vector['y']], dim=channel('gradient'))],
                      dim=channel("vector"))


def tgv_velocity_advect(x):
    sin, cos = math.sin(x), math.cos(x)
    velocities = tgv_velocity(x, vis=0, t=0)
    gradients = math.stack([math.stack([- sin.vector['x'] * sin.vector['y'],
                                        cos.vector['x'] * cos.vector['y']], dim=channel('gradient')),
                            math.stack([- cos.vector['x'] * cos.vector['y'],
                                        sin.vector['x'] * sin.vector['y']], dim=channel('gradient'))],
                           dim=channel('vector'))
    ret = -math.sum(gradients * math.rename_dims(velocities, "vector", "gradient"), "gradient")
    return math.stack(math.unstack(ret, "vector"), channel("vector"))


def tgv_velocity_div(x):
    return math.stack(math.unstack(math.zeros(x.shape), "vector"), channel("vector"))


def tgv_velocity_deriv_to_center_fst_comp(x):
    sin, cos = math.sin(x), math.cos(x)
    return - sin.vector['x'] * sin.vector['y']


def tgv_velocity_deriv_to_center_snd_comp(x):
    sin, cos = math.sin(x), math.cos(x)
    return sin.vector['x'] * sin.vector['y']


def TestRun(name, xy_nums, gridtype, operations, anal_sol_func,
            operations_strs=None,
            scalar_input=-1, input_gridtype=None, scalar_output=False, output_gridtype=None,
            jax_native=False, run_jitted=False, operation_args=[], pressure_input=False, boundaries=0):
    try:
        os.mkdir(f"data")
    except:
        pass

    if input_gridtype is None:
        input_gridtype = gridtype

    if output_gridtype is None:
        output_gridtype = gridtype

    error_by_operation = []
    for i, operation in enumerate(operations):
        operations_strs = operations_strs if operations_strs is not None else [op.__name__ for op in operations]
        print(f"operation: {operations_strs[i]}")

        if run_jitted:
            operation_jit = math.jit_compile(operation)
        else:
            operation_jit = operation

        errors_by_res = []

        for xy_num in xy_nums:

            print(f"xy_num: {xy_num}")


            if boundaries == -1:
                extrapola = math.nan
            elif boundaries == 0:
                extrapola = extrapolation.PERIODIC
            elif boundaries == 1:
                extrapola = extrapolation.ConstantExtrapolation(10000)
            elif boundaries == 2:
                extrapola = extrapolation.combine_sides(x=extrapolation.PERIODIC, y=extrapolation.ConstantExtrapolation(10000))
            elif boundaries == 3:
                extrapola = extrapolation.combine_sides(x=(extrapolation.PERIODIC, extrapolation.ConstantExtrapolation(10000)),
                                                        y=(extrapolation.ConstantExtrapolation(10000), extrapolation.PERIODIC))
            elif boundaries == 4:
                extrapola = extrapolation.combine_sides(x=extrapolation.PERIODIC, y=extrapolation.ZERO)
            elif boundaries == 5:
                extrapola = extrapolation.combine_sides(x=(extrapolation.PERIODIC, 10000),
                                                        y=(extrapolation.ConstantExtrapolation(10000), extrapolation.ZERO))
            else:
                raise NotImplementedError

            DOMAIN = dict(x=xy_num, y=xy_num, extrapolation=extrapola,
                          bounds=Box['x,y', 0:2 * math.pi, 0:2 * math.pi])
            if scalar_output:
                anal_sol = CenteredGrid(anal_sol_func, **DOMAIN)
            else:
                anal_sol = output_gridtype(anal_sol_func, **DOMAIN)

            if pressure_input:
                input = CenteredGrid(partial(tgv_pressure, vis=0, t=0), **DOMAIN)
            else:
                if scalar_input >= 0:
                    input = input_gridtype(partial(tgv_velocity, vis=0, t=0), **DOMAIN).vector[scalar_input]
                else:
                    input = input_gridtype(partial(tgv_velocity, vis=0, t=0), **DOMAIN)

            op_args = [input]
            for is_func, arg in operation_args:
                if not is_func:
                    op_args.append(arg)
                else:
                    op_args.append(arg(input))

            if not jax_native:
                operation_result = operation_jit(*op_args)
            else:
                op_args[0] = input.values.native(input.shape.names)
                operation_result_native = operation_jit(*op_args)

                if scalar_output:
                    operation_result_tensor = math.tensor(operation_result_native, math.spatial('x'), math.spatial('y'))
                    operation_result = CenteredGrid(operation_result_tensor, **DOMAIN)
                else:
                    operation_result_tensor = math.tensor(operation_result_native, math.spatial('x'), math.spatial('y'),
                                                          math.channel('vector'))
                    operation_result = CenteredGrid(operation_result_tensor, **DOMAIN)

            # for i in [1, 2]:
            #     for i2 in [1, 2]:
            #         plot(anal_sol.vector[i].gradient[i2])
            #         show()
            #         plot(operation_result.vector[i].gradient[i2])
            #         show()

            error = math.sum(math.abs(anal_sol.values - operation_result.values)) / xy_num ** 2
            print(error)

            error_point = math.stack([tensor(xy_num), error], channel('vector'))
            errors_by_res.append(error_point)

        error_by_operation.append(math.stack(errors_by_res, channel("res")))
        print()

    np.savez("data/" + name, data=math.stack(error_by_operation, channel("op")).numpy(('op', 'vector', 'res')),
             xy_nums=operations_strs)


# diverges  - biCG-stab
# TestRun(f"test", [35], CenteredGrid,
#         [partial(field.spatial_gradient, order=6, implicit=Solve('biCG-stab', 1e-12, 1e-12), implicitness=2, type=CenteredGrid)],
#         tgv_velocity_gradient_fst_comp, ["ord_2"],
#         scalar_input=0, input_gridtype=CenteredGrid, boundaries=1)


# works - biCG-stab(2)
# TestRun(f"test", [35], CenteredGrid,
#         [partial(field.spatial_gradient, order=6, implicit=Solve('biCG-stab(2)', 1e-12, 1e-12), implicitness=2, type=CenteredGrid)],
#         tgv_velocity_gradient_fst_comp, ["ord_2"],
#         scalar_input=0, input_gridtype=CenteredGrid, boundaries=1)


# works only if _field_math.py apply_stencils (line 335) is not jit linear compiled
TestRun(f"test", [35], StaggeredGrid,
        [partial(field.spatial_gradient, order=2, type=StaggeredGrid)],
        tgv_velocity_gradient_fst_comp, ["ord_2"],
        scalar_input=0, input_gridtype=CenteredGrid, boundaries=1)




print("done")
