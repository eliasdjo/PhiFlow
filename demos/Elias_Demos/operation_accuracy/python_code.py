from functools import partial

from phi.jax.flow import *

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


def tgv_pressure_laplacian(x):
    cos_2x = math.cos(2 * x)
    return cos_2x.vector['x'] + cos_2x.vector['y']


def tgv_pressure_gradient(x):
    sin, cos = math.sin(x), math.cos(x)
    return math.stack([sin.vector['x'] * cos.vector['x'],
                       sin.vector['y'] * cos.vector['y']], dim=channel('vector'))


def tgv_velocity_diffuse(x):
    return math.stack(math.unstack(tgv_velocity(x, vis=0, t=0) + 1 * tgv_velocity_laplacian(x), "vector"),
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
    ret = velocities - 0.1 * math.sum(gradients * math.rename_dims(velocities, "vector", "gradient"), "gradient")
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
            jax_native=False, run_jitted=False, operation_args=[], pressure_input=False):
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

            DOMAIN = dict(x=xy_num, y=xy_num, extrapolation=extrapolation.PERIODIC,
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


xy_nums = [5, 15, 35, 65, 105, 165, 225]
# xy_nums = [35, 65, 105, 165, 225]
# xy_nums = [5, 65, 225]
# xy_nums = [10]

# TestRun("laplacian", xy_nums, CenteredGrid,
#         [partial(field.laplace),
#          partial(field.laplace, scheme=Scheme(4)),
#          partial(field.laplace, scheme=Scheme(6, Solve('CG', 1e-12, 1e-12)))],
#         tgv_velocity_laplacian, ["laplace", "laplace_kamp", "laplace_laiz"])
#
# TestRun("laplacian_staggered", xy_nums, StaggeredGrid,
#         [partial(field.laplace),
#          partial(field.laplace, scheme=Scheme(4)),
#          partial(field.laplace, scheme=Scheme(6, Solve('CG', 1e-12, 1e-12)))],
#         tgv_velocity_laplacian, ["laplace", "laplace_kamp", "laplace_laiz"])
#
# TestRun("diffusion", xy_nums, CenteredGrid,
#         [partial(diffuse.finite_difference, diffusivity=1, dt=1),
#          partial(diffuse.finite_difference, diffusivity=1, dt=1, scheme=Scheme(4)),
#          partial(diffuse.finite_difference, diffusivity=1, dt=1, scheme=Scheme(6, Solve('CG', 1e-12, 1e-12)))],
#         tgv_velocity_diffuse, ["explicit", "kamp", "laiz"])
#
# TestRun("diffusion_staggered", xy_nums, StaggeredGrid,
#         [partial(diffuse.finite_difference, diffusivity=1, dt=1),
#          partial(diffuse.finite_difference, diffusivity=1, dt=1, scheme=Scheme(4)),
#          partial(diffuse.finite_difference, diffusivity=1, dt=1, scheme=Scheme(6, Solve('CG', 1e-12, 1e-12)))],
#         tgv_velocity_diffuse, ["explicit", "kamp", "laiz"])
#
# TestRun("gradient_fst_comp", xy_nums, CenteredGrid,
#         [partial(field.spatial_gradient), partial(field.spatial_gradient, scheme=Scheme(4)),
#          partial(field.spatial_gradient, scheme=Scheme(6, Solve('CG', 1e-12, 1e-12)))],
#         tgv_velocity_gradient_fst_comp, ["spatial_gradient_lo", "spatial_gradient_kamp", "spatial_gradient_laiz"],
#         scalar_input=0)
#
# TestRun("gradient_staggered_fst_comp", xy_nums, StaggeredGrid,
#         [partial(field.spatial_gradient, type=StaggeredGrid),
#          partial(field.spatial_gradient, scheme=Scheme(4), type=StaggeredGrid),
#          partial(field.spatial_gradient, scheme=Scheme(6, Solve('CG', 1e-12, 1e-12)), type=StaggeredGrid)],
#         tgv_velocity_gradient_fst_comp,
#         ["spatial_gradient", "spatial_gradient_kamp", "spatial_gradient_laiz"],
#         scalar_input=0, input_gridtype=CenteredGrid)
#
# TestRun("gradient_snd_comp", xy_nums, CenteredGrid,
#         [partial(field.spatial_gradient), partial(field.spatial_gradient, scheme=Scheme(4)),
#          partial(field.spatial_gradient, scheme=Scheme(6, Solve('CG', 1e-12, 1e-12)))],
#         tgv_velocity_gradient_snd_comp, ["spatial_gradient_lo", "spatial_gradient_kamp", "spatial_gradient_laiz"],
#         scalar_input=1)
#
# TestRun("gradient_staggered_snd_comp", xy_nums, StaggeredGrid,
#         [partial(field.spatial_gradient, type=StaggeredGrid),
#          partial(field.spatial_gradient, scheme=Scheme(4), type=StaggeredGrid),
#          partial(field.spatial_gradient, scheme=Scheme(6, Solve('CG', 1e-12, 1e-12)), type=StaggeredGrid)],
#         tgv_velocity_gradient_snd_comp,
#         ["spatial_gradient", "spatial_gradient_kamp", "spatial_gradient_laiz"],
#         scalar_input=1, input_gridtype=CenteredGrid)

# TestRun("gradient_both_comp", xy_nums, CenteredGrid,
#         [partial(field.spatial_gradient, stack_dim=channel('gradient')),
#          partial(field.spatial_gradient, scheme=Scheme(4), stack_dim=channel('gradient')),
#          partial(field.spatial_gradient, scheme=Scheme(6, Solve('CG', 1e-12, 1e-12)), stack_dim=channel('gradient'))
#          ],
#         tgv_velocity_gradient_both_comp, ["spatial_gradient_lo", "spatial_gradient_kamp", "spatial_gradient_laiz"])

# TestRun("gradient_both_comp_stagg", xy_nums, StaggeredGrid,
#         [partial(field.spatial_gradient, stack_dim=channel('gradient')),
#          partial(field.spatial_gradient, scheme=Scheme(4), stack_dim=channel('gradient')),
#          partial(field.spatial_gradient, scheme=Scheme(6, Solve('CG', 1e-12, 1e-12)), stack_dim=channel('gradient'))
#          ],
#         tgv_velocity_gradient_both_comp, ["spatial_gradient_lo", "spatial_gradient_kamp", "spatial_gradient_laiz"])

TestRun("advection", xy_nums, CenteredGrid,
        [lambda v: advect.finite_difference(v, v, 0.1), lambda v: advect.finite_difference(v, v, 0.1, scheme=Scheme(4)),
         lambda v: advect.finite_difference(v, v, 0.1, scheme=Scheme(6, Solve('CG', 1e-12, 1e-12)))],
        tgv_velocity_advect, ["std", "kamp", "laiz"])

TestRun("advection_staggered", xy_nums, StaggeredGrid,
        [
            lambda v: advect.finite_difference(v, v, 0.1),
            lambda v: advect.finite_difference(v, v, 0.1, scheme=Scheme(4)),
            lambda v: advect.finite_difference(v, v, 0.1, scheme=Scheme(6, Solve('CG', 1e-12, 1e-12)))
        ],
        tgv_velocity_advect, ["std", "kamp", "laiz"])
#
# TestRun("divergence", xy_nums, CenteredGrid,
#         [partial(field.divergence), partial(field.divergence, scheme=Scheme(4)),
#          partial(field.divergence, scheme=Scheme(6, Solve('CG', 1e-12, 1e-12)))],
#         tgv_velocity_div, ["divergence", "divergence_kamp", "divergence_laiz"])
#
# TestRun("divergence_staggered", xy_nums, StaggeredGrid,
#         [partial(field.divergence), partial(field.divergence, scheme=Scheme(4)),
#          partial(field.divergence, scheme=Scheme(6, Solve('CG', 1e-12, 1e-12)))],
#         tgv_velocity_div, ["divergence", "divergence_kamp", "divergence_laiz"])

print("done")
