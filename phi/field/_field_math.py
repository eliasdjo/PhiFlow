from numbers import Number
from typing import Callable, List, Tuple, Optional, Union

from phi import geom
from phi import math
from phi.geom import Box, Geometry
from phiml.math import Tensor, spatial, instance, tensor, channel, Shape, unstack, solve_linear, jit_compile_linear, shape, Solve, extrapolation, jit_compile, rename_dims, flatten, batch
from ._field import Field, SampledField, SampledFieldType, as_extrapolation
from ._grid import CenteredGrid, Grid, StaggeredGrid, GridType
from ._mesh import Mesh
from ._point_cloud import PointCloud
from phiml.math.extrapolation import Extrapolation, SYMMETRIC, REFLECT, ANTIREFLECT, ANTISYMMETRIC, combine_by_direction
from demos.Elias_Demos.fd_coefficient_calc3 import get_coefficients

def bake_extrapolation(grid: GridType) -> GridType:
    """
    Pads `grid` with its current extrapolation.
    For `StaggeredGrid`s, the resulting grid will have a consistent shape, independent of the original extrapolation.

    Args:
        grid: `CenteredGrid` or `StaggeredGrid`.

    Returns:
        Padded grid with extrapolation `phiml.math.extrapolation.NONE`.
    """
    if grid.extrapolation == math.extrapolation.NONE:
        return grid
    if isinstance(grid, StaggeredGrid):
        values = tuple(grid.values.vector)
        padded = []
        for dim, value in zip(grid.shape.spatial.names, values):
            lower, upper = grid.extrapolation.valid_outer_faces(dim)
            padded.append(math.pad(value, {dim: (0 if lower else 1, 0 if upper else 1)}, grid.extrapolation[{'vector': dim}], bounds=grid.bounds))
        return StaggeredGrid(math.stack(padded, grid.shape['vector']), bounds=grid.bounds, extrapolation=math.extrapolation.NONE)
    elif isinstance(grid, CenteredGrid):
        return pad(grid, 1).with_extrapolation(math.extrapolation.NONE)
    else:
        raise ValueError(f"Not a valid grid: {grid}")


def laplace(field: GridType,
            axes=spatial,
            order=2,
            implicit: math.Solve = None,
            weights: Union[Tensor, Field] = None,
            implicitness: int = None) -> GridType:
    """
    Spatial Laplace operator for scalar grid.
    If a vector grid is passed, it is assumed to be centered and the laplace is computed component-wise.

    Args:
        field: n-dimensional `CenteredGrid`
        axes: The second derivative along these dimensions is summed over
        weights: (Optional) Multiply the axis terms by these factors before summation.
            Must be a `phiml.math.Tensor` or `phi.field.Field` with a single channel dimension that lists all laplace axes by name.
        order: Spatial order of accuracy.
            Higher orders entail larger stencils and more computation time but result in more accurate results assuming a large enough resolution.
            Supported: 2 explicit, 4 explicit, 6 implicit.
        implicit: When a `Solve` object is passed, performs an implicit operation with the specified solver and tolerances.
            Otherwise, an explicit stencil is used.

    Returns:
        laplacian field as `CenteredGrid`
    """

    if implicitness is None:
        implicitness = 0 if implicit is None else 2
    elif implicitness != 0:
        assert implicit is not None, "for implicit treatment a `Solve` is required"

    laplace_dims = field.shape.only(axes).names

    if field.vector.exists:
        fields = [f for f in field.vector]
    else:
        fields = [field]

    result = []
    for f in fields:
        result_components = [perform_finite_difference_operation(f.values, dim, 2, f.dx.vector[dim], f.extrapolation,
                                                                 f.extrapolation, CenteredGrid, order, implicit, implicitness) for dim in laplace_dims]
        if weights is not None:
            assert channel(weights).rank == 1 and channel(
                weights).item_names is not None, f"weights must have one channel dimension listing the laplace dims but got {shape(weights)}"
            assert set(channel(weights).item_names[0]) >= set(laplace_dims), f"the channel dim of weights must contain all laplace dims {laplace_dims} but only has {channel(weights).item_names}"
            result_components = [c * weights[ax] for c, ax in zip(result_components, laplace_dims)]

        result.append(sum(result_components))

    if field.vector.exists:
        result = math.stack(result, channel('vector'))
    else:
        result = result[0]

    return field.with_values(result)


def spatial_gradient(field: CenteredGrid,
                     gradient_extrapolation: Extrapolation = None,
                     type: type = CenteredGrid,
                     grad_dims: math.DimFilter = spatial,
                     stack_dim: Shape = channel('vector'),
                     order=2,
                     implicit: Solve = None,
                     implicitness: int = None):
    """
    Finite difference spatial_gradient.

    This function can operate in two modes:

    * `type=CenteredGrid` approximates the spatial_gradient at cell centers using central differences
    * `type=StaggeredGrid` computes the spatial_gradient at face centers of neighbouring cells

    Args:
        field: centered grid of any number of dimensions (scalar field, vector field, tensor field)
        gradient_extrapolation: Extrapolation of the output
        type: either `CenteredGrid` or `StaggeredGrid`
        dims: Along which dimensions to compute the spatial gradient. Only supported when `type==CenteredGrid`.
        stack_dim: Dimension to be added. This dimension lists the spatial_gradient w.r.t. the spatial dimensions.
            The `field` must not have a dimension of the same name.
        order: Spatial order of accuracy.
            Higher orders entail larger stencils and more computation time but result in more accurate results assuming a large enough resolution.
            Supported: 2 explicit, 4 explicit, 6 implicit.
        implicit: When a `Solve` object is passed, performs an implicit operation with the specified solver and tolerances.
            Otherwise, an explicit stencil is used.

    Returns:
        spatial_gradient field of type `type`.
    """

    if field.vector.exists:
        assert stack_dim.name != 'vector', "`stack_dim=vector` is inadmissible if the input is a vector grid"
        if field == StaggeredGrid:
            assert type == StaggeredGrid, "for a `StaggeredGrid` input only `type == StaggeredGrid` is possible"
            type = CenteredGrid

    if type == StaggeredGrid:
        assert stack_dim.name == 'vector', f"spatial_gradient with type=StaggeredGrid requires stack_dim.name == 'vector' but got '{stack_dim.name}'"

    if gradient_extrapolation is None:
        # gradient_extrapolation = field.extrapolation.spatial_gradient()
        gradient_extrapolation = field.extrapolation

    if implicitness is None:
        implicitness = 0 if implicit is None else 2
    elif implicitness != 0:
        assert implicit is not None, "for implicit treatment a `Solve` is required"

    grad_dims = field.shape.only(grad_dims).names

    if stack_dim is None:
        assert len(grad_dims) == 1, "`stack_dim` `None` is only possible with single `grad_dim`"
    else:
        stack_dim = stack_dim._with_item_names((grad_dims,))

    result_components = [perform_finite_difference_operation(field.values, dim, 1, field.dx.vector[dim], field.extrapolation,
                                                             gradient_extrapolation, type, order, implicit, implicitness)
                         for dim in field.shape.only(grad_dims).names]

    if type == CenteredGrid:
        result = field.with_values(math.stack(result_components, stack_dim))
        result = result.with_extrapolation(gradient_extrapolation)
    else:
        result = StaggeredGrid(
            math.stack(result_components, stack_dim),
            bounds=field.bounds, extrapolation=gradient_extrapolation)

    if type == CenteredGrid and gradient_extrapolation == math.extrapolation.NONE:
        result = result.with_bounds(Box(field.bounds.lower - field.dx, field.bounds.upper + field.dx))
    else:
        result = result.with_bounds(field.bounds)

    return result


def get_stencils(order, differentiation_order, implicit_order=0, one_sided=False, left_border_one_sided=False, staggered=False, # ToDo ed raus in externe funktion mit param gridtype order impl dann aufruf mit with math.NUMPY
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
                off_rhs = max(0, rhs_extend - max_extend + i + (not output_boundary_valid and staggered))
                n_shifts = [*range(-extend + off, extend + 1 + off + off_rhs)]
                rhs_n_shifts = [*range(-rhs_extend + off_rhs, rhs_extend + 1)] if implicit_order else []

                if staggered:
                    n_shifts = [n + 1 for n in n_shifts]
                    coefficient_shifts = [n - 0.5 for n in n_shifts]
                    if input_boundary_valid:
                        del coefficient_shifts[-1]
                        coefficient_shifts.insert(0, coefficient_shifts[0]-0.5)
                        del n_shifts[-1]
                        n_shifts.insert(0, n_shifts[0]-1)
                    n_values, n_values_rhs = get_coefficients(coefficient_shifts, differentiation_order, rhs_n_shifts)
                else:
                    coefficient_shifts = n_shifts.copy()
                    if input_boundary_valid:
                        del coefficient_shifts[-1]
                        coefficient_shifts.insert(0, coefficient_shifts[0]-0.5)
                        del n_shifts[-1]
                        n_shifts.insert(0, n_shifts[0]-1)
                    n_values, n_values_rhs = get_coefficients(coefficient_shifts, differentiation_order, rhs_n_shifts)

                if left_border_one_sided:
                    n_values = [v*(-1)**differentiation_order for v in reversed(n_values)]
                    if staggered:
                        n_shifts = [-s+1 for s in reversed(n_shifts)]
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
                values, rhs_values = get_coefficients([s - 0.5 for s in shifts], differentiation_order, rhs_shifts)
            else:
                values, rhs_values = get_coefficients(shifts, differentiation_order, rhs_shifts)

            return values, shifts, rhs_values, rhs_shifts

        return [v_ns_b0, rhs_v_ns_b0]


def perform_finite_difference_operation(field: Tensor, dim: str, differentiation_order: int, dx: float,
                                        ext: Extrapolation,
                                        output_ext: Extrapolation = None,
                                        output_type: type = CenteredGrid,
                                        order=2,
                                        implicit: Solve = None,
                                        implicitness: int = None):


    if output_ext is None:
        output_ext = ext

    if implicitness is None:
        implicitness = 0 if implicit is None else 2
    elif implicitness != 0:
        assert implicit is not None, "for implicit treatment a `Solve` is required"

    assert dim in field.shape.spatial.names, "given Tensor needs to have the indicated spatial dimension"

    with math.NUMPY:
        base_values, base_shifts, base_rhs_values, base_rhs_shifts = get_stencils(order, differentiation_order, implicit_order=implicitness,
                                                                                  one_sided=False,
                                                                                  left_border_one_sided=False,
                                                                                  staggered=output_type==StaggeredGrid,
                                                                                  output_boundary_valid=False,
                                                                                  input_boundary_valid=False)

        one_sided_stencils = \
            [
                [
                    [get_stencils(order, differentiation_order, implicit_order=implicitness, one_sided=True, left_border_one_sided=left_side,
                                  staggered=output_type==StaggeredGrid, output_boundary_valid=out_valid,
                                  input_boundary_valid=in_valid)
                     for in_valid in [False, True]]
                    for out_valid in [False, True]]
                for left_side in [False, True]]

    # one_sided_stencil_tensor = tensor(one_sided_stencils,
    #                             batch('left_side', 'out_valid', 'in_valid', 'left_right', 'position', 'koeff_shifts', 'values'))
    left_one_sided_stencil_tensor = [[[tensor(l3[0], batch('position', 'koeff_shifts', 'values')) for l3 in l2] for l2 in l1] for l1 in one_sided_stencils]
    right_one_sided_stencil_tensor = [[[tensor(l3[1], batch('position', 'koeff_shifts', 'values')) for l3 in l2] for l2 in l1] for l1 in one_sided_stencils]

    # boundary masks
    if output_type == CenteredGrid:
        standard_mask = CenteredGrid(0, resolution=field.shape)     # ToDo ed is this okay with batch dimensions?
    else:
        standard_mask = CenteredGrid(0, resolution=field.shape + spatial(**{dim: sum(output_ext.valid_outer_faces(dim)) - 1}))

    output_valid_ext = extrapolation.combine_sides(**{dim: tuple(
        extrapolation.ONE if valid_tuple else extrapolation.ZERO for valid_tuple in
        output_ext.valid_outer_faces(dim)) for dim in field.shape.spatial.names})
    # output_valid_ext = extrapolation.ZERO
    output_valid_mask = standard_mask.with_extrapolation(output_valid_ext)

    def ext_list_to_map_func(target_extrapolations):
        def f(ext: Extrapolation):
            return extrapolation.ONE if ext in target_extrapolations else extrapolation.ZERO
        return f

    input_valid_ext = extrapolation.map(ext_list_to_map_func([extrapolation.ZERO, extrapolation.ZERO_GRADIENT]),
                                        ext)
    # input_valid_ext = extrapolation.ZERO
    input_valid_mask = standard_mask.with_extrapolation(input_valid_ext)

    one_sided_ext = extrapolation.map(ext_list_to_map_func([extrapolation.ConstantExtrapolation(math.nan),
                                                            extrapolation.ConstantExtrapolation(10000),
                                                            extrapolation.ConstantExtrapolation(100),
                                                            extrapolation.ZERO,
                                                            extrapolation.ZERO_GRADIENT,
                                                            extrapolation.SYMMETRIC,
                                                            ]), ext)
    # one_sided_ext = extrapolation.ONE
    one_sided_mask = standard_mask.with_extrapolation(one_sided_ext)

    result = apply_stencils(field, ext, output_ext, dx, base_values, base_shifts, output_type, dim,
                            masks=(output_valid_mask, input_valid_mask, one_sided_mask), stencil_tensors=left_one_sided_stencil_tensor,
                            differencing_order=differentiation_order)

    if implicit:
        implicit.x0 = result
        result = solve_linear(apply_stencils, result, solve=implicit, field_extrapolation=ext,
                              gradient_extrapolation=output_ext,
                              field_dx=dx, base_koeff=base_rhs_values, base_shifts=base_rhs_shifts,
                              type=CenteredGrid, dim=dim,
                              masks=(output_valid_mask, input_valid_mask, one_sided_mask), stencil_tensors=right_one_sided_stencil_tensor, differencing_order=0)

    return result


@jit_compile_linear(auxiliary_args="field_extrapolation, gradient_extrapolation, field_dx, base_koeff, base_shifts, type, dim, masks, stencil_tensors, differencing_order")
def apply_stencils(field, field_extrapolation, gradient_extrapolation, field_dx, base_koeff, base_shifts, type, dim, masks=None, stencil_tensors=None, differencing_order=1):
    from itertools import product
    spatial_dims = field.shape.spatial.names

    def apply_stencil(values_, needed_shifts_):
        needed_shifts_ = [int(i) for i in needed_shifts_]
        base_widths = (max(-min(needed_shifts_), 0), max(max(needed_shifts_), 0))

        std_widths = (0, 0)
        if type == CenteredGrid:
            if gradient_extrapolation == math.extrapolation.NONE:
                base_widths = (base_widths[0] + 1, base_widths[1] + 1)
                std_widths = (1, 1)
        elif type == StaggeredGrid:
            base_widths = (base_widths[0], base_widths[1] - 1)
            border_valid = gradient_extrapolation.valid_outer_faces(dim)
            base_widths = (border_valid[0] + base_widths[0], border_valid[1] + base_widths[1])
        else:
            raise ValueError(type)

        padded_component = math.pad(field,
                                    {dim_: base_widths if dim_ == dim else std_widths for dim_ in spatial_dims},
                                    field_extrapolation)

        shifted_component = math.shift(padded_component, tuple(needed_shifts_), stack_dim=None, padding=None, dims=dim)
        result_component = (sum([value * shift for value, shift in zip(values_, shifted_component)]) / field_dx**differencing_order)

        return result_component

    result_component = apply_stencil(base_koeff, base_shifts)
    if masks is not None and stencil_tensors is not None:
        output_valid_mask, input_valid_mask, one_sided_mask = masks
        one_mask = one_sided_mask.with_values(1).with_extrapolation(extrapolation.ONE)
        for left_side, out_valid, in_valid in product([0, 1], repeat=3):
            stencils = stencil_tensors[left_side][out_valid][in_valid]
            input_valid_mask_ = input_valid_mask if in_valid else one_mask - input_valid_mask
            output_valid_mask_ = output_valid_mask if out_valid else one_mask - output_valid_mask
            mask = input_valid_mask_ * output_valid_mask_ * one_sided_mask
            isolation_mask = 0          # for obstacles we will have to tinker around here
            for i, stencils_i in enumerate(stencils.position):
                values_b0, needed_shifts_b0 = stencils_i.koeff_shifts
                if len(values_b0) != 0 and len(values_b0) != 0:
                    # rc = apply_stencil(values_b0, needed_shifts_b0)

                    one_sided_components = apply_stencil(values_b0, needed_shifts_b0)
                    mask_ = shift(mask, ((i+1) if left_side else -(i+1),), dims=dim, stack_dim=None)[0].values - isolation_mask
                    isolation_mask = isolation_mask + mask_
                    # rc = result_component * (1 - mask_) + one_sided_components * mask_
                    rc = math.where(mask_, one_sided_components, result_component)

                    result_component = rc

    return result_component

def tond(input):
    return input.numpy(input.shape.names)


@jit_compile_linear(auxiliary_args="values_rhs, needed_shifts_rhs, v_ns_b0_rhs, stack_dim, staggered_output")
def _rhs_for_implicit_scheme(x, values_rhs, needed_shifts_rhs, v_ns_b0_rhs, stack_dim, staggered_output):
    # result = []
    # for dim, component in zip(x.shape.only(math.spatial).names, unstack(x, stack_dim.name)):
    #     shifted = shift(component, needed_shifts_rhs, stack_dim=None, dims=dim)
    #     result.append(sum([value * shift_ for value, shift_ in zip(values_rhs, shifted)]))
    def apply_stencil(values_rhs_, needed_shifts_rhs_):
        result = []
        for dim, component in zip(x.shape.only(math.spatial).names, unstack(x, stack_dim.name)):
            shifted = shift(component, needed_shifts_rhs_, stack_dim=None, dims=dim)
            result.append(sum([value * shift for value, shift in zip(values_rhs_, shifted)]))
        return result

    result = apply_stencil(values_rhs, needed_shifts_rhs)

    for i, (values_b0, needed_shifts_b0) in enumerate(v_ns_b0_rhs):

        one_sided_components = apply_stencil(values_b0, needed_shifts_b0)
        one_sided_components_top = apply_stencil([val for val in reversed(values_b0)],
                                                 [-shift for shift in reversed(needed_shifts_b0)])

        for dim_i, dim in enumerate(x.shape.spatial.names):
            rc = result[dim_i]
            shape = rc.values.shape
            # mask_tensor = math.zeros(shape) + math.scatter(math.zeros(shape.only(dim)),
            #                                                tensor([i], instance('points')),
            #                                                tensor([1], instance('points')))
            mask_tensor = math.zeros(shape) + math.concat(
                [math.zeros(spatial(**{dim: i})), math.ones(spatial(**{dim: 1})),
                 math.zeros(spatial(**{dim: shape.only(dim).size - (i + 1)}))], dim)

            # rc = rc * rc.with_values(mask_tensor_) * rc.with_values(mask_tensor_.flip(dim))
            # rc = rc + one_sided_components[dim_i] * rc.with_values(mask_tensor) + one_sided_components_top[
            #     dim_i] * rc.with_values(mask_tensor.flip(dim))
            # result[dim_i] = rc

            rc_tensor = math.where(mask_tensor, one_sided_components[dim_i].values, rc.values)
            rc_tensor = math.where(mask_tensor.flip(dim), one_sided_components_top[dim_i].values, rc_tensor)
            result[dim_i] = rc.with_values(rc_tensor)

    if staggered_output:
        result = x.with_values(math.stack([component.values for component in result], channel('vector')))
    else:
        result = stack(result, stack_dim)

    return result


def pad_for_staggered_output(field: CenteredGrid, output_extrapolation: Extrapolation, dims: tuple, base_widths: tuple):
    padded_components = []
    for dim in dims:
        border_valid = output_extrapolation.valid_outer_faces(dim)
        padding_widths = (border_valid[0] + base_widths[0], border_valid[1] + base_widths[1])
        padded_components.append(math.pad(field, {dim: padding_widths}))

    return padded_components


def shift(grid: CenteredGrid, offsets: tuple, stack_dim: Optional[Shape] = channel('shift'), dims=spatial, pad=True):
    """
    Wraps :func:`math.shift` for CenteredGrid.

    Args:
      grid: CenteredGrid: 
      offsets: tuple: 
      stack_dim:  (Default value = 'shift')
    """
    if pad:
        padding = grid.extrapolation
        new_bounds = grid.bounds
    else:
        padding = None
        max_lower_shift = min(offsets) if min(offsets) < 0 else 0
        max_upper_shift = max(offsets) if max(offsets) > 0 else 0
        w_lower = math.wrap([max_lower_shift if dim in dims else 0 for dim in grid.shape.spatial.names])
        w_upper = math.wrap([max_upper_shift if dim in dims else 0 for dim in grid.shape.spatial.names])
        new_bounds = Box(grid.box.lower - w_lower * grid.dx, grid.box.upper - w_upper * grid.dx)
    data = math.shift(grid.values, offsets, dims=dims, padding=padding, stack_dim=stack_dim)
    return [type(grid)(data[i], bounds=new_bounds, extrapolation=grid.extrapolation) for i in range(len(offsets))]


def stagger(field: CenteredGrid,
            face_function: Callable,
            extrapolation: Union[float, math.extrapolation.Extrapolation],
            type: type = StaggeredGrid):
    """
    Creates a new grid by evaluating `face_function` given two neighbouring cells.
    One layer of missing cells is inferred from the extrapolation.
    
    This method returns a Field of type `type` which must be either StaggeredGrid or CenteredGrid.
    When returning a StaggeredGrid, the new values are sampled at the faces of neighbouring cells.
    When returning a CenteredGrid, the new grid has the same resolution as `field`.

    Args:
      field: centered grid
      face_function: function mapping (value1: Tensor, value2: Tensor) -> center_value: Tensor
      extrapolation: extrapolation mode of the returned grid. Has no effect on the values.
      type: one of (StaggeredGrid, CenteredGrid)
      field: CenteredGrid: 
      face_function: Callable:
      extrapolation: math.extrapolation.Extrapolation: 
      type: type:  (Default value = StaggeredGrid)

    Returns:
      grid of type matching the `type` argument

    """
    extrapolation = as_extrapolation(extrapolation)
    all_lower = []
    all_upper = []
    if type == StaggeredGrid:
        for dim in field.resolution.names:
            valid_lo, valid_up = extrapolation.valid_outer_faces(dim)
            if valid_lo and valid_up:
                width_lower, width_upper = {dim: (1, 0)}, {dim: (0, 1)}
            elif valid_lo and not valid_up:
                width_lower, width_upper = {dim: (1, -1)}, {dim: (0, 0)}
            elif not valid_lo and valid_up:
                width_lower, width_upper = {dim: (0, 0)}, {dim: (-1, 1)}
            else:
                width_lower, width_upper = {dim: (0, -1)}, {dim: (-1, 0)}
            all_lower.append(math.pad(field.values, width_lower, field.extrapolation, bounds=field.bounds))
            all_upper.append(math.pad(field.values, width_upper, field.extrapolation, bounds=field.bounds))
        all_upper = math.stack(all_upper, channel('vector'))
        all_lower = math.stack(all_lower, channel('vector'))
        values = face_function(all_lower, all_upper)
        result = StaggeredGrid(values, bounds=field.bounds, extrapolation=extrapolation)
        assert result.shape.spatial == field.shape.spatial
        return result
    elif type == CenteredGrid:
        left, right = math.shift(field.values, (-1, 1), padding=field.extrapolation, stack_dim=channel('vector'))
        values = face_function(left, right)
        return CenteredGrid(values, bounds=field.bounds, extrapolation=extrapolation)
    else:
        raise ValueError(type)


def divergence(field: Grid, order=2, implicit: Solve = None, implicitness=None) -> CenteredGrid:
    """
    Computes the divergence of a grid using finite differences.

    This function can operate in two modes depending on the type of `field`:

    * `CenteredGrid` approximates the divergence at cell centers using central differences
    * `StaggeredGrid` exactly computes the divergence at cell centers

    Args:
        field: vector field as `CenteredGrid` or `StaggeredGrid`
        order: Spatial order of accuracy.
            Higher orders entail larger stencils and more computation time but result in more accurate results assuming a large enough resolution.
            Supported: 2 explicit, 4 explicit, 6 implicit.
        implicit: When a `Solve` object is passed, performs an implicit operation with the specified solver and tolerances.
            Otherwise, an explicit stencil is used.

    Returns:
        Divergence field as `CenteredGrid`
    """

    components = [spatial_gradient(f, grad_dims=dim, type=CenteredGrid, order=order, implicit=implicit, implicitness=implicitness, stack_dim=None)
                         for f, dim in zip(field.vector, field.shape.only(spatial).names)]

    return sum(components)


def curl(field: Grid, type: type = CenteredGrid):
    """ Computes the finite-difference curl of the give 2D `StaggeredGrid`. """
    assert field.spatial_rank in (2, 3), "curl is only defined in 2 and 3 spatial dimensions."
    if isinstance(field, CenteredGrid) and field.spatial_rank == 2:
        if 'vector' not in field.shape and type == StaggeredGrid:
            # 2D curl of scalar field
            grad = math.spatial_gradient(field.values, dx=field.dx, difference='forward', padding=None, stack_dim=channel('vector'))
            result = grad.vector[::-1] * (1, -1)  # (d/dy, -d/dx)
            bounds = Box(field.bounds.lower + 0.5 * field.dx, field.bounds.upper - 0.5 * field.dx)  # lose 1 cell per dimension
            return StaggeredGrid(result, bounds=bounds, extrapolation=field.extrapolation.spatial_gradient())
        if 'vector' in field.shape and type == CenteredGrid:
            # 2D curl of vector field
            x, y = field.shape.spatial.names
            vy_dx = math.spatial_gradient(field.values.vector[1], dx=field.dx.vector[0], padding=field.extrapolation, dims=x, stack_dim=None)
            vx_dy = math.spatial_gradient(field.values.vector[0], dx=field.dx.vector[1], padding=field.extrapolation, dims=y, stack_dim=None)
            c = vy_dx - vx_dy
            return field.with_values(c)
    elif isinstance(field, StaggeredGrid) and field.spatial_rank == 2:
        if type == CenteredGrid:
            values = bake_extrapolation(field).values
            x_padded = math.pad(values.vector['x'], {'y': (1, 1)}, field.extrapolation)
            y_padded = math.pad(values.vector['y'], {'x': (1, 1)}, field.extrapolation)
            vx_dy = math.spatial_gradient(x_padded, field.dx, 'forward', None, dims='y', stack_dim=None)
            vy_dx = math.spatial_gradient(y_padded, field.dx, 'forward', None, dims='x', stack_dim=None)
            result = vy_dx - vx_dy
            return CenteredGrid(result, field.extrapolation.spatial_gradient(), bounds=field.bounds)
    raise NotImplementedError()


def fourier_laplace(grid: GridType, times=1) -> GridType:
    """ See `phiml.math.fourier_laplace()` """
    assert grid.extrapolation.spatial_gradient() == math.extrapolation.PERIODIC
    values = math.fourier_laplace(grid.values, dx=grid.dx, times=times)
    return type(grid)(values=values, bounds=grid.bounds, extrapolation=grid.extrapolation)


def fourier_poisson(grid: GridType, times=1) -> GridType:
    """ See `phiml.math.fourier_poisson()` """
    assert grid.extrapolation.spatial_gradient() == math.extrapolation.PERIODIC
    values = math.fourier_poisson(grid.values, dx=grid.dx, times=times)
    return type(grid)(values=values, bounds=grid.bounds, extrapolation=grid.extrapolation)


def native_call(f, *inputs, channels_last=None, channel_dim='vector', extrapolation=None) -> Union[SampledField, Tensor]:
    """
    Similar to `phiml.math.native_call()`.

    Args:
        f: Function to be called on native tensors of `inputs.values`.
            The function output must have the same dimension layout as the inputs and the batch size must be identical.
        *inputs: `SampledField` or `phi.Tensor` instances.
        extrapolation: (Optional) Extrapolation of the output field. If `None`, uses the extrapolation of the first input field.

    Returns:
        `SampledField` matching the first `SampledField` in `inputs`.
    """
    input_tensors = [i.values if isinstance(i, SampledField) else tensor(i) for i in inputs]
    values = math.native_call(f, *input_tensors, channels_last=channels_last, channel_dim=channel_dim)
    for i in inputs:
        if isinstance(i, SampledField):
            result = i.with_values(values=values)
            if extrapolation is not None:
                result = result.with_extrapolation(extrapolation)
            return result
    else:
        raise AssertionError("At least one input must be a SampledField.")


def data_bounds(loc: Union[SampledField, Tensor]) -> Box:
    if isinstance(loc, SampledField):
        loc = loc.points
    assert isinstance(loc, Tensor), f"loc must be a Tensor or SampledField but got {type(loc)}"
    min_vec = math.min(loc, dim=loc.shape.non_batch.non_channel)
    max_vec = math.max(loc, dim=loc.shape.non_batch.non_channel)
    return Box(min_vec, max_vec)


def mean(field: SampledField) -> Tensor:
    """
    Computes the mean value by reducing all spatial / instance dimensions.

    Args:
        field: `SampledField`

    Returns:
        `phi.Tensor`
    """
    return math.mean(field.values, field.shape.non_channel.non_batch)


def normalize(field: SampledField, norm: SampledField, epsilon=1e-5):
    """ Multiplies the values of `field` so that its sum matches the source. """
    data = math.normalize_to(field.values, norm.values, epsilon)
    return field.with_values(data)


def center_of_mass(density: SampledField):
    """
    Compute the center of mass of a density field.

    Args:
        density: Scalar `SampledField`

    Returns:
        `Tensor` holding only batch dimensions.
    """
    assert 'vector' not in density.shape
    return mean(density.points * density) / mean(density)


def pad(grid: GridType, widths: Union[int, tuple, list, dict]) -> GridType:
    """
    Pads a `Grid` using its extrapolation.

    Unlike `phiml.math.pad()`, this function also affects the `bounds` of the grid, changing its size and origin depending on `widths`.

    Args:
        grid: `CenteredGrid` or `StaggeredGrid`
        widths: Either `int` or `(lower, upper)` to pad the same number of cells in all spatial dimensions
            or `dict` mapping dimension names to `(lower, upper)`.

    Returns:
        `Grid` of the same type as `grid`
    """
    if isinstance(widths, int):
        widths = {axis: (widths, widths) for axis in grid.shape.spatial.names}
    elif isinstance(widths, (tuple, list)):
        widths = {axis: (width if isinstance(width, (tuple, list)) else (width, width)) for axis, width in zip(grid.shape.spatial.names, widths)}
    else:
        assert isinstance(widths, dict)
    widths_list = [widths[axis] if axis in widths.keys() else (0, 0) for axis in grid.shape.spatial.names]
    if isinstance(grid, Grid):
        data = math.pad(grid.values, widths, grid.extrapolation, bounds=grid.bounds)
        w_lower = math.wrap([w[0] for w in widths_list])
        w_upper = math.wrap([w[1] for w in widths_list])
        bounds = Box(grid.box.lower - w_lower * grid.dx, grid.box.upper + w_upper * grid.dx)
        return type(grid)(values=data, bounds=bounds, extrapolation=grid.extrapolation)
    raise NotImplementedError(f"{type(grid)} not supported. Only Grid instances allowed.")


def downsample2x(grid: Grid) -> GridType:
    """
    Reduces the number of sample points by a factor of 2 in each spatial dimension.
    The new values are determined via linear interpolation.

    See Also:
        `upsample2x()`.

    Args:
        grid: `CenteredGrid` or `StaggeredGrid`.

    Returns:
        `Grid` of same type as `grid`.
    """
    if isinstance(grid, CenteredGrid):
        values = math.downsample2x(grid.values, grid.extrapolation)
        return CenteredGrid(values, bounds=grid.bounds, extrapolation=grid.extrapolation)
    elif isinstance(grid, StaggeredGrid):
        values = []
        for dim, centered_grid in zip(grid.shape.spatial.names, unstack(grid, 'vector')):
            odd_discarded = centered_grid.values[{dim: slice(None, None, 2)}]
            others_interpolated = math.downsample2x(odd_discarded, grid.extrapolation, dims=grid.shape.spatial.without(dim))
            values.append(others_interpolated)
        return StaggeredGrid(math.stack(values, channel('vector')), bounds=grid.bounds, extrapolation=grid.extrapolation)
    else:
        raise ValueError(type(grid))


def upsample2x(grid: GridType) -> GridType:
    """
    Increases the number of sample points by a factor of 2 in each spatial dimension.
    The new values are determined via linear interpolation.

    See Also:
        `downsample2x()`.

    Args:
        grid: `CenteredGrid` or `StaggeredGrid`.

    Returns:
        `Grid` of same type as `grid`.
    """
    if isinstance(grid, CenteredGrid):
        values = math.upsample2x(grid.values, grid.extrapolation)
        return CenteredGrid(values, bounds=grid.bounds, extrapolation=grid.extrapolation)
    elif isinstance(grid, StaggeredGrid):
        raise NotImplementedError()
    else:
        raise ValueError(type(grid))


def concat(fields: Union[List[SampledFieldType], Tuple[SampledFieldType, ...]], dim: Union[str, Shape]) -> SampledFieldType:
    """
    Concatenates the given `SampledField`s along `dim`.

    See Also:
        `stack()`.

    Args:
        fields: List of matching `SampledField` instances.
        dim: Concatenation dimension as `Shape`. Size is ignored.

    Returns:
        `SampledField` matching concatenated fields.
    """
    assert all(isinstance(f, SampledField) for f in fields)
    assert all(isinstance(f, type(fields[0])) for f in fields)
    if any(f.extrapolation != fields[0].extrapolation for f in fields):
        raise NotImplementedError("Concatenating extrapolations not supported")
    if isinstance(fields[0], Grid):
        values = math.concat([f.values for f in fields], dim)
        return fields[0].with_values(values)
    elif isinstance(fields[0], PointCloud):
        elements = geom.concat([f.elements for f in fields], dim)
        values = math.concat([math.expand(f.values, f.shape.only(dim)) for f in fields], dim)
        return PointCloud(elements=elements, values=values, extrapolation=fields[0].extrapolation, add_overlapping=fields[0]._add_overlapping, bounds=fields[0]._bounds)
    raise NotImplementedError(type(fields[0]))


def stack(fields, dim: Shape, dim_bounds: Box = None):
    """
    Stacks the given `SampledField`s along `dim`.

    See Also:
        `concat()`.

    Args:
        fields: List of matching `SampledField` instances.
        dim: Stack dimension as `Shape`. Size is ignored.
        dim_bounds: `Box` defining the physical size for `dim`.

    Returns:
        `SampledField` matching stacked fields.
    """
    assert all(isinstance(f, SampledField) for f in fields), f"All fields must be SampledFields of the same type but got {fields}"
    assert all(isinstance(f, type(fields[0])) for f in fields), f"All fields must be SampledFields of the same type but got {fields}"
    if any(f.extrapolation != fields[0].extrapolation for f in fields):
        raise NotImplementedError("Concatenating extrapolations not supported")
    if isinstance(fields[0], Grid):
        values = math.stack([f.values for f in fields], dim)
        if spatial(dim):
            if dim_bounds is None:
                dim_bounds = Box(**{dim.name: len(fields)})
            return type(fields[0])(values, extrapolation=fields[0].extrapolation, bounds=fields[0].bounds * dim_bounds)
        else:
            return fields[0].with_values(values)
    elif isinstance(fields[0], PointCloud):
        elements = geom.stack([f.elements for f in fields], dim)
        values = math.stack([f.values for f in fields], dim)
        return PointCloud(elements=elements, values=values, extrapolation=fields[0].extrapolation, add_overlapping=fields[0]._add_overlapping, bounds=fields[0]._bounds)
    raise NotImplementedError(type(fields[0]))


def assert_close(*fields: Union[SampledField, Tensor, Number],
                 rel_tolerance: float = 1e-5,
                 abs_tolerance: float = 0,
                 msg: str = "",
                 verbose: bool = True):
    """ Raises an AssertionError if the `values` of the given fields are not close. See `phiml.math.assert_close()`. """
    f0 = next(filter(lambda t: isinstance(t, SampledField), fields))
    values = [(f @ f0).values if isinstance(f, SampledField) else math.wrap(f) for f in fields]
    math.assert_close(*values, rel_tolerance=rel_tolerance, abs_tolerance=abs_tolerance, msg=msg, verbose=verbose)


def where(mask: Union[Field, Geometry, float], field_true: Union[Field, float], field_false: Union[Field, float]) -> SampledFieldType:
    """
    Element-wise where operation.
    Picks the value of `field_true` where `mask=1 / True` and the value of `field_false` where `mask=0 / False`.

    The fields are automatically resampled if necessary, preferring the sample points of `mask`.
    At least one of the arguments must be a `SampledField`.

    Args:
        mask: `Field` or `Geometry` object.
        field_true: `Field`
        field_false: `Field`

    Returns:
        `SampledField`
    """
    field_true, field_false, mask = _auto_resample(field_true, field_false, mask)
    values = math.where(mask.values, field_true.values, field_false.values)
    return field_true.with_values(values)


def maximum(f1: Union[Field, Geometry, float], f2: Union[Field, Geometry, float]):
    """
    Element-wise maximum.
    One of the given fields needs to be an instance of `SampledField` and the the result will be sampled at the corresponding points.
    If both are `SampledFields` but have different points, `f1` takes priority.

    Args:
        f1: `Field` or `Geometry` or constant.
        f2: `Field` or `Geometry` or constant.

    Returns:
        `SampledField`
    """
    f1, f2 = _auto_resample(f1, f2)
    return f1.with_values(math.maximum(f1.values, f2.values))


def minimum(f1: Union[Field, Geometry, float], f2: Union[Field, Geometry, float]):
    """
    Element-wise minimum.
    One of the given fields needs to be an instance of `SampledField` and the the result will be sampled at the corresponding points.
    If both are `SampledFields` but have different points, `f1` takes priority.

    Args:
        f1: `Field` or `Geometry` or constant.
        f2: `Field` or `Geometry` or constant.

    Returns:
        `SampledField`
    """
    f1, f2 = _auto_resample(f1, f2)
    return f1.with_values(math.minimum(f1.values, f2.values))


def _auto_resample(*fields: Field):
    """ Prefers extrapolation from first SampledField """
    for sampled_field in fields:
        if isinstance(sampled_field, SampledField):
            return [f @ sampled_field for f in fields]
    raise AssertionError(f"At least one argument must be a SampledField but got {fields}")


def vec_length(field: SampledField):
    """ See `phiml.math.vec_abs()` """
    assert isinstance(field, SampledField), f"SampledField required but got {type(field).__name__}"
    if isinstance(field, StaggeredGrid):
        field = field.at_centers()
    return field.with_values(math.vec_abs(field.values))


def vec_squared(field: SampledField):
    """ See `phiml.math.vec_squared()` """
    if isinstance(field, StaggeredGrid):
        field = field.at_centers()
    return field.with_values(math.vec_squared(field.values))


def finite_fill(grid: GridType, distance=1, diagonal=True) -> GridType:
    """
    Extrapolates values of `grid` which are marked by nonzero values in `valid` using `phiml.math.masked_fill().
    If `values` is a StaggeredGrid, its components get extrapolated independently.

    Args:
        grid: Grid holding the values for extrapolation and possible non-finite values to be filled.
        distance: Number of extrapolation steps, i.e. how far a cell can be from the closest finite value to get filled.
        diagonal: Whether to extrapolate values to their diagonal neighbors per step.

    Returns:
        grid: Grid with extrapolated values.
        valid: binary Grid marking all valid values after extrapolation.
    """
    if isinstance(grid, CenteredGrid):
        new_values = math.finite_fill(grid.values, distance=distance, diagonal=diagonal, padding=grid.extrapolation)
        return grid.with_values(new_values)
    elif isinstance(grid, StaggeredGrid):
        new_values = [finite_fill(c, distance=distance, diagonal=diagonal).values for c in grid.vector]
        return grid.with_values(math.stack(new_values, channel(grid)))
    else:
        raise ValueError(grid)


def discretize(grid: Grid, filled_fraction=0.25):
    """ Treats channel dimensions as batch dimensions. """
    import numpy as np
    data = math.reshaped_native(grid.values, [grid.shape.non_spatial, grid.shape.spatial])
    ranked_idx = np.argsort(data, axis=-1)
    filled_idx = ranked_idx[:, int(round(grid.shape.spatial.volume * (1 - filled_fraction))):]
    filled = np.zeros_like(data)
    np.put_along_axis(filled, filled_idx, 1, axis=-1)
    filled_t = math.reshaped_tensor(filled, [grid.shape.non_spatial, grid.shape.spatial])
    return grid.with_values(filled_t)


def integrate(field: Field, region: Geometry, **kwargs) -> Tensor:
    """
    Computes *∫<sub>R</sub> f(x) dx<sup>d</sup>* , where *f* denotes the `Field`, *R* the `region` and *d* the number of spatial dimensions (`d=field.shape.spatial_rank`).
    Depending on the `sample` implementation for `field`, the integral may be a rough approximation.

    This method is currently only implemented for `CenteredGrid`.

    Args:
        field: `Field` to integrate.
        region: Region to integrate over.
        **kwargs: Specify numerical scheme.

    Returns:
        Integral as `phi.Tensor`
    """
    if not isinstance(field, CenteredGrid):
        raise NotImplementedError()
    return field._sample(region, **kwargs) * region.volume


def pack_dims(field: SampledFieldType,
              dims: Union[Shape, tuple, list, str],
              packed_dim: Shape,
              pos: Union[int, None] = None) -> SampledFieldType:
    """
    Currently only supports grids and non-spatial dimensions.

    See Also:
        `phiml.math.pack_dims()`.

    Args:
        field: `SampledField`

    Returns:
        `SampledField` of same type as `field`.
    """
    if isinstance(field, Grid):
        if spatial(field.shape.only(dims)):
            raise NotImplementedError("Packing spatial dimensions not supported for grids")
        return field.with_values(math.pack_dims(field.values, dims, packed_dim, pos))
    else:
        raise NotImplementedError()


def support(field: SampledField, list_dim: Union[Shape, str] = instance('nonzero')) -> Tensor:
    """
    Returns the points at which the field values are non-zero.

    Args:
        field: `SampledField`
        list_dim: Dimension to list the non-zero values.

    Returns:
        `Tensor` with shape `(list_dim, vector)`
    """
    return field.points[math.nonzero(field.values, list_dim=list_dim)]


def mask(obj: Union[SampledFieldType, Geometry]) -> SampledFieldType:
    """
    Returns a `Field` that masks the inside (or non-zero values when `obj` is a grid) of a physical object.
    The mask takes the value 1 inside the object and 0 outside.
    For `CenteredGrid` and `StaggeredGrid`, the mask labels non-zero non-NaN entries as 1 and all other values as 0

    Returns:
        `Grid` type or `PointCloud`
    """
    if isinstance(obj, PointCloud):
        return PointCloud(obj.elements, 1, math.extrapolation.remove_constant_offset(obj.extrapolation), bounds=obj.bounds)
    elif isinstance(obj, Geometry):
        return PointCloud(obj, 1, 0)
    elif isinstance(obj, CenteredGrid):
        values = math.cast(obj.values != 0, int)
        return obj.with_values(values)
    else:
        raise ValueError(obj)


def connect(obj: SampledField, connections: Tensor) -> Mesh:
    """
    Build a `Mesh` by connecting elements from a field.

    See Also:
        `connect_neighbors()`.

    Args:
        obj: `PointCloud` or `Mesh`.
        connections: Connectivity matrix. Any non-zero entry represents a connection.

    Returns:
        `Mesh`
    """
    if isinstance(obj, (PointCloud, Mesh)):
        return Mesh(obj.elements, connections, obj.values, extrapolation=obj.extrapolation, bounds=obj.bounds)
    else:
        raise ValueError(f"connect requires a PointCloud or Mesh but got {type(obj)}")


def connect_neighbors(obj: SampledField, max_distance: float or Tensor, format: str = 'dense') -> Mesh:
    """
    Build  a `Mesh` by connecting proximate elements of a `SampledField`.

    See Also:
        `connect()`.

    Args:
        obj: `PointCloud`, `Mesh`, `CenteredGrid` or `StaggeredGrid`.
        max_distance: Connectivity threshold distance. Elements further apart than this will not be connected.
        format: Connectivity matrix format, `'dense'`, `'coo'` or `'csr'`.

    Returns:
        `Mesh`.
    """
    if isinstance(obj, CenteredGrid):
        elements = flatten(obj.elements, instance('elements'))
        values = math.pack_dims(obj.values, spatial, instance('elements'))
        obj = PointCloud(elements, values, obj.extrapolation, bounds=obj.bounds)
    elif isinstance(obj, StaggeredGrid):
        elements = flatten(obj.elements, instance('elements'), flatten_batch=True)
        values = math.pack_dims(obj.values, spatial(obj.values).names + ('vector',), instance('elements'))
        obj = PointCloud(elements, values, obj.extrapolation, bounds=obj.bounds)
    assert isinstance(obj, (PointCloud, Mesh)), f"obj must be a PointCloud, Mesh or Grid but got {type(obj)}"
    points = math.rename_dims(obj.elements, spatial, instance).center
    dx = math.pairwise_distances(points, max_distance=max_distance, format=format)
    con = math.vec_length(dx) > 0
    return connect(obj, con)
