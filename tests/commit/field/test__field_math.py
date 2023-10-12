from itertools import product
from unittest import TestCase

import numpy

import phi
from phiml import math
from phi import geom
from phi.field import StaggeredGrid, CenteredGrid, PointCloud, Noise
from phi.field._field_math import _lhs_for_implicit_scheme, _ex_map_f, pad, shift, stack
from phi.geom import Box, Sphere
from phi import field
from phiml.math import extrapolation, instance, channel, spatial, batch
from phiml.backend import Backend
from phiml.math.extrapolation import combine_by_direction, REFLECT, SYMMETRIC

BACKENDS = phi.detect_backends()


class TestFieldMath(TestCase):

    def test_spatial_gradient(self):
        s = CenteredGrid(1, x=4, y=3) * (1, 2)
        grad = field.spatial_gradient(s, stack_dim=channel('spatial_gradient'))
        self.assertEqual(('spatial', 'spatial', 'channel', 'channel'), grad.shape.types)

    def test_spatial_gradient_batched(self):
        bounds = geom.stack([Box['x,y', 0:1, 0:1], Box['x,y', 0:10, 0:10]], batch('batch'))
        grid = CenteredGrid(0, extrapolation.ZERO, bounds, x=10, y=10)
        grad = field.spatial_gradient(grid)
        self.assertIsInstance(grad, CenteredGrid)

    def test_laplace_batched(self):
        bounds = geom.stack([Box['x,y', 0:1, 0:1], Box['x,y', 0:10, 0:10]], batch('batch'))
        grid = CenteredGrid(0, extrapolation.ZERO, bounds, x=10, y=10)
        lap = field.laplace(grid)
        self.assertIsInstance(lap, CenteredGrid)

    def test_divergence_centered(self):
        v = CenteredGrid(1, extrapolation.ZERO, bounds=Box['x,y', 0:1, 0:1], x=3, y=3) * (1, 0)  # flow to the right
        div = field.divergence(v).values
        math.assert_close(div.y[0], [1.5, 0, -1.5])

    def test_trace_function(self):
        def f(x: StaggeredGrid, y: CenteredGrid):
            return x + (y @ x)

        ft = field.jit_compile(f)
        x = StaggeredGrid(1, x=4, y=3)
        y = CenteredGrid((1, 1), x=4, y=3)

        res_f = f(x, y)
        res_ft = ft(x, y)
        self.assertEqual(res_f.shape, res_ft.shape)
        field.assert_close(res_f, res_ft)

    def test_gradient_function(self):
        def f(x: StaggeredGrid, y: CenteredGrid):
            pred = x + (y @ x)
            loss = field.l2_loss(pred)
            return loss

        grad = field.functional_gradient(f, get_output=False)
        fgrad = field.functional_gradient(f, (0, 1), get_output=True)

        x = StaggeredGrid(1, x=4, y=3)
        y = CenteredGrid((1, 1), x=4, y=3)

        for backend in BACKENDS:
            if backend.supports(Backend.jacobian):
                with backend:
                    dx, = grad(x, y)
                    self.assertIsInstance(dx, StaggeredGrid)
                    loss, (dx, dy) = fgrad(x, y)
                    self.assertIsInstance(loss, math.Tensor)
                    self.assertIsInstance(dx, StaggeredGrid)
                    self.assertIsInstance(dy, CenteredGrid)

    def test_upsample_downsample_centered_1d(self):
        grid = CenteredGrid(math.tensor([0, 1, 2, 3], spatial('x')))
        upsampled = field.upsample2x(grid)
        downsampled = field.downsample2x(upsampled)
        math.assert_close(downsampled.values.x[1:-1], grid.values.x[1:-1])

    def test_downsample_staggered_2d(self):
        grid = StaggeredGrid(1, extrapolation.BOUNDARY, x=32, y=40)
        downsampled = field.downsample2x(grid)
        self.assertEqual(set(spatial(x=16, y=20) & channel(vector=2)), set(downsampled.shape))

    def test_abs(self):
        grid = StaggeredGrid(-1, x=4, y=3)
        field.assert_close(field.abs(grid), abs(grid), 1)

    def test_sign(self):
        grid = StaggeredGrid(0.5, x=4, y=3)
        field.assert_close(field.sign(grid), 1)

    def test_round_(self):
        grid = StaggeredGrid(1.7, x=4, y=3)
        field.assert_close(field.round(grid), 2)

    def test_ceil(self):
        grid = StaggeredGrid(1.1, x=4, y=3)
        field.assert_close(field.ceil(grid), 2)

    def test_floor(self):
        grid = StaggeredGrid(2.8, x=4, y=3)
        field.assert_close(field.floor(grid), 2)

    def test_sqrt(self):
        grid = StaggeredGrid(2, x=4, y=3)
        field.assert_close(field.sqrt(grid), numpy.sqrt(2))

    def test_exp(self):
        grid = StaggeredGrid(0, x=4, y=3)
        field.assert_close(field.exp(grid), 1)

    def test_isfinite(self):
        grid = StaggeredGrid(1, x=4, y=3)
        field.assert_close(field.isfinite(grid), True)

    def test_real(self):
        grid = StaggeredGrid(1, x=4, y=3)
        field.assert_close(field.real(grid), grid)

    def test_imag(self):
        grid = StaggeredGrid(1, x=4, y=3)
        field.assert_close(field.imag(grid), 0)

    def test_sin(self):
        grid = StaggeredGrid(0, x=4, y=3)
        field.assert_close(field.sin(grid), 0)

    def test_cos(self):
        grid = StaggeredGrid(0, x=4, y=3)
        field.assert_close(field.cos(grid), 1)

    def test_convert_grid(self):
        grid = CenteredGrid(0, x=4, y=3)
        for backend in BACKENDS:
            converted = field.convert(grid, backend)
            self.assertEqual(converted.values.default_backend, backend)

    def test_convert_point_cloud(self):
        loc = math.random_uniform(instance(points=4), channel(vector='x,y'))
        val = math.random_normal(instance(points=4), channel(vector='x,y'))
        points = PointCloud(Sphere(loc, radius=1), val)
        for backend in BACKENDS:
            converted = field.convert(points, backend)
            self.assertEqual(converted.values.default_backend, backend)
            self.assertEqual(converted.elements.center.default_backend, backend)
            self.assertEqual(converted.elements.radius.default_backend, backend)

    def test_center_of_mass(self):
        density = CenteredGrid(Box(x=1, y=(1, 2)), x=4, y=3)
        math.assert_close(field.center_of_mass(density), (0.5, 1.5))
        density = CenteredGrid(Box(x=None, y=(2, 3)), x=4, y=3)
        math.assert_close(field.center_of_mass(density), (2, 2.5))

    def test_curl_2d_centered_to_staggered(self):
        pot = CenteredGrid(Box['x,y', 1:2, 1:2], x=4, y=3)
        curl = field.curl(pot, type=StaggeredGrid)
        math.assert_close(field.mean(curl), 0)
        math.assert_close(curl.values.vector[0].x[1], [1, -1])
        math.assert_close(curl.values.vector[1].y[1], [-1, 1, 0])

    def test_curl_2d_staggered_to_centered(self):
        velocity = StaggeredGrid((2, 0), extrapolation.BOUNDARY, x=2, y=2)
        curl = field.curl(velocity)
        self.assertEqual(spatial(x=3, y=3), curl.resolution)

    def test_integrate_all(self):
        grid = CenteredGrid(field.Noise(vector=2), extrapolation.ZERO, x=10, y=10, bounds=Box['x,y', 0:10, 0:10])
        math.assert_close(field.integrate(grid, grid.bounds), math.sum(grid.values, 'x,y'))
        grid = CenteredGrid(field.Noise(vector=2), extrapolation.ZERO, x=10, y=10, bounds=Box['x,y', 0:1, 0:1])
        math.assert_close(field.integrate(grid, grid.bounds), math.sum(grid.values, 'x,y') / 100)

    def test_mask(self):
        mask = field.mask(Box(x=1, y=1))
        self.assertEqual(2, mask.spatial_rank)
        mask = field.mask(PointCloud(math.vec(x=0, y=0)))
        self.assertEqual(2, mask.spatial_rank)
        mask = field.mask(CenteredGrid(0, x=4, y=3))
        self.assertEqual(2, mask.spatial_rank)

    # def test_implicit_laplace_solve(self):
    #     grid = CenteredGrid(Noise(), x=5, y=5)
    #     axes_names = grid.shape.only(spatial).names
    #     extrap_map = {}
    #     extrap_map_rhs = {}
    #     values, needed_shifts = [3 / 44, 12 / 11, -51 / 22, 12 / 11, 3 / 44], (-2, -1, 0, 1, 2)
    #     extrap_map['symmetric'] = combine_by_direction(REFLECT, SYMMETRIC)
    #     values_rhs, needed_shifts_rhs = [2 / 11, 1, 2 / 11], (-1, 0, 1)
    #     extrap_map_rhs['symmetric'] = combine_by_direction(REFLECT, SYMMETRIC)
    #     base_widths = (abs(min(needed_shifts)), max(needed_shifts))
    #     grid.with_extrapolation(extrapolation.map(_ex_map_f(extrap_map), grid.extrapolation))
    #     padded_components = [pad(grid, {dim: base_widths}) for dim in axes_names]
    #     shifted_components = [shift(padded_component, needed_shifts, None, pad=False, dims=dim) for padded_component, dim in zip(padded_components, axes_names)]
    #     result_components = [sum([value * shift_ for value, shift_ in zip(values, shifted_component)]) / grid.dx.vector[dim] ** 2 for shifted_component, dim in zip(shifted_components, axes_names)]
    #     result_components = stack(result_components, channel('laplacian'))
    #     result_components.with_values(result_components.values._cache())
    #     result_components = result_components.with_extrapolation(extrapolation.map(_ex_map_f(extrap_map_rhs), grid.extrapolation))
    #     matrix, _ = math.matrix_from_function(_lhs_for_implicit_scheme, result_components, values_rhs=values_rhs, needed_shifts_rhs=needed_shifts_rhs, stack_dim=channel('laplacian'))
    #     direct_result = _lhs_for_implicit_scheme(result_components, values_rhs=values_rhs, needed_shifts_rhs=needed_shifts_rhs, stack_dim=channel('laplacian'))
    #     matrix_result = matrix @ result_components.values
    #     math.assert_close(matrix_result, direct_result)

