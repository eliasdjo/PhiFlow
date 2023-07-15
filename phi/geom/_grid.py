from typing import Tuple, Dict, Any, Optional

import numpy as np

from ._box import BaseBox, Box, Cuboid
from ._geom import Geometry, GeometryException
from .. import math
from ..math import Shape, Tensor, Extrapolation, stack, vec
from phiml.math._shape import shape_stack, dual
from ..math.magic import slicing_dict


class UniformGrid(BaseBox):
    """
    An instance of UniformGrid represents all cells of a regular grid as a batch of boxes.
    """

    def __init__(self, resolution: Shape, bounds: BaseBox):
        assert resolution.spatial_rank == resolution.rank, f"resolution must be purely spatial but got {resolution}"
        assert resolution.spatial_rank == bounds.spatial_rank, f"bounds must match dimensions of resolution but got {bounds} for resolution {resolution}"
        assert resolution.is_uniform, f"spatial dimensions must form a uniform grid but got {resolution}"
        assert set(bounds.vector.item_names) == set(resolution.names)
        self._resolution = resolution.only(bounds.vector.item_names, reorder=True)
        self._bounds = bounds
        self._shape = self._resolution & bounds.shape.non_spatial

    @property
    def resolution(self):
        return self._resolution

    @property
    def bounds(self):
        return self._bounds

    @property
    def spatial_rank(self) -> int:
        return self._resolution.spatial_rank

    @property
    def center(self):
        local_coords = math.meshgrid(**{dim.name: math.linspace(0.5 / dim.size, 1 - 0.5 / dim.size, dim) for dim in self.resolution})
        points = self.bounds.local_to_global(local_coords)
        return points

    @property
    def boundary_elements(self) -> Dict[str, Tuple[Dict[str, slice], Dict[str, slice]]]:
        return {}

    @property
    def boundary_faces(self) -> Dict[str, Tuple[Dict[str, slice], Dict[str, slice]]]:
        result = {}
        for dim in self.vector.item_names:
            result[(dim, False)] = {'~vector': dim, dim: slice(1)}
            result[(dim, True)] = {'~vector': dim, dim: slice(-1, None)}
        return result

    @property
    def face_centers(self) -> Tensor:
        centers = [self.stagger(dim, True, True).center for dim in self.vector.item_names]
        return stack(centers, dual(vector=self.vector.item_names))

    @property
    def face_normals(self) -> Tensor:
        normals = [vec(**{d: float(d == dim) for d in self.vector.item_names}) for dim in self.vector.item_names]
        return stack(normals, dual(vector=self.vector.item_names))

    @property
    def face_areas(self) -> Tensor:
        areas = [math.prod(self.dx.vector[[d for d in self.vector.item_names if d != dim]], 'vector') for dim in self.vector.item_names]
        return stack(areas, dual(vector=self.vector.item_names))

    @property
    def face_shape(self) -> Shape:
        shapes = [self._shape.spatial.with_dim_size(dim, self._shape.get_size(dim) + 1) for dim in self.vector.item_names]
        return shape_stack(dual(vector=self.vector.item_names), *shapes)

    def interior(self) -> 'Geometry':
        raise GeometryException("Regular grid does not have an interior")

    @property
    def grid_size(self):
        return self._bounds.size

    @property
    def size(self):
        return self.bounds.size / math.wrap(self.resolution.sizes)

    @property
    def dx(self):
        return self.bounds.size / self.resolution

    @property
    def lower(self):
        return self.center - self.half_size

    @property
    def upper(self):
        return self.center + self.half_size

    @property
    def half_size(self):
        return self.bounds.size / self.resolution.sizes / 2

    def __getitem__(self, item):
        item = slicing_dict(self, item)
        bounds = self._bounds
        dx = self.size
        gather_dict = {}
        for dim, selection in item.items():
            if dim in self._resolution:
                if isinstance(selection, int):
                    start = selection
                    stop = selection + 1
                elif isinstance(selection, slice):
                    start = selection.start or 0
                    if start < 0:
                        start += self.resolution.get_size(dim)
                    stop = selection.stop or self.resolution.get_size(dim)
                    if stop < 0:
                        stop += self.resolution.get_size(dim)
                    assert selection.step is None or selection.step == 1
                else:
                    raise ValueError(f"Illegal selection: {item}")
                dim_mask = math.wrap(self.resolution.mask(dim))
                lower = bounds.lower + start * dim_mask * dx
                upper = bounds.upper + (stop - self.resolution.get_size(dim)) * dim_mask * dx
                bounds = Box(lower, upper)
                gather_dict[dim] = slice(start, stop)
        resolution = self._resolution.after_gather(gather_dict)
        return UniformGrid(resolution, bounds[{d: s for d, s in item.items() if d != 'vector'}])

    def __pack_dims__(self, dims: Tuple[str, ...], packed_dim: Shape, pos: Optional[int], **kwargs) -> 'Cuboid':
        return math.pack_dims(self.center_representation(), dims, packed_dim, pos, **kwargs)

    @staticmethod
    def __stack__(values: tuple, dim: Shape, **kwargs) -> 'Geometry':
        from ._stack import GeometryStack
        return GeometryStack(math.layout(values, dim))

    def list_cells(self, dim_name):
        center = math.pack_dims(self.center, self._shape.spatial.names, dim_name)
        return Cuboid(center, self.half_size)

    def stagger(self, dim: str, lower: bool, upper: bool):
        dim_mask = np.array(self.resolution.mask(dim))
        unit = self.bounds.size / self.resolution * dim_mask
        bounds = Box(self.bounds.lower + unit * (-0.5 if lower else 0.5), self.bounds.upper + unit * (0.5 if upper else -0.5))
        ext_res = self.resolution.sizes + dim_mask * (int(lower) + int(upper) - 1)
        return UniformGrid(self.resolution.with_sizes(ext_res), bounds)

    def staggered_cells(self, boundaries: Extrapolation) -> Dict[str, 'UniformGrid']:
        grids = {}
        for dim in self.vector.item_names:
            grids[dim] = self.stagger(dim, *boundaries.valid_outer_faces(dim))
        return grids

    def padded(self, widths: dict):
        resolution, bounds = self.resolution, self.bounds
        for dim, (lower, upper) in widths.items():
            masked_dx = self.dx * math.dim_mask(self.resolution, dim)
            resolution = resolution.with_dim_size(dim, self.resolution.get_size(dim) + lower + upper)
            bounds = Box(bounds.lower - masked_dx * lower, bounds.upper + masked_dx * upper)
        return UniformGrid(resolution, bounds)

    @property
    def shape(self):
        return self._shape

    def shifted(self, delta: Tensor, **delta_by_dim) -> BaseBox:
        # delta += math.padded_stack()
        if delta.shape.spatial_rank == 0:
            return UniformGrid(self.resolution, self.bounds.shifted(delta))
        else:
            center = self.center + delta
            return Cuboid(center, self.half_size)

    def rotated(self, angle) -> Geometry:
        raise NotImplementedError("Grids cannot be rotated. Use center_representation() to convert it to Cuboids first.")

    def __eq__(self, other):
        return isinstance(other, UniformGrid) and self._bounds == other._bounds and self._resolution == other._resolution

    def shallow_equals(self, other):
        return self == other

    def __hash__(self):
        return hash(self._resolution) + hash(self._bounds)

    def __repr__(self):
        return f"{self._resolution}, bounds={self._bounds}"

    def __variable_attrs__(self):
        return ()

    def __with_attrs__(self, **attrs):
        if not attrs:
            return self
        else:
            raise NotImplementedError

    @property
    def _center(self):
        return self.center

    @property
    def _half_size(self):
        return self.half_size

    @property
    def normal(self) -> Tensor:
        raise GeometryException("UniformGrid does not have normals")


# class UniformGridFaces(Geometry):
#
#     def __init__(self, grid: UniformGrid, include_boundaries: Dict[str, Tuple[bool, bool]], staggered_dim: Shape):
#         self._grid = grid
#         self._include_boundaries = include_boundaries
#         self._staggered_dim = staggered_dim
#
#     @property
#     def resolution(self):
#         return self._grid.resolution
#
#     @property
#     def bounds(self):
#         return self._grid.bounds
#
#     def __getitem__(self, item):
#         item: dict = slicing_dict(self, item)
#         if not item:
#             return self
#         grid = self._grid[item]
#         staggered_dim = self._staggered_dim.after_gather(item)
#         if staggered_dim:
#             remaining_dirs = staggered_dim.names
#         else:
#             sel = item[self._staggered_dim.name]
#             if isinstance(sel, int):
#                 remaining_dirs = [self._staggered_dim.item_names[0][sel]]
#             elif isinstance(sel, str):
#                 remaining_dirs = [sel]
#             else:
#                 raise AssertionError(f"selection must be str or int but got {type(sel)}")
#         include_boundaries = {dim: incl for dim, incl in self._include_boundaries.items() if dim in remaining_dirs}
#         return UniformGridFaces(grid, include_boundaries, staggered_dim)
#
#     @property
#     def center(self) -> Tensor:
#         result = {}
#         for dim in self._grid.shape.spatial.names:
#             lower, upper = self._include_boundaries[dim]
#             dim_mask = np.array(self.resolution.mask(dim))
#             unit = self.bounds.size / self.resolution * dim_mask
#             bounds = Box(self.bounds.lower + unit * (-0.5 if lower else 0.5), self.bounds.upper + unit * (0.5 if upper else -0.5))
#             ext_res = self.resolution.sizes + dim_mask * (int(lower) + int(upper) - 1)
#             result[dim] = UniformGrid(self.resolution.with_sizes(ext_res), bounds).center
#         return math.stack(result, self._staggered_dim)
#
#     @property
#     def normal(self) -> Tensor:
#         raise NotImplementedError
#
#     @property
#     def volume(self) -> Tensor:
#         raise NotImplementedError  # ToDo compute face area
#
#     def interior(self) -> 'Geometry':
#         return self._grid
#
#     def lies_inside(self, location: Tensor) -> Tensor:
#         raise NotImplementedError
#
#     def __hash__(self):
#         raise NotImplementedError
#
#     @property
#     def area(self) -> Tensor:
#         raise NotImplementedError
#
#     @property
#     def shape_type(self) -> Tensor:
#         raise NotImplementedError
#
#     def approximate_signed_distance(self, location: Tensor or tuple) -> Tensor:
#         raise NotImplementedError
#
#     def push(self, positions: Tensor, outward: bool = True, shift_amount: float = 0) -> Tensor:
#         raise NotImplementedError
#
#     def sample_uniform(self, *shape: math.Shape) -> Tensor:
#         raise NotImplementedError
#
#     def bounding_radius(self) -> Tensor:
#         raise NotImplementedError
#
#     def bounding_half_extent(self) -> Tensor:
#         raise NotImplementedError
#
#     def at(self, center: Tensor) -> 'Geometry':
#         raise NotImplementedError
#
#     def rotated(self, angle: float or Tensor) -> 'Geometry':
#         raise NotImplementedError
#
#     def scaled(self, factor: float or Tensor) -> 'Geometry':
#         raise NotImplementedError
#
#     def surface(self, include_boundaries):
#         raise NotImplementedError
#
#     def __variable_attrs__(self):
#         return ()
