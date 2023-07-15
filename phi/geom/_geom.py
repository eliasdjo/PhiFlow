import warnings
from numbers import Number
from typing import Union, Dict, Any, Tuple

from phi import math
from phi.math import Tensor, Shape, EMPTY_SHAPE, non_channel, wrap, shape, Extrapolation
from phiml.math._magic_ops import variable_attributes, expand
from phi.math.magic import BoundDim, slicing_dict


class Geometry:
    """
    Abstract base class for N-dimensional shapes.

    Main implementing classes:

    * Sphere
    * box family: box (generator), Box, Cuboid, BaseBox

    All geometry objects support batching.
    Thereby any parameter defining the geometry can be varied along arbitrary batch dims.
    All batch dimensions are listed in Geometry.shape.
    """

    @property
    def center(self) -> Tensor:
        """
        Center location in single channel dimension.
        """
        raise NotImplementedError(self.__class__)

    @property
    def shape(self) -> Shape:
        """
        The `shape` of a `Geometry` consists of the following dimensions:

        * A single *channel* dimension called `'vector'` specifying the physical space
        * Instance dimensions denote that this geometry consists of multiple copies in the same space
        * Spatial dimensions denote a crystal (repeating structure) of this geometric primitive in space
        * Batch dimensions indicate non-interacting versions of this geometry for parallelization only.
        """
        raise NotImplementedError(self.__class__)

    @property
    def volume(self) -> Tensor:
        """
        `phi.math.Tensor` representing the volume of each element.
        The result retains batch, spatial and instance dimensions.
        """
        raise NotImplementedError(self.__class__)

    @property
    def face_centers(self) -> Tensor:
        """
        Center of face connecting a pair of cells. Shape `(elements, ~, vector)`.
        Here, `~` represents arbitrary internal dual dimensions, such as `~staggered_direction` or `~elements`.
        Returns 0-vectors for unconnected cells.
        """
        raise NotImplementedError(self.__class__)

    @property
    def face_areas(self) -> Tensor:
        """
        Area of face connecting a pair of cells. Shape `(elements, ~)`.
        Returns 0 for unconnected cells.
        """
        raise NotImplementedError(self.__class__)

    @property
    def face_normals(self) -> Tensor:
        """
        Normal vector of face connecting a pair of cells. Shape `(elements, ~, vector)`.
        Unconnected cells are assigned the vector 0.
        The vector points out of cells and into ~.
        """
        raise NotImplementedError(self.__class__)

    @property
    def boundary_elements(self) -> Dict[str, Tuple[Dict[str, slice], Dict[str, slice]]]:
        """
        Slices on the primal dimensions to mark boundary elements.
        Grids and meshes have no boundary elements and return `{}`.
        Dynamic graphs can define boundary elements for obstacles and walls.

        Returns:
            Map from `(name, is_upper)` to slicing `dict`.
        """
        raise NotImplementedError(self.__class__)

    @property
    def boundary_faces(self) -> Dict[str, Tuple[Dict[str, slice], Dict[str, slice]]]:
        """
        Slices on the dual dimensions to mark boundary faces.

        Regular grids use the keys (dim, is_upper) to identify boundaries.
        Unstructured meshes use string identifiers for the boundaries.
        Dynamic graphs return slices along the dual dimensions.

        Returns:
            Map from `(name, is_upper)` to slicing `dict`.
        """
        raise NotImplementedError(self.__class__)

    @property
    def face_shape(self) -> Shape:
        """
        Returns:
            Full Shape to identify each face of this `Geometry`, including instance/spatial dimensions for the elements and dual dimensions listing the faces per element.
            If this `Geometry` has no faces, returns an empty `Shape`.
        """
        raise NotImplementedError(self.__class__)

    @property
    def shape_type(self) -> Tensor:
        """
        Returns the type (or types) of this geometry as a string `Tensor`
        Boxes return `'B'`, spheres return `'S'`, points return `'P'`.
        Returns `'?'` for unknown types, e.g. a union over multiple types.
        Custom types can return their own identifiers.

        Returns:
            String `Tensor`
        """
        raise NotImplementedError(self.__class__)  # ToDo deprecate this

    def integrate_surface(self, values: Tensor) -> Tensor:
        """
        Multiplies `values´ by the corresponding face area, computes the sum over all faces and divides by the cell volume.
        ∑ values * A / V.

        Args:
            values: Values sampled at the face centers.

        Returns:
            `Tensor` of values sampled at the centroids.
        """
        return math.sum(values * self.face_areas, self.face_shape.dual) / self.volume

    # def resample_to_faces(self, values: Tensor, boundary: Extrapolation, **kwargs):
    #     raise NotImplementedError(self.__class__)
    #
    # def resample_to_centers(self, values: Tensor, boundary: Extrapolation, **kwargs):
    #     raise NotImplementedError(self.__class__)
    #
    # def centered_gradient_of(self, values: Tensor, boundary: Extrapolation, dims=None, **kwargs):
    #     raise NotImplementedError(self.__class__)
    #
    # def staggered_gradient_of(self, values: Tensor, boundary: Extrapolation, dims=None, **kwargs):
    #     raise NotImplementedError(self.__class__)
    #
    # def divergence_of(self, values: Tensor, boundary: Extrapolation, dims=None, **kwargs):
    #     raise NotImplementedError(self.__class__)
    #
    # def laplace_of(self, values: Tensor, boundary: Extrapolation, dims=None, **kwargs):
    #     raise NotImplementedError(self.__class__)
    #
    # def centered_curl_of(self, values: Tensor, boundary: Extrapolation, dims=None, **kwargs):
    #     raise NotImplementedError(self.__class__)
    #
    # def staggered_curl_of(self, values: Tensor, boundary: Extrapolation, dims=None, **kwargs):
    #     raise NotImplementedError(self.__class__)

    def unstack(self, dimension: str) -> tuple:
        """
        Unstacks this Geometry along the given dimension.
        The shapes of the returned geometries are reduced by `dimension`.

        Args:
            dimension: dimension along which to unstack

        Returns:
            geometries: tuple of length equal to `geometry.shape.get_size(dimension)`
        """
        warnings.warn(f"Geometry.unstack() is deprecated. Use math.unstack(geometry) instead.", DeprecationWarning)
        return math.unstack(self, dimension)

    @property
    def spatial_rank(self) -> int:
        """ Number of spatial dimensions of the geometry, 1 = 1D, 2 = 2D, 3 = 3D, etc. """
        return self.shape.get_size('vector')

    def lies_inside(self, location: Tensor) -> Tensor:
        """
        Tests whether the given location lies inside or outside of the geometry. Locations on the surface count as inside.

        When dealing with unions or collections of geometries (instance dimensions), a point lies inside the geometry if it lies inside any instance.

        Args:
          location: float tensor of shape (batch_size, ..., rank)

        Returns:
          bool tensor of shape (*location.shape[:-1], 1).

        """
        raise NotImplementedError(self.__class__)

    def approximate_signed_distance(self, location: Union[Tensor, tuple]) -> Tensor:
        """
        Computes the approximate distance from location to the surface of the geometry.
        Locations outside return positive values, inside negative values and zero exactly at the boundary.

        The exact distance metric used depends on the geometry.
        The approximation holds close to the surface and the distance grows to infinity as the location is moved infinitely far from the geometry.
        The distance metric is differentiable and its gradients are bounded at every point in space.

        When dealing with unions or collections of geometries (instance dimensions), the shortest distance to any instance is returned.
        This also holds for negative distances.

        Args:
          location: float tensor of shape (batch_size, ..., rank)
          location: Tensor:

        Returns:
          float tensor of shape (*location.shape[:-1], 1).

        """
        raise NotImplementedError(self.__class__)

    def approximate_fraction_inside(self, other_geometry: 'Geometry', balance: Union[Tensor, Number] = 0.5) -> Tensor:
        """
        Computes the approximate overlap between the geometry and a small other geometry.
        Returns 1.0 if `other_geometry` is fully enclosed in this geometry and 0.0 if there is no overlap.
        Close to the surface of this geometry, the fraction filled is differentiable w.r.t. the location and size of `other_geometry`.

        To call this method on batches of geometries of same shape, pass a batched Geometry instance.
        The result tensor will match the batch shape of `other_geometry`.

        The result may only be accurate in special cases.
        The given geometries may be approximated as spheres or boxes using `bounding_radius()` and `bounding_half_extent()`.

        The default implementation of this method approximates other_geometry as a Sphere and computes the fraction using `approximate_signed_distance()`.

        Args:
            other_geometry: `Geometry` or geometry batch for which to compute the overlap with `self`.
            balance: Mid-level between 0 and 1, default 0.5.
                This value is returned when exactly half of `other_geometry` lies inside `self`.
                `0.5 < balance <= 1` makes `self` seem larger while `0 <= balance < 0.5`makes `self` seem smaller.

        Returns:
          fraction of cell volume lying inside the geometry. float tensor of shape (other_geometry.batch_shape, 1).

        """
        assert isinstance(other_geometry, Geometry)
        radius = other_geometry.bounding_radius()
        location = other_geometry.center
        distance = self.approximate_signed_distance(location)
        inside_fraction = balance - distance / radius
        inside_fraction = math.clip(inside_fraction, 0, 1)
        return inside_fraction

    def push(self, positions: Tensor, outward: bool = True, shift_amount: float = 0) -> Tensor:
        """
        Shifts positions either into or out of geometry.

        Args:
            positions: Tensor holding positions to shift
            outward: Flag for indicating inward (False) or outward (True) shift
            shift_amount: Minimum distance between positions and box boundaries after shifting

        Returns:
            Tensor holding shifted positions
        """
        raise NotImplementedError(self.__class__)

    def sample_uniform(self, *shape: math.Shape) -> Tensor:
        """
        Samples uniformly distributed random points inside this volume.

        Args:
            *shape: How many points to sample per individual geometry.

        Returns:
            `Tensor` containing all dimensions from `Geometry.shape`, `shape` as well as a `channel` dimension `vector` matching the dimensionality of this `Geometry`.
        """
        raise NotImplementedError(self.__class__)

    def bounding_radius(self) -> Tensor:
        """
        Returns the radius of a Sphere object that fully encloses this geometry.
        The sphere is centered at the center of this geometry.

        :return: radius of type float

        Args:

        Returns:

        """
        raise NotImplementedError(self.__class__)

    def bounding_half_extent(self) -> Tensor:
        """
        The bounding half-extent sets a limit on the outer-most point for each coordinate axis.
        Each component is non-negative.

        Let the bounding half-extent have value `e` in dimension `d` (`extent[...,d] = e`).
        Then, no point of the geometry lies further away from its center point than `e` along `d` (in both axis directions).

        :return: float vector

        Args:

        Returns:

        """
        raise NotImplementedError(self.__class__)

    def bounding_box(self) -> 'BaseBox':
        """
        Returns the approximately smallest axis-aligned box that contains this `Geometry`.
        The center of the box may not be equal to `self.center`.

        Returns:
            `Box` or `Cuboid` that fully contains this `Geometry`.
        """
        from ._box import Cuboid
        return Cuboid(self.center, half_size=self.bounding_half_extent())

    def shifted(self, delta: Tensor) -> 'Geometry':
        """
        Returns a translated version of this geometry.

        See Also:
            `Geometry.at()`.

        Args:
          delta: direction vector
          delta: Tensor:

        Returns:
          Geometry: shifted geometry

        """
        return self.at(self.center + delta)

    def at(self, center: Tensor) -> 'Geometry':
        """
        Returns a copy of this `Geometry` with the center at `center`.
        This is equal to calling `self @ center`.

        See Also:
            `Geometry.shifted()`.

        Args:
            center: New center as `Tensor`.

        Returns:
            `Geometry`.
        """
        raise NotImplementedError(self.__class__)

    def __matmul__(self, other):
        return self.at(other)

    def rotated(self, angle: Union[float, Tensor]) -> 'Geometry':
        """
        Returns a rotated version of this geometry.
        The geometry is rotated about its center point.

        Args:
          angle: scalar (2d) or vector (3D+) representing delta angle

        Returns:
            Rotated `Geometry`
        """
        raise NotImplementedError(self.__class__)

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        """
        Scales each individual geometry by `factor`.
        The individual `center` points act as pivots for the operation.

        Args:
            factor:

        Returns:

        """
        raise NotImplementedError(self.__class__)

    def __invert__(self):
        return _InvertedGeometry(self)

    def __eq__(self, other):
        """
        Slow equality check.
        Unlike `==`, this method compares all tensor elements to check whether they are equal.
        Use `==` for a faster check which only checks whether the referenced tensors are the same.

        See Also:
            `shallow_equals()`
        """
        if self is other:
            return True
        if not isinstance(other, type(self)):
            return False
        if self.shape != other.shape:
            return False
        c1 = {a: getattr(self, a) for a in variable_attributes(self)}
        c2 = {a: getattr(other, a) for a in variable_attributes(self)}
        for c in c1.keys():
            if c1[c] is not c2[c] and math.any(c1[c] != c2[c]):
                return False
        return True

    def shallow_equals(self, other):
        """
        Quick equality check.
        May return `False` even if `other == self`.
        However, if `True` is returned, the geometries are guaranteed to be equal.

        The `shallow_equals()` check does not compare all tensor elements but merely checks whether the same tensors are referenced.
        """
        if self is other:
            return True
        if not isinstance(other, type(self)):
            return False
        if self.shape != other.shape:
            return False
        c1 = {a: getattr(self, a) for a in variable_attributes(self)}
        c2 = {a: getattr(other, a) for a in variable_attributes(self)}
        for c in c1.keys():
            if c1[c] is not c2[c]:
                return False
        return True

    @staticmethod
    def __stack__(values: tuple, dim: Shape, **kwargs) -> 'Geometry':
        if all(type(v) == type(values[0]) for v in values):
            return NotImplemented  # let attributes be stacked
        else:
            from ._stack import GeometryStack
            return GeometryStack(math.layout(values, dim))

    def __flatten__(self, flat_dim: Shape, flatten_batch: bool, **kwargs) -> 'Geometry':
        dims = self.shape.without('vector')
        if not flatten_batch:
            dims = dims.non_batch
        return math.pack_dims(self, dims, flat_dim, **kwargs)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        raise NotImplementedError(self.__class__)

    def __repr__(self):
        return f"{self.__class__.__name__}{self.shape}"

    def __getitem__(self, item):
        raise NotImplementedError
        # assert isinstance(item, dict), "Index must be dict of type {dim: slice/int}."
        # item = {dim: sel for dim, sel in item.items() if dim != 'vector'}
        # attrs = {a: getattr(self, a)[item] for a in variable_attributes(self)}
        # return copy_with(self, **attrs)

    def __getattr__(self, name: str) -> BoundDim:
        return BoundDim(self, name)


class _InvertedGeometry(Geometry):

    def __init__(self, geometry):
        self.geometry = geometry

    @property
    def volume(self) -> Tensor:
        return math.wrap(math.INF)

    @property
    def shape_type(self) -> Tensor:
        raise NotImplementedError

    def sample_uniform(self, *shape: math.Shape) -> Tensor:
        raise NotImplementedError

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        return _InvertedGeometry(self.geometry.scaled(factor))

    def __getitem__(self, item: dict):
        return _InvertedGeometry(self.geometry[item])

    @property
    def center(self):
        raise NotImplementedError

    @property
    def shape(self):
        return self.geometry.shape

    def lies_inside(self, location: Tensor) -> Tensor:
        return ~self.geometry.lies_inside(location)

    def approximate_signed_distance(self, location: Tensor) -> Tensor:
        return -self.geometry.approximate_signed_distance(location)

    def approximate_fraction_inside(self, other_geometry: 'Geometry', balance: Union[Tensor, Number] = 0.5) -> Tensor:
        return 1 - self.geometry.approximate_fraction_inside(other_geometry, 1 - balance)

    def push(self, positions: Tensor, outward: bool = True, shift_amount: float = 0) -> Tensor:
        return self.geometry.push(positions, outward=not outward, shift_amount=shift_amount)

    def bounding_radius(self) -> Tensor:
        raise NotImplementedError()

    def bounding_half_extent(self) -> Tensor:
        raise NotImplementedError()

    def at(self, center: Tensor) -> 'Geometry':
        return _InvertedGeometry(self.geometry.at(center))

    def rotated(self, angle) -> Geometry:
        return _InvertedGeometry(self.geometry.rotated(angle))

    def unstack(self, dimension):
        return [_InvertedGeometry(g) for g in math.unstack(self.geometry, dimension)]

    def __eq__(self, other):
        return isinstance(other, _InvertedGeometry) and self.geometry == other.geometry

    def __hash__(self):
        return -hash(self.geometry)

    @property
    def normal(self) -> Tensor:
        return -self.geometry.normal


def invert(geometry: Geometry):
    """
    Swaps inside and outside.

    Args:
        geometry: `phi.geom.Geometry` to swap

    Returns:
        New `phi.geom.Geometry` object with same surface but swapped normals
    """
    return ~geometry


class _NoGeometry(Geometry):

    @property
    def shape_type(self) -> Tensor:
        raise NotImplementedError

    def push(self, positions: Tensor, outward: bool = True, shift_amount: float = 0) -> Tensor:
        return positions

    def sample_uniform(self, *shape: math.Shape) -> Tensor:
        raise NotImplementedError

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        return self

    def __getitem__(self, item: dict):
        return self

    @property
    def shape(self):
        return EMPTY_SHAPE

    @property
    def volume(self) -> Tensor:
        return wrap(0)

    @property
    def center(self) -> Tensor:
        return wrap(0)

    def bounding_radius(self) -> Tensor:
        return wrap(0)

    def bounding_half_extent(self) -> Tensor:
        return wrap(0)

    def approximate_signed_distance(self, location):
        return math.expand(math.INF, non_channel(location))

    def lies_inside(self, location):
        return math.zeros(non_channel(location), dtype=bool)

    def approximate_fraction_inside(self, other_geometry: 'Geometry', balance: Union[Tensor, Number] = 0.5) -> Tensor:
        return math.zeros(other_geometry.shape)

    def at(self, center: Tensor) -> 'Geometry':
        return self

    def rotated(self, angle):
        return self

    def unstack(self, dimension):
        raise AssertionError('empty geometry cannot be unstacked')

    def __eq__(self, other):
        return isinstance(other, _NoGeometry)

    def __hash__(self):
        return 1

    @property
    def normal(self) -> Tensor:
        raise GeometryException("Empty geometry does not have normals")

    def surface(self, include_boundaries: Union[bool, Dict[str, Any]]) -> 'Geometry':
        raise GeometryException("Empty geometry does not have a surface")

    def interior(self) -> 'Geometry':
        raise GeometryException("Empty geometry does not have an interior")


NO_GEOMETRY = _NoGeometry()


class Point(Geometry):
    """
    Points have zero volume and are determined by a single location.
    An instance of `Point` represents a single n-dimensional point or a batch of points.
    """

    def __init__(self, location: math.Tensor):
        assert 'vector' in location.shape, "location must have a vector dimension"
        assert location.shape.get_item_names('vector') is not None, "Vector dimension needs to list spatial dimension as item names."
        self._location = location

    @property
    def center(self) -> Tensor:
        return self._location

    @property
    def shape(self) -> Shape:
        return self._location.shape

    def unstack(self, dimension: str) -> tuple:
        return tuple(Point(loc) for loc in math.unstack(self._location, dimension))

    def lies_inside(self, location: Tensor) -> Tensor:
        return expand(math.wrap(False), shape(location).without('vector'))

    def approximate_signed_distance(self, location: Union[Tensor, tuple]) -> Tensor:
        return math.vec_abs(location - self._location)

    def push(self, positions: Tensor, outward: bool = True, shift_amount: float = 0) -> Tensor:
        return positions

    def bounding_radius(self) -> Tensor:
        return math.zeros()

    def bounding_half_extent(self) -> Tensor:
        return math.zeros()

    def at(self, center: Tensor) -> 'Geometry':
        return Point(center)

    def rotated(self, angle) -> 'Geometry':
        return self

    def __hash__(self):
        return hash(self._location)

    def __variable_attrs__(self):
        return '_location',

    @property
    def volume(self) -> Tensor:
        return math.wrap(0)

    @property
    def shape_type(self) -> Tensor:
        return math.tensor('P')

    def sample_uniform(self, *shape: math.Shape) -> Tensor:
        raise NotImplementedError

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        return self

    @property
    def face_centers(self) -> Tensor:
        raise NotImplementedError

    @property
    def face_areas(self) -> Tensor:
        raise NotImplementedError

    @property
    def face_normals(self) -> Tensor:
        raise NotImplementedError

    @property
    def boundary_elements(self) -> Dict[str, Tuple[Dict[str, slice], Dict[str, slice]]]:
        return {}

    @property
    def boundary_faces(self) -> Dict[str, Tuple[Dict[str, slice], Dict[str, slice]]]:
        raise NotImplementedError

    @property
    def face_shape(self) -> Shape:
        return math.EMPTY_SHAPE

    def __getitem__(self, item):
        return Point(self._location[_keep_vector(slicing_dict(self, item))])


class GeometryException(BaseException):
    """
    Raised when an operation is fundamentally not possible for a `Geometry`.
    Possible causes:

    * Trying to get the interior of a non-surface `Geometry`
    * Trying to get the surface of a point-like `Geometry`
    """


def assert_same_rank(rank1, rank2, error_message):
    """ Tests that two objects have the same spatial rank. Objects can be of types: `int`, `None` (no check), `Geometry`, `Shape`, `Tensor` """
    rank1_, rank2_ = _rank(rank1), _rank(rank2)
    if rank1_ is not None and rank2_ is not None:
        assert rank1_ == rank2_, 'Ranks do not match: %s and %s. %s' % (rank1_, rank2_, error_message)


def _rank(rank):
    if rank is None:
        return None
    elif isinstance(rank, int):
        pass
    elif isinstance(rank, Geometry):
        rank = rank.spatial_rank
    elif isinstance(rank, Shape):
        rank = rank.spatial.rank
    elif isinstance(rank, Tensor):
        rank = rank.shape.spatial_rank
    else:
        raise NotImplementedError(f"{type(rank)} now allowed. Allowed are (int, Geometry, Shape, Tensor).")
    return None if rank == 0 else rank


def _keep_vector(dim_selection: dict) -> dict:
    if 'vector' not in dim_selection:
        return dim_selection
    item = dict(dim_selection)
    if isinstance(item['vector'], int) or (isinstance(item['vector'], str) and ',' not in item['vector']):
        item['vector'] = (item['vector'],)
    return item
