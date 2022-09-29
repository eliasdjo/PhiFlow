from phi.flow import *
from phi.math import Shape
from phi.math._functional import ShiftLinTracer

EMPTY_SHAPE = Shape((), (), (), ())

t_func = extrapolation.ANTIREFLECT.pad
mask_t_func = extrapolation.BOUNDARY.pad


def test_tracer(tensor, padding):
    tracer = ShiftLinTracer(math.ones(tensor.shape), {EMPTY_SHAPE: math.ones()}, tensor.shape, math.zeros(tensor.shape))
    result_tracer = t_func(tracer, padding)
    e1 = result_tracer.apply(tensor)

    mask_tracer = ShiftLinTracer(math.ones(tensor.shape), {EMPTY_SHAPE: math.ones()}, tensor.shape,
                                 math.zeros(tensor.shape))
    mask_result_tracer = mask_t_func(mask_tracer, padding)
    mask = mask_result_tracer.apply(math.ones(tensor.shape))

    if len(e1.shape.names) == 2:
        e1 = math.transpose(e1, (['x', 'y']))
    elif len(e1.shape.names) == 3:
        e1 = math.transpose(e1, (['x', 'y', 'z']))

    e2 = t_func(tensor, padding)
    e2 = e2 * mask

    plot(e1)
    show()
    plot(e2)
    show()

    assert (e1 == e2).all


tensors = [
    tensor((np.linspace(0, 49, 10)), spatial('x')),
    tensor((np.linspace(1, 12 * 21, 12 * 21).reshape(12, 21)), spatial('x'), spatial('y')),
    tensor((np.linspace(0, 49, 6000).reshape(10, 20, 30)), spatial('x'), spatial('y'), spatial('z'))
]

paddings = [
    {'x': (0, 0)},
    {'x': (3, 2)},
    {'x': (5, 5)},
    {'x': (3, 5), 'y': (2, 7)},
    {'x': (0, 0), 'y': (7, 7)},
    {'x': (8, 5), 'y': (4, 7)},
    {'x': (5, 7), 'y': (3, 2), 'z': (10, 6)}
]

for i, tensor in enumerate(tensors):
    for i2, padding in enumerate(paddings):
        if len(tensor.shape.names) >= len(padding.items()):
            print(f't: {i}  -   p: {i2}')
            test_tracer(tensor, padding)

print('done')
