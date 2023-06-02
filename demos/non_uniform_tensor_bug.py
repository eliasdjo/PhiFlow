from phi.flow import *

test_1 = [[[0] * 6, [0] * 5], [[0] * 2, [0] * 3]]
test_2 = [[[0] * 6, [0] * 5], [[0] * 6, [0] * 5]]
from copy import deepcopy
test_3 = deepcopy(test_2)
test = [test_1, test_2, test_3]
test_tensors = [tensor(t, batch('dim1', 'dim2', 'dim3')) for t in test]
test_tensor = tensor(test_tensors[:2], batch('temp'))
test_tensor = tensor(test_tensors, batch('temp')) # failed auf 2.4