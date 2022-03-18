from phi.jax.flow import *
from phi.math.extrapolation import PERIODIC


test = tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


def test_f(a, shift):
    return math.shift(a, (shift, ), "vector", PERIODIC, None)[0] + math.shift(a, (-shift, ), "vector", PERIODIC, None)[0]


# nicht so wichtig da die obere Zeile 13 ja funktioniert aber evtl. Fehlerhaft?
# print(solve_linear(test_f, test, f_kwargs={"shift": 2}, solve=Solve('CG', 1e-5, 1e-5, x0=test)))
# print(solve_linear(test_f, test, f_args=[2], solve=Solve('CG', 1e-5, 1e-5, x0=test)))


def solve(a):
    return solve_linear(test_f, a, f_kwargs={"shift": 2},
                        solve=Solve('CG', 1e-5, 1e-5, x0=test))

solve_jit = jit_compile_linear(solve)
print(solve(test))
print(solve_jit(test)) # shift argument kommt in Zeile 9 als Tracer an