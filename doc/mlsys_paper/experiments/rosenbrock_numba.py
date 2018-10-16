"""
@file rosenbrock.py
@author Ryan Curtin

Use simulated annealing to optimize the Rosenbrock function with
scipy.optimize.anneal.
"""
from numba import jit
import anneal_numba as anneal
import timeit

@jit(nopython=True)
def rosen(x):
  return 100 * (pow(x[1] - pow(x[0], 2), 2)) + pow(1 - x[0], 2)

# Run the optimization.  Note that I had to select parameters in such a way that
# it actually took the whole 100k function evaluations (just like mlpack does).
# That shouldn't actually affect the comparison from an angle that we care about
# (which is the overhead of calling the function, etc.).
@jit(nopython=False, parallel=True)
def run():
  return anneal.anneal(rosen, [-1.2, 1], (), 'cauchy', 0, 10.0, 1e-12, 100000,
None, 100000, 1.0, 0.5, 0.0, 1.0, 1.0, 1.0, -100, 100, 50, True)

# Break in the JIT...
run()
run()
run()

t = timeit.timeit(stmt=run, number=1)
print(t)
print(run())
