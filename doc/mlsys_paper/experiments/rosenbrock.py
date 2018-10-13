"""
@file rosenbrock.py
@author Ryan Curtin

Use simulated annealing to optimize the Rosenbrock function with
scipy.optimize.anneal.
"""
import numpy as np
import anneal
import timeit

def rosen(x):
  return 100 * (pow(x[1] - pow(x[0], 2), 2)) + pow(1 - x[0], 2)

# Run the optimization.  Note that I had to select parameters in such a way that
# it actually took the whole 100k function evaluations (just like mlpack does).
# That shouldn't actually affect the comparison from an angle that we care about
# (which is the overhead of calling the function, etc.).
def run():
  return anneal.anneal(rosen, [-1.2, 1],
                       schedule='cauchy',
                       maxiter=100000,
                       maxeval=100000,
                       T0=10.0,
                       feps=0,
                       disp=True)

t = timeit.timeit(stmt=run, number=1)
print(t)
print(run())
