"""
@file linear_regression.py
@author Ryan Curtin

Use scipy to optimize the linear regression objective function using L-BFGS and
automatic differentation.
"""
import autograd.numpy as np
from autograd import elementwise_grad as egrad
import scipy.optimize
import random
import sys

if len(sys.argv) != 3:
  print("args: <dim> <points>")
  exit(1)

dim = int(sys.argv[1])
points = int(sys.argv[2])

x = np.random.rand(points, dim)
y = np.random.rand(points)

for i in range(points):
  a = random.random()
  x[i, 1] = x[i, 1] + a
  y[i] = y[i] + a

def f(theta):
  v = (y - np.matmul(x, theta))
  return np.sum(np.multiply(v, v))

# Compute gradient.
g = egrad(f)

theta = np.random.rand(dim)

import timeit

def run():
  # Try to match mlpack configuration.
  return scipy.optimize.fmin_l_bfgs_b(f, theta, fprime=g, maxiter=10, maxls=50)

t = timeit.timeit(stmt=run, number=1)
print(t)
