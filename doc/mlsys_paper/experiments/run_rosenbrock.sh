#!/bin/sh
#
# Run simulated annealing Rosenbrock function simulations.

echo -n "ensmallen: "
./rosenbrock | head -1
echo -n "julia: "
julia rosenbrock.jl | tail -1
echo -n "scipy: "
python rosenbrock.py | head -1
echo -n "octave: "
octave-cli rosenbrock.m | tail -1 | sed 's/Elapsed time is //'
echo -n "scipy-numba: "
numba rosenbrock_numba.py | head -1
echo -n "octave-mex: "
octave-cli rosenbrock_mex.m | tail -1 | sed 's/Elapsed time is //'
