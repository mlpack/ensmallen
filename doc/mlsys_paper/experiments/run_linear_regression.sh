#!/bin/sh
#
# Run L-BFGS linear regression simulations.

for n in 1000 10000 100000;
do
  echo "data size $n points, 100 dims";
  echo -n "ensmallen: ";
  ./linear_regression_two_functions 100 $n | head -1
  echo -n "ensmallen, EvaluateWithGradient(): ";
  ./linear_regression 100 $n | head -1
  #echo -n "julia, default autodiff: "
  #julia linear_regression.jl 100 $n | tail -1
  echo -n "julia, forwarddiff package: "
  julia linear_regression_autodiff.jl 100 $n | tail -1
  echo -n "julia, gradient implemented: "
  julia linear_regression_with_gradient.jl 100 $n | tail -1
  echo -n "scipy: "
  python linear_regression.py 100 $n | head -1
  echo -n "autograd: "
  python linear_regression_autograd.py 100 $n | head -1
  echo -n "octave: "
  octave-cli linear_regression.m 100 $n | tail -1 | sed 's/Elapsed time is //'
  echo ""
done

echo "data size 100000 points, 1000 dims";
echo -n "ensmallen: ";
./linear_regression_two_functions 1000 100000 | head -1
echo -n "ensmallen, EvaluateWithGradient(): ";
./linear_regression 1000 100000 | head -1
#echo -n "julia, default autodiff: "
#julia linear_regression.jl 1000 100000 | tail -1
echo -n "julia, forwarddiff package: "
julia linear_regression_autodiff.jl 1000 100000 | tail -1
echo -n "julia, gradient implemented: "
julia linear_regression_with_gradient.jl 1000 100000 | tail -1
echo -n "scipy: "
python linear_regression.py 1000 100000 | head -1
echo -n "autograd: "
python linear_regression_autograd.py 1000 100000 | head -1
echo -n "octave: "
octave-cli linear_regression.m 1000 100000 | tail -1 | sed 's/Elapsed time is //'
