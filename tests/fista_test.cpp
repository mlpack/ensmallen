/**
 * @file fista_test.cpp
 * @author Ryan Curtin
 *
 * Tests for FISTA (Fast Iterative Shrinkage-Thresholding Algorithm).
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <ensmallen.hpp>
#include "catch.hpp"
#include "test_function_tools.hpp"

using namespace arma;
using namespace ens;
using namespace ens::test;

TEMPLATE_TEST_CASE("FISTASimpleTest", "[FISTA]", float, double)
{
  typedef TestType eT;

  // Make sure that we can get a decent result with no g(x) constraint.
  FISTA<L1Penalty> fista(L1Penalty(0.0), 10000);
  GeneralizedRosenbrockFunction f(20);
  FunctionTest<GeneralizedRosenbrockFunction, Mat<eT>>(fista, f, 0.02, 0.002);
}

TEMPLATE_TEST_CASE("FISTASphereFunctionTest", "[FISTA]", fmat, mat, sp_mat)
{
  typedef TestType MatType;

  // The sphere function optimizes to the origin anyway, so the L1 penalty does
  // not affect the result.
  FISTA<L1Penalty> fista(L1Penalty(0.1));
  FunctionTest<SphereFunction, MatType>(fista);
}

// FISTA struggles with the fmat type on the Wood function, so we skip it.  Even
// a regular matrix can be a bit tricky so we allow a few trials.
TEMPLATE_TEST_CASE("FISTAWoodFunctionTest", "[FISTA]", mat)
{
  typedef TestType MatType;

  // Set the L1 constraint to be sufficiently large that the final solution is
  // just inside the ball.
  FISTA<L1Constraint> fista(L1Constraint(5.2));
  fista.Tolerance() = 1e-10;
  FunctionTest<WoodFunction, MatType>(fista, 0.01, 0.001, 3);
}

TEMPLATE_TEST_CASE("FISTALogisticRegressionFunctionTest", "[FISTA]", fmat,
    mat)
{
  typedef TestType MatType;

  FISTA<L1Penalty> fista(L1Penalty(0.001));
  LogisticRegressionFunctionTest<MatType>(fista, 0.05, 0.05, 5);
}

// Check that maxIterations does anything.
TEST_CASE("FISTAMaxIterationsTest", "[FISTA]")
{
  FISTA<L1Penalty> fista1(L1Penalty(0.001)), fista2(L1Penalty(0.001));
  fista1.MaxIterations() = 5;
  fista2.MaxIterations() = 100;

  BoothFunction f;
  mat coordinates1 = f.GetInitialPoint<mat>();
  mat coordinates2 = coordinates1;

  fista1.Optimize(f, coordinates1);
  fista2.Optimize(f, coordinates2);

  // The second optimization should have proceeded further.
  REQUIRE(f.Evaluate(coordinates1) >= f.Evaluate(coordinates2));
}

// Check that the step size estimate works at least reasonably.
TEST_CASE("FISTAStepSizeEstimateTest", "[FISTA]")
{
  QuadraticFunction f;
  FISTA<L1Penalty> fista1;

  mat coordinates1 = f.GetInitialPoint<mat>();
  fista1.Optimize(f, coordinates1);

  // Check the step size to ensure that it's reasonable.  If the guess is
  // perfect, then the step size will be 10 / L, where L = 1.
  REQUIRE(fista1.MaxStepSize() >= 1.0);
  REQUIRE(fista1.MaxStepSize() <= 11.0);
}

// Check what happens when 0 estimate trials are used.
TEST_CASE("FISTAZeroEstimateTrialsTest", "[FISTA]")
{
  QuadraticFunction f;
  FISTA<L1Penalty> fista1;
  REQUIRE_THROWS(fista1 = FISTA<L1Penalty>(1000, 1e-10, 50, 2.0, true, 0));

  FISTA<L1Penalty> fista2;
  fista2.EstimateTrials() = 0;

  mat coordinates = f.GetInitialPoint<mat>();
  REQUIRE_THROWS(fista2.Optimize(f, coordinates));
}

// Check what happens when the step size is manually set to something much
// smaller than the estimate would give.
TEST_CASE("FISTATooSmallManualStepSizeTest", "[FISTA]")
{
  QuadraticFunction f;
  FISTA<L1Penalty> fista(L1Penalty(0.0), 1000, 1e-10, 50, 2.0, false, 10,
      1e-10);

  mat coordinates = f.GetInitialPoint<mat>();
  fista.Optimize(f, coordinates);

  // We should be far away from the optimum.
  REQUIRE(std::abs(coordinates[0]) >= 1.0);
}

// Check that we can converge even when a gigantic manual maximum step size is
// specified.
TEST_CASE("FISTATooLargeManualStepSizeTest", "[FISTA]")
{
  // Use a huge step size.  We should still successfully optimize the function.
  FISTA<L1Penalty> fista(L1Penalty(0.0), 1000, 1e-10, 50, 2.0, false, 10,
      10000.0);
  GeneralizedRosenbrockFunction f(20);
  FunctionTest<GeneralizedRosenbrockFunction, mat>(fista, f);
}
