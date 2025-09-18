/**
 * @file fasta_test.cpp
 * @author Ryan Curtin
 *
 * Tests for FASTA (Fast Adaptive Shrinkage/Thresholding Algorithm).
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

TEMPLATE_TEST_CASE("FASTASimpleTest", "[FASTA]", ENS_TEST_TYPES)
{
  // Make sure that we can get a decent result with no g(x) constraint.
  FASTA<L1Penalty> fasta(L1Penalty(0.0), 5000);
  GeneralizedRosenbrockFunction f(20);
  FunctionTest<GeneralizedRosenbrockFunction, TestType>(fasta, f,
      100 * Tolerances<TestType>::Obj,
      100 * Tolerances<TestType>::Coord);
}

TEMPLATE_TEST_CASE("FASTASphereFunctionTest", "[FASTA]", ENS_ALL_TEST_TYPES,
    ENS_SPARSE_TEST_TYPES)
{
  // The sphere function optimizes to the origin anyway, so the L1 penalty does
  // not affect the result.
  FASTA<L1Penalty> fasta(L1Penalty(0.1));
  FunctionTest<SphereFunction, TestType>(fasta,
      Tolerances<TestType>::Obj,
      Tolerances<TestType>::Coord);
}

// FASTA has step size issues on the Wood function with arma::fmat, so we don't
// test with that type.
TEMPLATE_TEST_CASE("FASTAWoodFunctionTest", "[FASTA]", arma::mat)
{
  // Set the L1 constraint to be sufficiently large that the final solution is
  // just inside the ball.
  FASTA<L1Constraint> fasta(L1Constraint(5.1));
  fasta.Tolerance() = 1e-10; // This converges too early otherwise.

  // Optimization can sometimes diverge, so we allow a few trials.
  FunctionTest<WoodFunction, TestType>(fasta,
      Tolerances<TestType>::LargeObj,
      Tolerances<TestType>::LargeCoord,
      5);
}

TEMPLATE_TEST_CASE("FASTALogisticRegressionFunctionTest", "[FASTA]",
    ENS_TEST_TYPES) // low precision is too flaky for this test
{
  FASTA<L1Penalty> fasta(L1Penalty(0.001));
  LogisticRegressionFunctionTest<TestType>(fasta,
      Tolerances<TestType>::LRTrainAcc,
      Tolerances<TestType>::LRTestAcc,
      5);
}

// Check that maxIterations does anything.
TEST_CASE("FASTAMaxIterationsTest", "[FASTA]")
{
  FASTA<L1Penalty> fasta1(L1Penalty(0.001)), fasta2(L1Penalty(0.001));
  fasta1.MaxIterations() = 5;
  fasta2.MaxIterations() = 100;

  BoothFunction f;
  mat coordinates1 = f.GetInitialPoint<mat>();
  mat coordinates2 = coordinates1;

  fasta1.Optimize(f, coordinates1);
  fasta2.Optimize(f, coordinates2);

  // The second optimization should have proceeded further.
  REQUIRE(f.Evaluate(coordinates1) >= f.Evaluate(coordinates2));
}

// Check that the step size estimate works at least reasonably.
TEST_CASE("FASTAStepSizeEstimateTest", "[FASTA]")
{
  QuadraticFunction f;
  FASTA<L1Penalty> fasta1;

  mat coordinates1 = f.GetInitialPoint<mat>();
  fasta1.Optimize(f, coordinates1);

  // Check the step size to ensure that it's reasonable.  If the guess is
  // perfect, then the step size will be 10 / L, where L = 1.
  REQUIRE(fasta1.MaxStepSize() >= 1.0);
  REQUIRE(fasta1.MaxStepSize() <= 11.0);
}

// Check what happens when 0 estimate trials are used.
TEST_CASE("FASTAZeroEstimateTrialsTest", "[FASTA]")
{
  QuadraticFunction f;
  FASTA<L1Penalty> fasta1;
  REQUIRE_THROWS(fasta1 = FASTA<L1Penalty>(1000, 1e-10, 50, 2.0, 10, true, 0));

  FASTA<L1Penalty> fasta2;
  fasta2.EstimateTrials() = 0;

  mat coordinates = f.GetInitialPoint<mat>();
  REQUIRE_THROWS(fasta2.Optimize(f, coordinates));
}

// Check what happens when the step size is manually set to something much
// smaller than the estimate would give.
TEST_CASE("FASTATooSmallManualStepSizeTest", "[FASTA]")
{
  QuadraticFunction f;
  FASTA<L1Penalty> fasta(L1Penalty(0.0), 1000, 1e-10, 50, 2.0, 10, false, 10,
      1e-10);

  mat coordinates = f.GetInitialPoint<mat>();
  fasta.Optimize(f, coordinates);

  // We should be far away from the optimum.
  REQUIRE(std::abs(coordinates[0]) >= 1.0);
}

// Check that we can converge even when a gigantic manual maximum step size is
// specified.
TEST_CASE("FASTATooLargeManualStepSizeTest", "[FASTA]")
{
  // Use a huge step size.  We should still successfully optimize the function.
  FASTA<L1Penalty> fasta(L1Penalty(0.0), 10000, 1e-10, 50, 2.0, 10, false, 10,
      10000.0);
  GeneralizedRosenbrockFunction f(20);
  FunctionTest<GeneralizedRosenbrockFunction, mat>(fasta, f);
}

// Check that we can't specify a lineSearchLookback of 0.
TEST_CASE("FASTAInvalidLineSearchLookbackTest", "[FASTA]")
{
  QuadraticFunction f;
  FASTA<L1Penalty> fasta1;
  REQUIRE_THROWS(fasta1 = FASTA<L1Penalty>(1000, 1e-10, 50, 2.0, 0));

  FASTA<L1Penalty> fasta2;
  fasta2.LineSearchLookback() = 0;

  mat coordinates = f.GetInitialPoint<mat>();
  REQUIRE_THROWS(fasta2.Optimize(f, coordinates));
}
