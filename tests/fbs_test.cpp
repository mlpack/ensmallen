/**
 * @file fbs_test.cpp
 * @author Ryan Curtin
 *
 * Tests for FBS (forward-backward splitting).
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

TEMPLATE_TEST_CASE("FBSSimpleTest", "[FBS]", ENS_ALL_TEST_TYPES)
{
  // Make sure that we can get a decent result with no g(x) constraint.
  FBS<L1Penalty> fbs(L1Penalty(0.0), 0.001, 50000);
  GeneralizedRosenbrockFunction f(20);
  FunctionTest<GeneralizedRosenbrockFunction, TestType>(fbs, f,
      100 * Tolerances<TestType>::Obj,
      100 * Tolerances<TestType>::Coord);
}

// The L1 penalty backward step should have zero-valued g(x) and make no changes
// when the penalty is 0.
TEMPLATE_TEST_CASE("L1PenaltyZeroTest", "[FBS]", ENS_ALL_TEST_TYPES)
{
  L1Penalty l(0.0);

  TestType coordinates(100, 1, fill::randu);

  REQUIRE(std::abs(l.Evaluate(coordinates)) <= 1e-15);

  TestType coordinatesCopy(coordinates);
  l.ProximalStep(coordinates, 1.0);

  REQUIRE(all(all(coordinates == coordinatesCopy)));
}

// The L1 penalty backward step shouldn't do anything if the step size is 0.
TEMPLATE_TEST_CASE("L1PenaltyZeroStepSizeTest", "[FBS]", ENS_ALL_TEST_TYPES)
{
  L1Penalty l(1.0);

  TestType coordinates(100, 1, fill::randu);
  TestType coordinatesCopy(coordinates);

  l.ProximalStep(coordinatesCopy, 0.0);
  REQUIRE(all(all(coordinates == coordinatesCopy)));
}

// The L1 constraint backward step should have zero-valued g(x) when the
// condition is satisfied, and make no changes.
TEMPLATE_TEST_CASE("L1ConstraintZeroTest", "[FBS]", ENS_ALL_TEST_TYPES)
{
  L1Constraint l(1.0);

  TestType coordinates(100, 1, fill::zeros);

  REQUIRE(std::abs(l.Evaluate(coordinates)) <= 1e-15);

  TestType coordinatesCopy(coordinates);
  l.ProximalStep(coordinates, 1.0);

  REQUIRE(all(all(coordinates == coordinatesCopy)));
}

// The L1 constraint should evaluate to Inf when it's not satisfied.
TEMPLATE_TEST_CASE("L1ConstraintTooBigTest", "[FBS]", ENS_ALL_TEST_TYPES)
{
  L1Constraint l(0.01);
  TestType coordinates(100, 1, fill::ones);

  REQUIRE(l.Evaluate(coordinates) ==
      std::numeric_limits<typename TestType::elem_type>::infinity());
}

// Ensure that the L1 constraint projects back onto the unit ball.
TEMPLATE_TEST_CASE("L1Constraint1DProjectionTest", "[FBS]", ENS_ALL_TEST_TYPES,
    ENS_SPARSE_TEST_TYPES)
{
  TestType m(1, 1);
  m(0, 0) = 100.0;

  L1Constraint l(1.0);
  l.ProximalStep(m, 1.0);

  REQUIRE(m(0, 0) == Approx(1.0));

  // Even when the step size is 0, the constraint should still apply.
  m(0, 0) = -50.0;
  l.ProximalStep(m, 0.0);

  REQUIRE(m(0, 0) == Approx(-1.0));
}

template<typename eT>
void RandomFill(Mat<eT>& m)
{
  m.randu();
}

template<typename eT>
void RandomFill(SpMat<eT>& m)
{
  m.sprandu(m.n_rows, m.n_cols, 0.1);
  m *= 10;
}

// Same as the test above, but in 3 dimensions.
TEMPLATE_TEST_CASE("L1Constraint3DProjectionTest", "[FBS]", ENS_ALL_TEST_TYPES,
    ENS_SPARSE_TEST_TYPES)
{
  L1Constraint l(1.0);

  TestType m(3, 1);
  m(0, 0) = 5.0;
  m(1, 0) = 3.0;
  m(2, 0) = -4.0;

  l.ProximalStep(m, 1.0);

  REQUIRE(std::abs(l.Evaluate(m)) <= 1e-15);
  REQUIRE(norm(m, 1) == Approx(1.0));
}

// Same as the test above, but in higher dimensionality.
TEMPLATE_TEST_CASE("L1ConstraintProjectionTest", "[FBS]", ENS_ALL_TEST_TYPES,
    ENS_SPARSE_TEST_TYPES)
{
  L1Constraint l(2.5);

  for (size_t trial = 0; trial < 50; ++trial)
  {
    TestType m(500, 1);
    RandomFill(m);
    m *= 8;

    l.ProximalStep(m, 1.0);

    const double tol =
        (std::is_same<typename TestType::elem_type, float>::value) ? 1e-3 :
        1e-8;
    REQUIRE(std::abs(l.Evaluate(m)) <= 1e-15);
    REQUIRE(norm(m, 1) == Approx(2.5).epsilon(tol));
  }
}

TEMPLATE_TEST_CASE("FBSSphereFunctionTest", "[FBS]", ENS_ALL_TEST_TYPES,
    ENS_SPARSE_TEST_TYPES)
{
  // The sphere function optimizes to the origin anyway, so the L1 penalty does
  // not affect the result.
  FBS<L1Penalty> fbs(L1Penalty(0.1));
  FunctionTest<SphereFunction, TestType>(fbs,
      Tolerances<TestType>::Obj,
      Tolerances<TestType>::Coord);
}

TEMPLATE_TEST_CASE("FBSWoodFunctionTest", "[FBS]", ENS_ALL_TEST_TYPES)
{
  // Set the L1 constraint to be sufficiently large that the final solution is
  // just inside the ball.
  FBS<L1Constraint> fbs(L1Constraint(4.2), 0.0006, 100000);
  FunctionTest<WoodFunction, TestType>(fbs,
      50 * Tolerances<TestType>::Obj,
      50 * Tolerances<TestType>::Coord);
}

TEMPLATE_TEST_CASE("FBSLogisticRegressionFunctionTest", "[FBS]",
    ENS_ALL_TEST_TYPES)
{
  FBS<L1Penalty> fbs(L1Penalty(0.001));
  LogisticRegressionFunctionTest<TestType>(fbs,
      Tolerances<TestType>::LRTrainAcc,
      Tolerances<TestType>::LRTestAcc,
      12);
}

// Check that maxIterations does anything.
TEST_CASE("FBSMaxIterationsTest", "[FBS]")
{
  FBS<L1Penalty> fbs1(L1Penalty(0.001)), fbs2(L1Penalty(0.001));
  fbs1.MaxIterations() = 10;
  fbs2.MaxIterations() = 50000;

  BoothFunction f;
  mat coordinates1 = f.GetInitialPoint<mat>();
  mat coordinates2 = coordinates1;

  fbs1.Optimize(f, coordinates1);
  fbs2.Optimize(f, coordinates2);

  // The second optimization should have proceeded further.
  REQUIRE(f.Evaluate(coordinates1) >= f.Evaluate(coordinates2));
}
