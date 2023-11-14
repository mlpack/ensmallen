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

TEMPLATE_TEST_CASE("FBSSimpleTest", "[FBSTest]", float, double)
{
  typedef TestType eT;

  // Make sure that we can get a decent result with no g(x) constraint.
  FBS<L1Penalty> fbs(L1Penalty(0.0), 0.001, 50000);
  GeneralizedRosenbrockFunction f(20);
  FunctionTest<GeneralizedRosenbrockFunction, Mat<eT>>(fbs, f);
}

// The L1 penalty backward step should have zero-valued g(x) and make no changes
// when the penalty is 0.
TEMPLATE_TEST_CASE("L1PenaltyZeroTest", "[FBSTest]", float, double)
{
  typedef TestType eT;

  L1Penalty l(0.0);

  Mat<eT> coordinates(100, 1, fill::randu);

  REQUIRE(std::abs(l.Evaluate(coordinates)) <= 1e-15);

  Mat<eT> coordinatesCopy(coordinates);
  l.ProximalStep(coordinates, 1.0);

  REQUIRE(all(all(coordinates == coordinatesCopy)));
}

// The L1 penalty backward step shouldn't do anything if the step size is 0.
TEMPLATE_TEST_CASE("L1PenaltyZeroStepSizeTest", "[FBSTest]", float, double)
{
  typedef TestType eT;

  L1Penalty l(1.0);

  Mat<eT> coordinates(100, 1, fill::randu);
  Mat<eT> coordinatesCopy(coordinates);

  l.ProximalStep(coordinatesCopy, 0.0);
  REQUIRE(all(all(coordinates == coordinatesCopy)));
}

// The L1 constraint backward step should have zero-valued g(x) when the
// condition is satisfied, and make no changes.
TEMPLATE_TEST_CASE("L1ConstraintZeroTest", "[FBSTest]", float, double)
{
  typedef TestType eT;

  L1Constraint l(1.0);

  Mat<eT> coordinates(100, 1, fill::zeros);

  REQUIRE(std::abs(l.Evaluate(coordinates)) <= 1e-15);

  Mat<eT> coordinatesCopy(coordinates);
  l.ProximalStep(coordinates, 1.0);

  REQUIRE(all(all(coordinates == coordinatesCopy)));
}

// The L1 constraint should evaluate to Inf when it's not satisfied.
TEMPLATE_TEST_CASE("L1ConstraintTooBigTest", "[FBSTest]", float, double)
{
  typedef TestType eT;

  L1Constraint l(0.01);
  Mat<eT> coordinates(100, 1, fill::ones);

  REQUIRE(l.Evaluate(coordinates) == std::numeric_limits<eT>::infinity());
}

// Ensure that the L1 constraint projects back onto the unit ball.
TEMPLATE_TEST_CASE("L1Constraint1DProjectionTest", "[FBSTest]", fmat, sp_fmat,
    mat, sp_mat)
{
  typedef TestType MatType;

  MatType m(1, 1);
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
TEMPLATE_TEST_CASE("L1Constraint3DProjectionTest", "[FBSTest]", fmat, sp_fmat,
    mat, sp_mat)
{
  typedef TestType MatType;

  L1Constraint l(1.0);

  MatType m(3, 1);
  m(0, 0) = 5.0;
  m(1, 0) = 3.0;
  m(2, 0) = -4.0;

  l.ProximalStep(m, 1.0);

  REQUIRE(std::abs(l.Evaluate(m)) <= 1e-15);
  REQUIRE(norm(m, 1) == Approx(1.0));
}

// Same as the test above, but in higher dimensionality.
TEMPLATE_TEST_CASE("L1ConstraintProjectionTest", "[FBSTest]", fmat, sp_fmat,
    mat, sp_mat)
{
  typedef TestType MatType;

  L1Constraint l(2.5);

  for (size_t trial = 0; trial < 50; ++trial)
  {
    MatType m(500, 1);
    RandomFill(m);
    m *= 8;

    l.ProximalStep(m, 1.0);

    REQUIRE(std::abs(l.Evaluate(m)) <= 1e-15);
    REQUIRE(norm(m, 1) == Approx(2.5));
  }
}

TEMPLATE_TEST_CASE("FBSSphereFunctionTest", "[FBSTest]", fmat, mat, sp_mat)
{
  typedef TestType MatType;

  // The sphere function optimizes to the origin anyway, so the L1 penalty does
  // not affect the result.
  FBS<L1Penalty> fbs(L1Penalty(0.1));
  FunctionTest<SphereFunction, MatType>(fbs);
}

TEMPLATE_TEST_CASE("FBSWoodFunctionTest", "[FBSTest]", fmat, mat)
{
  typedef TestType MatType;

  // Set the L1 constraint to be sufficiently large that the final solution is
  // just inside the ball.
  FBS<L1Constraint> fbs(L1Constraint(4.2), 0.0006, 100000);
  FunctionTest<WoodFunction, MatType>(fbs);
}

TEMPLATE_TEST_CASE("FBSLogisticRegressionFunctionTest", "[FBSTest]", fmat, mat)
{
  typedef TestType MatType;

  FBS<L1Penalty> fbs(L1Penalty(0.001));
  LogisticRegressionFunctionTest<MatType>(fbs, 0.05, 0.05, 5);
}

// Check that maxIterations does anything.
TEST_CASE("FBSMaxIterationsTest", "[FBSTest]")
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
