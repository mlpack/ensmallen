/**
 * @file lbfgsb_test.cpp
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#if defined(ENS_USE_COOT)
  #include <armadillo>
  #include <bandicoot>
#endif
#include <ensmallen.hpp>
#include "catch.hpp"
#include "test_function_tools.hpp"

using namespace ens;
using namespace ens::test;

TEMPLATE_TEST_CASE("LBFGSB_Unbounded_RosenbrockFunction", "[LBFGSB]",
    ENS_FULLPREC_TEST_TYPES, ENS_SPARSE_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  L_BFGS_B lbfgsb;
  lbfgsb.MaxIterations() = 10000;

  // No bounds set, should behave like L-BFGS
  RosenbrockFunction f;

  TestType coords = f.GetInitialPoint<TestType>();
  lbfgsb.Optimize(f, coords);

  ElemType finalValue = f.Evaluate(coords);

  REQUIRE(finalValue == Approx(0.0).margin(Tolerances<TestType>::Obj));
  REQUIRE(coords(0) == Approx(1.0).epsilon(Tolerances<TestType>::Coord));
  REQUIRE(coords(1) == Approx(1.0).epsilon(Tolerances<TestType>::Coord));
}

TEMPLATE_TEST_CASE("LBFGSB_Bounded_RosenbrockFunction", "[LBFGSB]",
    ENS_FULLPREC_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  // Set lower bound to [2.0, 2.0]
  arma::mat lowerBound(2, 1);
  lowerBound(0, 0) = 2.0;
  lowerBound(1, 0) = 2.0;

  // Set upper bound to [10.0, 10.0]
  arma::mat upperBound(2, 1);
  upperBound(0, 0) = 10.0;
  upperBound(1, 0) = 10.0;

  L_BFGS_B lbfgsb(10, lowerBound, upperBound);
  lbfgsb.MaxIterations() = 10000;

  RosenbrockFunction f;

  // Initial point inside bounds
  TestType coords(2, 1);
  coords(0, 0) = 2.5;
  coords(1, 0) = 3.0;

  lbfgsb.Optimize(f, coords);

  // The true constrained minimum of Rosenbrock for x >= 2, y >= 2 is at:
  // x = 2, y = 4 (since it minimizes (1-x)^2 + 100(y-x^2)^2).
  // The function value there is (1-2)^2 + 100(4-4)^2 = 1.0.
  ElemType finalValue = f.Evaluate(coords);

  REQUIRE(finalValue == Approx(1.0).margin(1e-4));
  REQUIRE(coords(0) == Approx(2.0).margin(1e-3));
  REQUIRE(coords(1) == Approx(4.0).margin(1e-3));
}

TEMPLATE_TEST_CASE("LBFGSB_ScalarBounded_RosenbrockFunction", "[LBFGSB]",
    ENS_FULLPREC_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  // Set scalar bounds 2.0 <= x <= 10.0
  L_BFGS_B lbfgsb(10, 2.0, 10.0);
  lbfgsb.MaxIterations() = 10000;

  RosenbrockFunction f;

  // Initial point inside bounds
  TestType coords(2, 1);
  coords(0, 0) = 2.5;
  coords(1, 0) = 3.0;

  lbfgsb.Optimize(f, coords);

  ElemType finalValue = f.Evaluate(coords);

  REQUIRE(finalValue == Approx(1.0).margin(1e-4));
  REQUIRE(coords(0) == Approx(2.0).margin(1e-3));
  REQUIRE(coords(1) == Approx(4.0).margin(1e-3));
}
