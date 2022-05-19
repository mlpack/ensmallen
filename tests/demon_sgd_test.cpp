/**
 * @file demon_sgd_test.cpp
 * @author Marcus Edel
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <ensmallen.hpp>
#include "catch.hpp"
#include "test_function_tools.hpp"

using namespace ens;
using namespace ens::test;

/**
 * Run DemonSGD on logistic regression and make sure the results are
 * acceptable.
 */
TEST_CASE("DemonSGDLogisticRegressionTest", "[DemonSGDTest]")
{
  DemonSGD optimizer(0.1, 32, 0.9, 1000000, 1e-9, true, true, true);
  LogisticRegressionFunctionTest(optimizer, 0.003, 0.006, 6);
}

/**
 * Tests the DemonSGD optimizer using a simple test function.
 */
TEST_CASE("DemonSGDSimpleTestFunctionFloat", "[DemonSGDTest]")
{
  SGDTestFunction f;
  DemonSGD optimizer(1e-2, 1, 0.9, 400000);

  arma::fmat coordinates = f.GetInitialPoint<arma::fmat>();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(2) == Approx(0.0).margin(0.1));
}

