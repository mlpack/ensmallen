/**
 * @file yogi_test.cpp
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
 * Test the Yogi optimizer on the Sphere function.
 */
TEST_CASE("YogiSphereFunctionTest", "[YogiTest]")
{
  SphereFunction f(2);
  Yogi optimizer(0.5, 2, 0.7, 0.999, 1e-8, 500000, 1e-3, false);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
}

/**
 * Test the Yogi optimizer on the Sphere function with arma::fmat.
 */
TEST_CASE("YogiSphereFunctionTestFMat", "[YogiTest]")
{
  SphereFunction f(2);
  Yogi optimizer(0.5, 2, 0.7, 0.999, 1e-8, 500000, 1e-3, false);

  arma::fmat coordinates = f.GetInitialPoint<arma::fmat>();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
}

/**
 * Test the Yogi optimizer on the McCormick function.
 */
TEST_CASE("YogiMcCormickFunctionTest", "[YogiTest]")
{
  Yogi optimizer(0.5, 1, 0.7, 0.999, 1e-8, 500000, 1e-5, false);
  FunctionTest<McCormickFunction>(optimizer, 0.5, 0.1);
}

/**
 * Run Yogi on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("YogiLogisticRegressionTest", "[YogiTest]")
{
  Yogi optimizer;
  LogisticRegressionFunctionTest(optimizer, 0.003, 0.006);
}

/**
 * Run Yogi on logistic regression and make sure the results are acceptable,
 * using arma::fmat.
 */
TEST_CASE("YogiLogisticRegressionFMatTest", "[YogiTest]")
{
  Yogi optimizer;
  LogisticRegressionFunctionTest<arma::fmat>(optimizer, 0.003, 0.006);
}
