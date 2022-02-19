/**
 * @file ada_sqrt_test.cpp
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
 * Tests the Adagrad optimizer using a simple test function.
 */
TEST_CASE("SimpleAdaGradTestFunction", "[AdaGradTest]")
{
  SGDTestFunction f;
  AdaGrad optimizer(0.99, 1, 1e-8, 5000000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.003));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.003));
  REQUIRE(coordinates(2) == Approx(0.0).margin(0.003));
}

/**
 * Run AdaGrad on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("AdaGradLogisticRegressionTest", "[AdaGradTest]")
{
  AdaGrad adagrad(0.99, 32, 1e-8, 5000000, 1e-9, true);
  LogisticRegressionFunctionTest(adagrad, 0.003, 0.006);
}

/**
 * Tests the Adagrad optimizer using a simple test function with arma::fmat.
 */
TEST_CASE("SimpleAdaGradTestFunctionFMat", "[AdaGradTest]")
{
  size_t trials = 3;
  SGDTestFunction f;
  arma::fmat coordinates;

  for (size_t i = 0; i < trials; ++i)
  {
    coordinates = f.GetInitialPoint<arma::fmat>();

    AdaGrad optimizer(0.99, 1, 1e-8, 5000000, 1e-9, true);
    optimizer.Optimize(f, coordinates);

    if (arma::max(arma::vectorise(arma::abs(coordinates))) < 0.01f)
      break;
  }

  REQUIRE(coordinates(0) == Approx(0.0f).margin(0.01));
  REQUIRE(coordinates(1) == Approx(0.0f).margin(0.01));
  REQUIRE(coordinates(2) == Approx(0.0f).margin(0.01));
}

/**
 * Run AdaGrad on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("AdaGradLogisticRegressionTestFMat", "[AdaGradTest]")
{
  AdaGrad adagrad(0.99, 32, 1e-8, 5000000, 1e-9, true);
  LogisticRegressionFunctionTest<arma::fmat>(adagrad, 0.003, 0.006);
}
