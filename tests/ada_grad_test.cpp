/**
 * @file ada_grad_test.cpp
 * @author Abhinav Moudgil
 * @author Marcus Edel
 * @author Conrad Sanderson
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
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

  AdaGrad adagrad(0.99, 32, 1e-8, 5000000, 1e-9, true);
  arma::mat coordinates = lr.GetInitialPoint();
  adagrad.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}

/**
 * Tests the Adagrad optimizer using a simple test function with arma::fmat.
 */
TEST_CASE("SimpleAdaGradTestFunctionFMat", "[AdaGradTest]")
{
  SGDTestFunction f;
  AdaGrad optimizer(0.99, 1, 1e-8, 5000000, 1e-9, true);

  arma::fmat coordinates = f.GetInitialPoint<arma::fmat>();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0f).margin(0.01));
  REQUIRE(coordinates(1) == Approx(0.0f).margin(0.01));
  REQUIRE(coordinates(2) == Approx(0.0f).margin(0.01));
}

/**
 * Run AdaGrad on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("AdaGradLogisticRegressionTestFMat", "[AdaGradTest]")
{
  arma::fmat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<arma::fmat> lr(shuffledData, shuffledResponses, 0.5);

  AdaGrad adagrad(0.99, 32, 1e-8, 5000000, 1e-9, true);
  arma::fmat coordinates = lr.GetInitialPoint();
  adagrad.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}
