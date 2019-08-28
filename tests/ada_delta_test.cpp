/**
 * @file ada_delta_test.cpp
 * @author Marcus Edel
 * @author Vasanth Kalingeri
 * @author Abhinav Moudgil
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
 * Tests the Adadelta optimizer using a simple test function.
 */
TEST_CASE("SimpleAdaDeltaTestFunction", "[AdaDeltaTest]")
{
  SGDTestFunction f;
  AdaDelta optimizer(1.0, 1, 0.05, 1e-6, 5000000, 1e-15, true, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.003));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.003));
  REQUIRE(coordinates(2) == Approx(0.0).margin(0.003));
}

/**
 * Run AdaDelta on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("AdaDeltaLogisticRegressionTest", "[AdaDeltaTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

  AdaDelta adaDelta;
  arma::mat coordinates = lr.GetInitialPoint();
  adaDelta.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}

/**
 * Tests the Adadelta optimizer using a simple test function with arma::fmat as
 * the type.
 */
TEST_CASE("SimpleAdaDeltaTestFunctionFMat", "[AdaDeltaTest]")
{
  SGDTestFunction f;
  AdaDelta optimizer(1.0, 1, 0.05, 1e-6, 5000000, 1e-15, true, true);

  arma::fmat coordinates = f.GetInitialPoint<arma::fmat>();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0f).margin(0.01));
  REQUIRE(coordinates(1) == Approx(0.0f).margin(0.01));
  REQUIRE(coordinates(2) == Approx(0.0f).margin(0.01));
}

/**
 * Run AdaDelta on logistic regression and make sure the results are acceptable
 * with arma::fmat as the type.
 */
TEST_CASE("AdaDeltaLogisticRegressionTestFMat", "[AdaDeltaTest]")
{
  arma::fmat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<arma::fmat> lr(shuffledData, shuffledResponses, 0.5);

  AdaDelta adaDelta;
  arma::fmat coordinates = lr.GetInitialPoint();
  adaDelta.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}
