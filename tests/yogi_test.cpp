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
  McCormickFunction f;
  Yogi optimizer(0.5, 1, 0.7, 0.999, 1e-8, 500000, 1e-5, false);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  // 3% error tolerance.
  REQUIRE(coordinates(0) == Approx(-0.547).epsilon(0.03));
  REQUIRE(coordinates(1) == Approx(-1.547).epsilon(0.03));
}

/**
 * Tests the Yogi optimizer using a simple test function.
 */
TEST_CASE("SimpleYogiTestFunction", "[YogiTest]")
{
  SGDTestFunction f;
  Yogi optimizer(1e-3, 1, 0.9, 0.999, 1e-8, 500000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.3));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.3));
  REQUIRE(coordinates(2) == Approx(0.0).margin(0.3));
}

/**
 * Run Yogi on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("YogiLogisticRegressionTest", "[YogiTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

  Yogi yogi;
  arma::mat coordinates = lr.GetInitialPoint();
  yogi.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}

/**
 * Run Yogi on logistic regression and make sure the results are acceptable,
 * using arma::fmat.
 */
TEST_CASE("YogiLogisticRegressionFMatTest", "[YogiTest]")
{
  arma::fmat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<arma::fmat> lr(shuffledData, shuffledResponses, 0.5);

  Yogi optimizer;
  arma::fmat coordinates = lr.GetInitialPoint();
  optimizer.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const float acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.03)); // 3% error tolerance.

  const float testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.06)); // 6% error tolerance.
}
