/**
 * @file demon_adam_test.cpp
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
 * Tests the DemonAdam optimizer using a simple test function.
 */
TEST_CASE("DemonAdamSimpleTestFunction", "[DemonAdamTest]")
{
  SGDTestFunction f;
  DemonAdam optimizer(1e-3, 1, 0.9, 0.9, 0.999, 1e-8, 900000);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(2) == Approx(0.0).margin(0.1));
}

/**
 * Run DemonAdam on logistic regression and make sure the results are
 * acceptable.
 */
TEST_CASE("DemonAdamLogisticRegressionTest", "[DemonAdamTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

  DemonAdam optimizer(0.5, 10, 0.9, 0.9, 0.999, 1e-8,
      10000, 1e-9, true, true, true);
  arma::mat coordinates = lr.GetInitialPoint();
  optimizer.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}

/**
 * Test the Adam optimizer on the Sphere function.
 */
TEST_CASE("DemonAdamSphereFunctionTest", "[DemonAdamTest]")
{
  SphereFunction f(2);
  DemonAdam optimizer(0.5, 2, 0.9);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
}

/**
 * Test the DemonAdam optimizer on the Matyas function.
 */
TEST_CASE("DemonAdamMatyasFunctionTest", "[DemonAdamTest]")
{
  MatyasFunction f;
  DemonAdam optimizer(0.5, 1, 0.9);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  // 3% error tolerance.
  REQUIRE((std::trunc(100.0 * coordinates(0)) / 100.0) ==
      Approx(0.0).epsilon(0.003));
  REQUIRE((std::trunc(100.0 * coordinates(1)) / 100.0) ==
      Approx(0.0).epsilon(0.003));
}

/**
 * Tests the DemonAdam optimizer using a simple test function.
 */
TEST_CASE("DemonAdamSimpleTestFunctionFloat", "[DemonAdamTest]")
{
  SGDTestFunction f;
  DemonAdam optimizer(1e-3, 1, 0.9, 0.7, 0.999, 1e-8, 900000);

  arma::fmat coordinates = f.GetInitialPoint<arma::fmat>();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(2) == Approx(0.0).margin(0.1));
}

/**
 * Test the Adam optimizer on the Sphere function.
 */
TEST_CASE("DemonAdamSphereFunctionTestFloat", "[DemonAdamTest]")
{
  SphereFunction f(2);
  DemonAdam optimizer(0.5, 2, 0.9);

  arma::fmat coordinates = f.GetInitialPoint<arma::fmat>();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
}

/**
 * Test the DemonAdam optimizer on the Matyas function.
 */
TEST_CASE("DemonAdamMatyasFunctionTestFloat", "[DemonAdamTest]")
{
  MatyasFunction f;
  DemonAdam optimizer(0.5, 1, 0.9);

  arma::fmat coordinates = f.GetInitialPoint<arma::fmat>();
  optimizer.Optimize(f, coordinates);

  // 3% error tolerance.
  REQUIRE((std::trunc(100.0 * coordinates(0)) / 100.0) ==
      Approx(0.0).epsilon(0.003));
  REQUIRE((std::trunc(100.0 * coordinates(1)) / 100.0) ==
      Approx(0.0).epsilon(0.003));
}

/**
 * Run DemonAdam (AdaMax update) on logistic regression and make sure the
 * results are acceptable.
 */
TEST_CASE("DemonAdaMaxLogisticRegressionTest", "[DemonAdamTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);
  
  DemonAdamType<AdaMaxUpdate> optimizer(0.5, 10, 0.9, 0.9, 0.999, 1e-8,
      10000, 1e-9, true, true, true);
  arma::mat coordinates = lr.GetInitialPoint();
  optimizer.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}
