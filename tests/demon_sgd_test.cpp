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
 * Tests the DemonSGD optimizer using a simple test function.
 */
TEST_CASE("DemonSGDSimpleTestFunction", "[DemonSGDTest]")
{
  SGDTestFunction f;
  DemonSGD optimizer(1e-2, 1, 0.9, 400000);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(2) == Approx(0.0).margin(0.1));
}

/**
 * Run DemonSGD on logistic regression and make sure the results are
 * acceptable.
 */
TEST_CASE("DemonSGDLogisticRegressionTest", "[DemonSGDTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

  DemonSGD optimizer(0.8, 32, 0.9, 100000, 1e-9, true, true, true);
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

