// Copyright (c) 2018 ensmallen developers.
//
// Licensed under the 3-clause BSD license (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.opensource.org/licenses/BSD-3-Clause

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
  AdaDelta optimizer(1.0, 1, 0.99, 1e-8, 5000000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates[0] == Approx(0.0).margin(0.003));
  REQUIRE(coordinates[1] == Approx(0.0).margin(0.003));
  REQUIRE(coordinates[2] == Approx(0.0).margin(0.003));
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
