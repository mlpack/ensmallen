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
 * Tests the CMA-ES optimizer using a simple test function.
 */
TEST_CASE("SimpleTestFunction", "[CMAESTest]")
{
  SGDTestFunction f;
  CMAES<> optimizer(0, -1, 1, 32, 200, -1);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates[0] == Approx(0.0).margin(0.003));
  REQUIRE(coordinates[1] == Approx(0.0).margin(0.003));
  REQUIRE(coordinates[2] == Approx(0.0).margin(0.003));
}

/**
 * Run CMA-ES with the full selection policy on logistic regression and
 * make sure the results are acceptable.
 */
TEST_CASE("CMAESLogisticRegressionTest", "[CMAESTest]")
{
  const size_t trials = 3;
  bool success = false;
  for (size_t trial = 0; trial < trials; ++trial)
  {
    arma::mat data, testData, shuffledData;
    arma::Row<size_t> responses, testResponses, shuffledResponses;

    LogisticRegressionTestData(data, testData, shuffledData,
        responses, testResponses, shuffledResponses);
    LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

    CMAES<> cmaes(0, -1, 1, 32, 200, 1e-3);
    arma::mat coordinates = lr.GetInitialPoint();
    cmaes.Optimize(lr, coordinates);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses, coordinates);
    const double testAcc = lr.ComputeAccuracy(testData, testResponses,
        coordinates);
    if (acc >= 99.7 && testAcc >= 99.4)
    {
      success = true;
      break;
    }
  }

  REQUIRE(success == true);
}

/**
 * Run CMA-ES with the random selection policy on logistic regression and
 * make sure the results are acceptable.
 */
TEST_CASE("ApproxCMAESLogisticRegressionTest", "[CMAESTest]")
{
  const size_t trials = 3;
  bool success = false;
  for (size_t trial = 0; trial < trials; ++trial)
  {
    arma::mat data, testData, shuffledData;
    arma::Row<size_t> responses, testResponses, shuffledResponses;

    LogisticRegressionTestData(data, testData, shuffledData,
        responses, testResponses, shuffledResponses);
    LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

    ApproxCMAES<> cmaes(0, -1, 1, 32, 200, 1e-3);
    arma::mat coordinates = lr.GetInitialPoint();
    cmaes.Optimize(lr, coordinates);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses, coordinates);
    const double testAcc = lr.ComputeAccuracy(testData, testResponses,
        coordinates);
    if (acc >= 99.7 && testAcc >= 99.4)
    {
      success = true;
      break;
    }
  }

  REQUIRE(success == true);
}
