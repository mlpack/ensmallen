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
 * Run big-batch SGD using BBS_BB on logistic regression and make sure the
 * results are acceptable.
 */
TEST_CASE("BBSBBLogisticRegressionTest", "[BigBatchSGDTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  // Now run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 30; batchSize < 40; batchSize += 5)
  {
    BBS_BB bbsgd(batchSize, 0.01, 0.1, 8000, 1e-4);

    LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);
    arma::mat coordinates = lr.GetInitialPoint();
    bbsgd.Optimize(lr, coordinates);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses, coordinates);
    REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses,
        coordinates);
    REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
  }
}

/**
 * Run big-batch SGD using BBS_Armijo on logistic regression and make sure the
 * results are acceptable.
 */
TEST_CASE("BBSArmijoLogisticRegressionTest", "[BigBatchSGDTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  // Now run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 30; batchSize < 40; batchSize += 1)
  {
    BBS_Armijo bbsgd(batchSize, 0.01, 0.1, 8000, 1e-4);

    LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);
    arma::mat coordinates = lr.GetInitialPoint();
    bbsgd.Optimize(lr, coordinates);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses, coordinates);
    REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses,
        coordinates);
    REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
  }
}
