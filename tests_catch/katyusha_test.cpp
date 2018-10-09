// Copyright (c) 2018 ensmallen developers.
// 
// Licensed under the 3-clause BSD license (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.opensource.org/licenses/BSD-3-Clause

#include <ensmallen.hpp>
#include "catch.hpp"

using namespace ens;
using namespace ens::test;

// #include <mlpack/core.hpp>
// #include <mlpack/core/optimizers/katyusha/katyusha.hpp>
// 
// using namespace mlpack;
// using namespace mlpack::optimization;

/**
 * Run Katyusha on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("KatyushaLogisticRegressionTest", "[KatyushaTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  // Now run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 30; batchSize < 45; batchSize += 5)
  {
    Katyusha optimizer(1.0, 10.0, batchSize, 100, 0, 1e-10, true);
    LogisticRegression<> lr(shuffledData, shuffledResponses, optimizer, 0.5);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses);
    REQUIRE(acc == Approx(100.0).epsilon(0.015)); // 1.5% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses);
    REQUIRE(testAcc == Approx(100.0).epsilon(0.015)); // 1.5% error tolerance.
  }
}

/**
 * Run Proximal Katyusha on logistic regression and make sure the results are
 * acceptable.
 */
TEST_CASE("KatyushaProximalLogisticRegressionTest", "[KatyushaTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  // Now run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 30; batchSize < 45; batchSize += 5)
  {
    KatyushaProximal optimizer(1.0, 10.0, batchSize, 100, 0, 1e-10, true);
    LogisticRegression<> lr(shuffledData, shuffledResponses, optimizer, 0.5);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses);
    REQUIRE(acc == Approx(100.0).epsilon(0.015)); // 1.5% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses);
    REQUIRE(testAcc == Approx(100.0).epsilon(0.015)); // 1.5% error tolerance.
  }
}
