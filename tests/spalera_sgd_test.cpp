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
 * Run SPALeRA SGD on logistic regression and make sure the results are
 * acceptable.
 */
TEST_CASE("LogisticRegressionTest","[SPALeRASGDTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  // Now run mini-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 30; batchSize < 50; batchSize += 5)
  {
    bool success = false;

    // It's possible that convergence can randomly fail.  So allow up to three
    // trials.
    for (size_t trial = 0; trial < 3; ++trial)
    {
      SPALeRASGD<> optimizer(0.05 / batchSize, batchSize, 10000, 1e-4);
      LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

      arma::mat coordinates = lr.GetInitialPoint();
      optimizer.Optimize(lr, coordinates);

      // Ensure that the error is close to zero.
      const double acc = lr.ComputeAccuracy(data, responses, coordinates);
      const double testAcc = lr.ComputeAccuracy(testData, testResponses,
          coordinates);

      if (acc >= 98.5 && testAcc >= 97.6)
      {
        success = true;
        break;
      }
    }

    REQUIRE(success == true); // At least one trial must succeed.
  }
}
