/**
 * @file spalera_sgd_test.cpp
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

/**
 * Run SPALeRA SGD on logistic regression and make sure the results are
 * acceptable.  Use arma::fmat.
 */
TEST_CASE("LogisticRegressionFMatTest","[SPALeRASGDTest]")
{
  arma::fmat data, testData, shuffledData;
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
      LogisticRegression<arma::fmat> lr(shuffledData, shuffledResponses, 0.5);

      arma::fmat coordinates = lr.GetInitialPoint();
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
