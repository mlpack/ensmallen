/**
 * @file sgdr_test.cpp
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

/*
 * Test that the step size resets after a specified number of epochs.
 */
TEST_CASE("SGDRCyclicalResetTest","[SGDRTest]")
{
  const double stepSize = 0.5;
  arma::mat iterate;

  // Now run cyclical decay policy with a couple of multiplicators and initial
  // restarts.
  for (size_t restart = 5; restart < 100; restart += 10)
  {
    for (size_t mult = 2; mult < 5; ++mult)
    {
      double epochStepSize = stepSize;

      CyclicalDecay cyclicalDecay(restart, double(mult), stepSize);
      cyclicalDecay.EpochBatches() = (double) 1000 / 10;

      // Create all restart epochs.
      arma::Col<size_t> nextRestart(1000 / 10 /  mult);
      nextRestart(0) = restart;
      for (size_t j = 1; j < nextRestart.n_elem; ++j)
        nextRestart(j) = nextRestart(j - 1) + restart * std::pow(mult, j);

      for (size_t i = 0; i < 1000; ++i)
      {
        cyclicalDecay.Update(iterate, epochStepSize, iterate);
        if (arma::accu(arma::find(nextRestart == i)) > 0)
        {
          REQUIRE(epochStepSize == stepSize);
        }
      }
    }
  }
}

/**
 * Run SGDR on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("SGDRLogisticRegressionTest","[SGDRTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  // Now run SGDR with a couple of batch sizes.
  for (size_t batchSize = 5; batchSize < 50; batchSize += 5)
  {
    SGDR<> sgdr(0.01, 50, 2.0, batchSize, 10000, 1e-3);
    LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

    arma::mat coordinates = lr.GetInitialPoint();
    sgdr.Optimize(lr, coordinates);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses, coordinates);
    REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses,
        coordinates);
    REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
  }
}

/**
 * Run SGDWR on logistic regression and make sure the results are better.
 */
TEST_CASE("SGDWRLogisticRegressionTest","[SGDRTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  // Now run SGDWR with a couple of batch sizes.
  for (size_t batchSize = 5; batchSize < 50; batchSize += 5)
  {
    SGDWR sgdr(0.02, 50, 2.0, batchSize, 10000, 1e-3);
    LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

    arma::mat coordinates = lr.GetInitialPoint();
    sgdr.Optimize(lr, coordinates);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses, coordinates);
    REQUIRE(acc == Approx(100.0).epsilon(0.001)); // 0.1% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses,
        coordinates);
    REQUIRE(testAcc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.
  }
}
