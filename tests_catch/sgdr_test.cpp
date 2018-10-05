// Copyright (c) 2018 ensmallen developers.
// 
// Licensed under the 3-clause BSD license (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.opensource.org/licenses/BSD-3-Clause

#include <ensmallen.hpp>
#include "catch.hpp"

using namespace std;
using namespace arma;
using namespace ens;

// #include <mlpack/core.hpp>
// #include <mlpack/core/optimizers/sgdr/cyclical_decay.hpp>
// #include <mlpack/core/optimizers/sgdr/sgdr.hpp>
// #include <mlpack/methods/logistic_regression/logistic_regression.hpp>
// 
// using namespace mlpack;
// using namespace mlpack::optimization;
// using namespace mlpack::distribution;
// using namespace mlpack::regression;

/*
 * Test that the step size resets after a specified number of epochs.
 */
TEST_CASE("CyclicalResetTest","[SGDRTest]")
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
        nextRestart(j) = nextRestart(j - 1) * mult;

      for (size_t i = 0; i < 1000; ++i)
      {
        cyclicalDecay.Update(iterate, epochStepSize, iterate);
        if (i <= restart || arma::accu(arma::find(nextRestart == i)) > 0)
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
TEST_CASE("LogisticRegressionTest","[SGDRTest]")
{
  // Generate a two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"), arma::eye<arma::mat>(3, 3));

  arma::mat data(3, 1000);
  arma::Row<size_t> responses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }

  // Shuffle the dataset.
  arma::uvec indices = arma::shuffle(arma::linspace<arma::uvec>(0,
      data.n_cols - 1, data.n_cols));
  arma::mat shuffledData(3, 1000);
  arma::Row<size_t> shuffledResponses(1000);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    shuffledData.col(i) = data.col(indices[i]);
    shuffledResponses[i] = responses[indices[i]];
  }

  // Create a test set.
  arma::mat testData(3, 1000);
  arma::Row<size_t> testResponses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    testData.col(i) = g1.Random();
    testResponses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    testData.col(i) = g2.Random();
    testResponses[i] = 1;
  }

  // Now run SGDR with a couple of batch sizes.
  for (size_t batchSize = 5; batchSize < 50; batchSize += 5)
  {
    SGDR<> sgdr(50, 2.0, batchSize, 0.01, 10000, 1e-3);
    LogisticRegression<> lr(shuffledData, shuffledResponses, sgdr, 0.5);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses);
    REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses);
    REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
  }
}
