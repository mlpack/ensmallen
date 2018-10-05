// Copyright (c) 2018 ensmallen developers.
// 
// Licensed under the 3-clause BSD license (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.opensource.org/licenses/BSD-3-Clause

#include <ensmallen.hpp>
#include "catch.hpp"

using namespace ens;

// #include <mlpack/core.hpp>
// #include <mlpack/core/optimizers/spalera_sgd/spalera_sgd.hpp>
// #include <mlpack/methods/logistic_regression/logistic_regression.hpp>
// 
// using namespace mlpack;
// using namespace mlpack::optimization;
// using namespace mlpack::distribution;
// using namespace mlpack::regression;

/**
 * Run SPALeRA SGD on logistic regression and make sure the results are
 * acceptable.
 */
TEST_CASE("LogisticRegressionTest","[SPALeRASGDTest]")
{
  // Generate a two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"), arma::eye<arma::mat>(3, 3));

  arma::mat data(3, 500);
  arma::Row<size_t> responses(500);
  for (size_t i = 0; i < 250; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 250; i < 500; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }

  // Shuffle the dataset.
  arma::uvec indices = arma::shuffle(arma::linspace<arma::uvec>(0,
      data.n_cols - 1, data.n_cols));
  arma::mat shuffledData(3, 500);
  arma::Row<size_t> shuffledResponses(500);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    shuffledData.col(i) = data.col(indices[i]);
    shuffledResponses[i] = responses[indices[i]];
  }

  // Create a test set.
  arma::mat testData(3, 500);
  arma::Row<size_t> testResponses(500);
  for (size_t i = 0; i < 250; ++i)
  {
    testData.col(i) = g1.Random();
    testResponses[i] = 0;
  }
  for (size_t i = 250; i < 500; ++i)
  {
    testData.col(i) = g2.Random();
    testResponses[i] = 1;
  }

  // Now run mini-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 30; batchSize < 50; batchSize += 5)
  {
    SPALeRASGD<> mbsgd(0.05 / batchSize, batchSize, 10000, 1e-4);
    LogisticRegression<> lr(shuffledData, shuffledResponses, mbsgd, 0.5);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses);
    REQUIRE(acc == Approx(100.0).epsilon(0.015)); // 1.5% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses);
    REQUIRE(testAcc == Approx(100.0).epsilon(0.024)); // 2.4% error tolerance.
  }
}
