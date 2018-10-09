// Copyright (c) 2018 ensmallen developers.
// 
// Licensed under the 3-clause BSD license (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.opensource.org/licenses/BSD-3-Clause

#include <ensmallen.hpp>
#include "catch.hpp"

// #include <mlpack/core.hpp>
// #include <mlpack/core/optimizers/smorms3/smorms3.hpp>
// #include <mlpack/core/optimizers/problems/sgd_test_function.hpp>
// #include <mlpack/methods/logistic_regression/logistic_regression.hpp>

using namespace arma;
using namespace ens;
using namespace ens::test;

// using namespace mlpack::optimization;
// using namespace mlpack::optimization::test;
// using namespace mlpack::distribution;
// using namespace mlpack::regression;
// using namespace mlpack;

/**
 * Tests the SMORMS3 optimizer using a simple test function.
 */
TEST_CASE("SimpleSMORMS3TestFunction","[SMORMS3Test]")
{
  SGDTestFunction f;
  SMORMS3 s(0.001, 1, 1e-16, 5000000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  s.Optimize(f, coordinates);

  REQUIRE(coordinates[0] == Approx(0.0).margin(0.1));
  REQUIRE(coordinates[1] == Approx(0.0).margin(0.1));
  REQUIRE(coordinates[2] == Approx(0.0).margin(0.1));
}

/**
 * Run SMORMS3 on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("SMORMS3LogisticRegressionTest","[SMORMS3Test]")
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

  SMORMS3 smorms3;
  LogisticRegression<> lr(shuffledData, shuffledResponses, smorms3, 0.5);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}
