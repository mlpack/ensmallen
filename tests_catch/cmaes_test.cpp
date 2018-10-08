// Copyright (c) 2018 ensmallen developers.
// 
// Licensed under the 3-clause BSD license (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.opensource.org/licenses/BSD-3-Clause

#include <ensmallen.hpp>
#include "catch.hpp"

using namespace arma;
using namespace ens;


// #include <mlpack/core.hpp>
// #include <mlpack/core/optimizers/cmaes/cmaes.hpp>
// #include <mlpack/core/optimizers/problems/sgd_test_function.hpp>
// #include <mlpack/methods/logistic_regression/logistic_regression.hpp>
// 
// using namespace mlpack::optimization;
// using namespace mlpack::optimization::test;
// using namespace mlpack::distribution;
// using namespace mlpack::regression;
// using namespace mlpack;

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
 * Create the data for the logistic regression test case.
 */
void CreateLogisticRegressionTestData(arma::mat& data,
                                      arma::mat& testData,
                                      arma::mat& shuffledData,
                                      arma::Row<size_t>& responses,
                                      arma::Row<size_t>& testResponses,
                                      arma::Row<size_t>& shuffledResponses)
{
  // Generate a two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"), arma::eye<arma::mat>(3, 3));

  data = arma::mat(3, 1000);
  responses = arma::Row<size_t>(1000);
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
  shuffledData = arma::mat(3, 1000);
  shuffledResponses = arma::Row<size_t>(1000);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    shuffledData.col(i) = data.col(indices[i]);
    shuffledResponses[i] = responses[indices[i]];
  }

  // Create a test set.
  testData = arma::mat(3, 1000);
  testResponses = arma::Row<size_t>(1000);
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

    CreateLogisticRegressionTestData(data, testData, shuffledData,
        responses, testResponses, shuffledResponses);

    CMAES<> cmaes(0, -1, 1, 32, 200, 1e-3);
    LogisticRegression<> lr(shuffledData, shuffledResponses, cmaes, 0.5);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses);
    const double testAcc = lr.ComputeAccuracy(testData, testResponses);
    if (acc >= 99.7 && testAcc >= 99.4)
    {
      success = true;
      break;
    }
  }

  REQUIRE(success, true);
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

    CreateLogisticRegressionTestData(data, testData, shuffledData,
        responses, testResponses, shuffledResponses);

    ApproxCMAES<> cmaes(0, -1, 1, 32, 200, 1e-3);
    LogisticRegression<> lr(shuffledData, shuffledResponses, cmaes, 0.5);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses);
    const double testAcc = lr.ComputeAccuracy(testData, testResponses);
    if (acc >= 99.7 && testAcc >= 99.4)
    {
      success = true;
      break;
    }
  }

  REQUIRE(success == true);
}
