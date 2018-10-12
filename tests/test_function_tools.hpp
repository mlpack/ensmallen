// Copyright (c) 2018 ensmallen developers.
//
// Licensed under the 3-clause BSD license (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.opensource.org/licenses/BSD-3-Clause

#ifndef ENSMALLEN_TESTS_TEST_FUNCTION_TOOLS_HPP
#define ENSMALLEN_TESTS_TEST_FUNCTION_TOOLS_HPP

#include "catch.hpp"

/**
 * Create the data for the a logistic regression test.
 *
 * @param data Matrix object to store the data into.
 * @param testData Matrix object to store the test data into.
 * @param shuffledData Matrix object to store the shuffled data into.
 * @param responses Matrix object to store the overall responses into.
 * @param testResponses Matrix object to store the test responses into.
 * @param shuffledResponses Matrix object to store the shuffled responses into.
 */
inline void LogisticRegressionTestData(arma::mat& data,
                                       arma::mat& testData,
                                       arma::mat& shuffledData,
                                       arma::Row<size_t>& responses,
                                       arma::Row<size_t>& testResponses,
                                       arma::Row<size_t>& shuffledResponses)
{
  // Generate a two-Gaussian dataset.
  data = arma::mat(3, 1000);
  responses = arma::Row<size_t>(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    // The first Gaussian is centered at (1, 1, 1) and has covariance 3*I.
    data.col(i) = std::sqrt(3) * arma::randn<arma::vec>(3) +
        arma::vec("1.0 1.0 1.0");
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    // The second Gaussian is centered at (9, 9, 9) and has covariance 3*I.
    data.col(i) = std::sqrt(3) * arma::randn<arma::vec>(3) +
        arma::vec("9.0 9.0 9.0");
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
    testData.col(i) = std::sqrt(3) * arma::randn<arma::vec>(3) +
        arma::vec("1.0 1.0 1.0");
    testResponses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    testData.col(i) = std::sqrt(3) * arma::randn<arma::vec>(3) +
        arma::vec("9.0 9.0 9.0");
    testResponses[i] = 1;
  }
}

// Check the values of two matrices.
inline void CheckMatrices(const arma::mat& a,
                          const arma::mat& b,
                          double tolerance = 1e-5)
{
  REQUIRE(a.n_rows == b.n_rows);
  REQUIRE(a.n_cols == b.n_cols);

  for (size_t i = 0; i < a.n_elem; ++i)
  {
    if (std::abs(a[i]) < tolerance / 2)
      REQUIRE(b[i] == Approx(0.0).margin(tolerance / 2.0));
    else
      REQUIRE(a[i] == Approx(b[i]).epsilon(tolerance));
  }
}

#endif
