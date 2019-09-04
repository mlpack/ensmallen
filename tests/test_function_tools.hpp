/**
 * @file test_function_tools.hpp
 * @author Marcus Edel
 * @author Ryan Curtin
 * @author Conrad Sanderson
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

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
template<typename MatType>
inline void LogisticRegressionTestData(MatType& data,
                                       MatType& testData,
                                       MatType& shuffledData,
                                       arma::Row<size_t>& responses,
                                       arma::Row<size_t>& testResponses,
                                       arma::Row<size_t>& shuffledResponses)
{
  // Generate a two-Gaussian dataset.
  data = MatType(3, 1000);
  responses = arma::Row<size_t>(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    // The first Gaussian is centered at (1, 1, 1) and has covariance I.
    data.col(i) = arma::randn<arma::Col<typename MatType::elem_type>>(3) +
        arma::Col<typename MatType::elem_type>("1.0 1.0 1.0");
    responses(i) = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    // The second Gaussian is centered at (9, 9, 9) and has covariance I.
    data.col(i) = arma::randn<arma::Col<typename MatType::elem_type>>(3) +
        arma::Col<typename MatType::elem_type>("9.0 9.0 9.0");
    responses(i) = 1;
  }

  // Shuffle the dataset.
  arma::uvec indices = arma::shuffle(arma::linspace<arma::uvec>(0,
      data.n_cols - 1, data.n_cols));
  shuffledData = MatType(3, 1000);
  shuffledResponses = arma::Row<size_t>(1000);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    shuffledData.col(i) = data.col(indices(i));
    shuffledResponses(i) = responses[indices(i)];
  }

  // Create a test set.
  testData = MatType(3, 1000);
  testResponses = arma::Row<size_t>(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    testData.col(i) = arma::randn<arma::Col<typename MatType::elem_type>>(3) +
        arma::Col<typename MatType::elem_type>("1.0 1.0 1.0");
    testResponses(i) = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    testData.col(i) = arma::randn<arma::Col<typename MatType::elem_type>>(3) +
        arma::Col<typename MatType::elem_type>("9.0 9.0 9.0");
    testResponses(i) = 1;
  }
}

// Check the values of two matrices.
template<typename MatType>
inline void CheckMatrices(const MatType& a,
                          const MatType& b,
                          double tolerance = 1e-5)
{
  REQUIRE(a.n_rows == b.n_rows);
  REQUIRE(a.n_cols == b.n_cols);

  for (size_t i = 0; i < a.n_elem; ++i)
  {
    if (std::abs(a(i)) < tolerance / 2)
      REQUIRE(b(i) == Approx(0.0).margin(tolerance / 2.0));
    else
      REQUIRE(a(i) == Approx(b(i)).epsilon(tolerance));
  }
}

#endif
