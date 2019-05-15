/**
 * @file de_test.cpp
 * @author Rahul Ganesh Prabhu
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include "../include/ensmallen.hpp"
#include "catch.hpp"
#include "test_function_tools.hpp"

using namespace ens;
using namespace ens::test;

/**
 * Train and test a logistic regression function using DE optimizer.
 */
TEST_CASE("DELogisticRegressionTest", "[DETest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

  DE opt(200, 1000, 0.6, 0.8, 1e-5);
  arma::mat coordinates = lr.GetInitialPoint();
  opt.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}

/**
 * Train and test a logistic regression function using DE optimizer.  Use
 * arma::fmat.
 */
TEST_CASE("DELogisticRegressionFMatTest", "[DETest]")
{
  arma::fmat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData, responses,
      testResponses, shuffledResponses);
  LogisticRegression<arma::fmat> lr(shuffledData, shuffledResponses, 0.5);

  DE opt(200, 1000, 0.6, 0.8, 1e-5);
  arma::fmat coordinates = lr.GetInitialPoint();
  opt.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const float acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.03)); // 3% error tolerance.

  const float testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.06)); // 6% error tolerance.
}
