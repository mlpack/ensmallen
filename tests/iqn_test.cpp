/**
 * @file iqn_test.cpp
 * @author Marcus Edel
 *
 * Test file for IQN (incremental Quasi-Newton).
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
 * Run IQN on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("IQNLogisticRegressionTest", "[IQNTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

  // Now run SGDR with snapshot ensembles on a couple of batch sizes.
  for (size_t batchSize = 1; batchSize < 9; batchSize += 4)
  {
    IQN iqn(0.01, batchSize, 5000, 0.01);
    LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

    arma::mat coordinates = lr.GetInitialPoint();
    iqn.Optimize(lr, coordinates);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses, coordinates);
    REQUIRE(acc == Approx(100.0).epsilon(0.013)); // 1.3% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
    REQUIRE(testAcc == Approx(100.0).epsilon(0.016)); // 1.6% error tolerance.
  }
}

/**
 * Run IQN on logistic regression and make sure the results are acceptable.  Use
 * arma::fmat.
 */
TEST_CASE("IQNLogisticRegressionFMatTest", "[IQNTest]")
{
  arma::fmat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<arma::fmat> lr(shuffledData, shuffledResponses, 0.5);

  // Now run SGDR with snapshot ensembles on a couple of batch sizes.
  for (size_t batchSize = 1; batchSize < 9; batchSize += 4)
  {
    IQN iqn(0.001, batchSize, 5000, 0.01);
    LogisticRegression<arma::fmat> lr(shuffledData, shuffledResponses, 0.5);

    arma::fmat coordinates = lr.GetInitialPoint();
    iqn.Optimize(lr, coordinates);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses, coordinates);
    REQUIRE(acc == Approx(100.0).epsilon(0.013)); // 1.3% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
    REQUIRE(testAcc == Approx(100.0).epsilon(0.016)); // 1.6% error tolerance.
  }
}
