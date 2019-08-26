/**
 * @file bigbatch_sgd_test.cpp
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
 * Run big-batch SGD using BBS_BB on logistic regression and make sure the
 * results are acceptable.
 */
TEST_CASE("BBSBBLogisticRegressionTest", "[BigBatchSGDTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  // Now run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 40; batchSize < 50; batchSize += 5)
  {
    BBS_Armijo bbsgd(batchSize, 0.005, 0.1, 10000, 1e-6, true, true);

    LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);
    arma::mat coordinates = lr.GetInitialPoint();
    bbsgd.Optimize(lr, coordinates);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses, coordinates);
    REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses,
        coordinates);
    REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
  }
}

/**
 * Run big-batch SGD using BBS_Armijo on logistic regression and make sure the
 * results are acceptable.
 */
TEST_CASE("BBSArmijoLogisticRegressionTest", "[BigBatchSGDTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  // Now run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 40; batchSize < 50; batchSize += 1)
  {
    BBS_Armijo bbsgd(batchSize, 0.005, 0.1, 10000, 1e-6, true, true);

    LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);
    arma::mat coordinates = lr.GetInitialPoint();
    bbsgd.Optimize(lr, coordinates);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses, coordinates);
    REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses,
        coordinates);
    REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
  }
}

/**
 * Run big-batch SGD using BBS_BB on logistic regression and make sure the
 * results are acceptable.  Use arma::fmat as the objective type.
 */
TEST_CASE("BBSBBLogisticRegressionFMatTest", "[BigBatchSGDTest]")
{
  arma::fmat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  // Now run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 350; batchSize < 360; batchSize += 5)
  {
    BBS_BB bbsgd(batchSize, 0.001, 0.1, 10000, 1e-8, true, true);

    LogisticRegression<arma::fmat> lr(shuffledData, shuffledResponses, 0.5);
    arma::fmat coordinates = lr.GetInitialPoint();
    bbsgd.Optimize(lr, coordinates);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses, coordinates);
    REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses,
        coordinates);
    REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
  }
}

/**
 * Run big-batch SGD using BBS_Armijo on logistic regression and make sure the
 * results are acceptable.  Use arma::fmat as the objective type.
 */
TEST_CASE("BBSArmijoLogisticRegressionFMatTest", "[BigBatchSGDTest]")
{
  arma::fmat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  // Now run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 40; batchSize < 50; batchSize += 1)
  {
    BBS_Armijo bbsgd(batchSize, 0.01, 0.1, 10000, 1e-6, true, true);

    LogisticRegression<arma::fmat> lr(shuffledData, shuffledResponses, 0.5);
    arma::fmat coordinates = lr.GetInitialPoint();
    bbsgd.Optimize(lr, coordinates);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses, coordinates);
    REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses,
        coordinates);
    REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
  }
}

#if ARMA_VERSION_MAJOR > 9 ||\
    (ARMA_VERSION_MAJOR == 9 && ARMA_VERSION_MINOR >= 400)

/**
 * Run big-batch SGD using BBS_BB on logistic regression and make sure the
 * results are acceptable.  Use arma::sp_mat as the objective type.
 */
TEST_CASE("BBSBBLogisticRegressionSpMatTest", "[BigBatchSGDTest]")
{
  arma::sp_mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  // Now run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 350; batchSize < 360; batchSize += 5)
  {
    BBS_BB bbsgd(batchSize, 0.005, 0.5, 10000, 1e-8, true, true);

    LogisticRegression<arma::sp_mat> lr(shuffledData, shuffledResponses, 0.5);
    arma::sp_mat coordinates = lr.GetInitialPoint();
    bbsgd.Optimize(lr, coordinates);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses, coordinates);
    REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses,
        coordinates);
    REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
  }
}

/**
 * Run big-batch SGD using BBS_Armijo on logistic regression and make sure the
 * results are acceptable.  Use arma::sp_mat as the objective type.
 */
TEST_CASE("BBSArmijoLogisticRegressionSpMatTest", "[BigBatchSGDTest]")
{
  arma::sp_mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  // Now run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 40; batchSize < 50; batchSize += 1)
  {
    BBS_Armijo bbsgd(batchSize, 0.01, 0.001, 10000, 1e-6, true, true);

    LogisticRegression<arma::sp_mat> lr(shuffledData, shuffledResponses, 0.5);
    arma::sp_mat coordinates = lr.GetInitialPoint();
    bbsgd.Optimize(lr, coordinates);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses, coordinates);
    REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses,
        coordinates);
    REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
  }
}

#endif
