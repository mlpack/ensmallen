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
  // Run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 350; batchSize < 360; batchSize += 5)
  {
    BBS_BB bbsgd(batchSize, 0.001, 0.1, 10000, 1e-8, true, true);
    LogisticRegressionFunctionTest(bbsgd, 0.003, 0.006, 3);
  }
}

/**
 * Run big-batch SGD using BBS_Armijo on logistic regression and make sure the
 * results are acceptable.
 */
TEST_CASE("BBSArmijoLogisticRegressionTest", "[BigBatchSGDTest]")
{
  // Run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 40; batchSize < 50; batchSize += 1)
  {
    BBS_Armijo bbsgd(batchSize, 0.005, 0.1, 10000, 1e-6, true, true);
    LogisticRegressionFunctionTest(bbsgd, 0.003, 0.006, 3);
  }
}

/**
 * Run big-batch SGD using BBS_BB on logistic regression and make sure the
 * results are acceptable.  Use arma::fmat as the objective type.
 */
TEST_CASE("BBSBBLogisticRegressionFMatTest", "[BigBatchSGDTest]")
{
  // Run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 350; batchSize < 360; batchSize += 5)
  {
    BBS_BB bbsgd(batchSize, 0.001, 0.1, 10000, 1e-8, true, true);
    LogisticRegressionFunctionTest<arma::fmat>(bbsgd, 0.003, 0.006, 3);
  }
}

/**
 * Run big-batch SGD using BBS_Armijo on logistic regression and make sure the
 * results are acceptable.  Use arma::fmat as the objective type.
 */
TEST_CASE("BBSArmijoLogisticRegressionFMatTest", "[BigBatchSGDTest]")
{
  // Run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 40; batchSize < 50; batchSize += 1)
  {
    BBS_Armijo bbsgd(batchSize, 0.01, 0.1, 10000, 1e-6, true, true);
    LogisticRegressionFunctionTest<arma::fmat>(bbsgd, 0.003, 0.006, 5);
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
  // Run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 350; batchSize < 360; batchSize += 5)
  {
    BBS_BB bbsgd(batchSize, 0.005, 0.5, 10000, 1e-8, true, true);
    LogisticRegressionFunctionTest<arma::sp_mat>(bbsgd, 0.003, 0.006, 3);
  }
}

/**
 * Run big-batch SGD using BBS_Armijo on logistic regression and make sure the
 * results are acceptable.  Use arma::sp_mat as the objective type.
 */
TEST_CASE("BBSArmijoLogisticRegressionSpMatTest", "[BigBatchSGDTest]")
{
  // Run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 40; batchSize < 50; batchSize += 1)
  {
    BBS_Armijo bbsgd(batchSize, 0.01, 0.001, 10000, 1e-6, true, true);
    LogisticRegressionFunctionTest<arma::sp_mat>(bbsgd, 0.003, 0.006, 3);
  }
}

#endif
