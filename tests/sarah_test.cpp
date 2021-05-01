/**
 * @file sarah_test.cpp
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
 * Run SARAH on logistic regression and make sure the results are
 * acceptable.
 */
TEST_CASE("SARAHLogisticRegressionTest","[SARAHTest]")
{
  // Run SARAH with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 45; batchSize += 5)
  {
    SARAH optimizer(0.01, batchSize, 250, 0, 1e-5, true);
    LogisticRegressionFunctionTest(optimizer, 0.015, 0.015);
  }
}

/**
 * Run SARAH_Plus on logistic regression and make sure the results are
 * acceptable.
 */
TEST_CASE("SARAHPlusLogisticRegressionTest","[SARAHTest]")
{
  // Run SARAH_Plus with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 45; batchSize += 5)
  {
    SARAH_Plus optimizer(0.01, batchSize, 250, 0, 1e-5, true);
    LogisticRegressionFunctionTest(optimizer, 0.015, 0.015);
  }
}

/**
 * Run SARAH on logistic regression and make sure the results are
 * acceptable.  Use arma::fmat.
 */
TEST_CASE("SARAHLogisticRegressionFMatTest","[SARAHTest]")
{
  // Run SARAH with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 45; batchSize += 5)
  {
    SARAH optimizer(0.01, batchSize, 250, 0, 1e-5, true);
    LogisticRegressionFunctionTest<arma::fmat>(optimizer, 0.015, 0.015);
  }
}

/**
 * Run SARAH_Plus on logistic regression and make sure the results are
 * acceptable.  Use arma::fmat.
 */
TEST_CASE("SARAHPlusLogisticRegressionFMatTest","[SARAHTest]")
{
  // Run SARAH_Plus with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 45; batchSize += 5)
  {
    SARAH_Plus optimizer(0.01, batchSize, 250, 0, 1e-5, true);
    LogisticRegressionFunctionTest<arma::fmat>(optimizer, 0.015, 0.015);
  }
}

#if ARMA_VERSION_MAJOR > 9 ||\
    (ARMA_VERSION_MAJOR == 9 && ARMA_VERSION_MINOR >= 400)

/**
 * Run SARAH on logistic regression and make sure the results are
 * acceptable.  Use arma::sp_mat.
 */
TEST_CASE("SARAHLogisticRegressionSpMatTest","[SARAHTest]")
{
  // Run SARAH with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 45; batchSize += 5)
  {
    SARAH optimizer(0.01, batchSize, 250, 0, 1e-5, true);
    LogisticRegressionFunctionTest<arma::sp_mat>(optimizer, 0.015, 0.015);
  }
}

/**
 * Run SARAH_Plus on logistic regression and make sure the results are
 * acceptable.  Use arma::sp_mat.
 */
TEST_CASE("SARAHPlusLogisticRegressionSpMatTest","[SARAHTest]")
{
  // Run SARAH_Plus with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 45; batchSize += 5)
  {
    SARAH_Plus optimizer(0.01, batchSize, 250, 0, 1e-5, true);
    LogisticRegressionFunctionTest<arma::sp_mat>(optimizer, 0.015, 0.015);
  }
}

#endif
