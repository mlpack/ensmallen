/**
 * @file svrg_test.cpp
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
 * Run SVRG on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("SVRGLogisticRegressionTest", "[SVRGTest]")
{
  // Run SVRG with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 50; batchSize += 5)
  {
    SVRG optimizer(0.005, batchSize, 300, 0, 1e-5, true);
    LogisticRegressionFunctionTest(optimizer, 0.015, 0.015);
  }
}

/**
 * Run SVRG_BB on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("SVRGBBLogisticRegressionTest", "[SVRGTest]")
{
  // Run SVRG with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 50; batchSize += 5)
  {
    SVRG_BB optimizer(0.005, batchSize, 300, 0, 1e-5, true, SVRGUpdate(),
        BarzilaiBorweinDecay(0.1));
    LogisticRegressionFunctionTest(optimizer, 0.015, 0.015);
  }
}

/**
 * Run SVRG on logistic regression and make sure the results are acceptable.
 * Use arma::fmat.
 */
TEST_CASE("SVRGLogisticRegressionFMatTest", "[SVRGTest]")
{
  // Run SVRG with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 50; batchSize += 5)
  {
    SVRG optimizer(0.005, batchSize, 300, 0, 1e-5, true);
    LogisticRegressionFunctionTest<arma::fmat>(optimizer, 0.015, 0.015);
  }
}

/**
 * Run SVRG_BB on logistic regression and make sure the results are acceptable.
 * Use arma::fmat.
 */
TEST_CASE("SVRGBBLogisticRegressionFMatTest", "[SVRGTest]")
{
  // Run SVRG_BB with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 50; batchSize += 5)
  {
    SVRG_BB optimizer(0.005, batchSize, 300, 0, 1e-5, true,
        SVRGUpdate(), BarzilaiBorweinDecay(0.1));
    LogisticRegressionFunctionTest<arma::fmat>(optimizer, 0.015, 0.015);
  }
}

#if ARMA_VERSION_MAJOR > 9 ||\
    (ARMA_VERSION_MAJOR == 9 && ARMA_VERSION_MINOR >= 400)

/**
 * Run SVRG on logistic regression and make sure the results are acceptable.
 * Use arma::sp_mat.
 */
TEST_CASE("SVRGLogisticRegressionSpMatTest", "[SVRGTest]")
{
  // Run SVRG with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 50; batchSize += 5)
  {
    SVRG optimizer(0.005, batchSize, 300, 0, 1e-5, true);
    LogisticRegressionFunctionTest<arma::sp_mat>(optimizer, 0.015, 0.015);
  }
}

/**
 * Run SVRG_BB on logistic regression and make sure the results are acceptable.
 * Use arma::sp_mat.
 */
TEST_CASE("SVRGBBLogisticRegressionSpMatTest", "[SVRGTest]")
{
  // Run SVRG with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 50; batchSize += 5)
  {
    SVRG_BB optimizer(0.005, batchSize, 300, 0, 1e-5, true,
        SVRGUpdate(), BarzilaiBorweinDecay(0.1));
    LogisticRegressionFunctionTest<arma::sp_mat>(optimizer, 0.015, 0.015);
  }
}

#endif
