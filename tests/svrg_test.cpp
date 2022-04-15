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

TEMPLATE_TEST_CASE("SVRGLogisticRegressionTest", "[SVRG]",
    arma::mat, arma::fmat)
{
  // Run SVRG with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 50; batchSize += 5)
  {
    SVRG optimizer(0.005, batchSize, 300, 0, 1e-5, true);
    LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(
        optimizer, 0.015, 0.015);
  }
}

TEMPLATE_TEST_CASE("SVRGBBLogisticRegressionTest", "[SVRG]",
    arma::mat, arma::fmat)
{
  // Run SVRG with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 50; batchSize += 5)
  {
    SVRG_BB optimizer(0.005, batchSize, 300, 0, 1e-5, true, SVRGUpdate(),
        BarzilaiBorweinDecay(0.1));
    LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(
        optimizer, 0.015, 0.015);
  }
}

#if ARMA_VERSION_MAJOR > 9 ||\
    (ARMA_VERSION_MAJOR == 9 && ARMA_VERSION_MINOR >= 400)

/* TEST_CASE("SVRGLogisticRegressionSpMatTest", "[SVRG]") */
/* { */
/*   // Run SVRG with a couple of batch sizes. */
/*   for (size_t batchSize = 35; batchSize < 50; batchSize += 5) */
/*   { */
/*     SVRG optimizer(0.005, batchSize, 300, 0, 1e-5, true); */
/*     LogisticRegressionFunctionTest<arma::sp_mat>(optimizer, 0.015, 0.015); */
/*   } */
/* } */

/* TEST_CASE("SVRGBBLogisticRegressionSpMatTest", "[SVRG]") */
/* { */
/*   // Run SVRG with a couple of batch sizes. */
/*   for (size_t batchSize = 35; batchSize < 50; batchSize += 5) */
/*   { */
/*     SVRG_BB optimizer(0.005, batchSize, 300, 0, 1e-5, true, */
/*         SVRGUpdate(), BarzilaiBorweinDecay(0.1)); */
/*     LogisticRegressionFunctionTest<arma::sp_mat>(optimizer, 0.015, 0.015); */
/*   } */
/* } */

#endif


#ifdef USE_COOT

TEMPLATE_TEST_CASE("SVRGLogisticRegressionTest", "[SVRG]",
    coot::mat, coot::fmat)
{
  // Run SVRG with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 50; batchSize += 5)
  {
    SVRG optimizer(0.005, batchSize, 300, 0, 1e-5, true);
    LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
        optimizer, 0.015, 0.015);
  }
}

TEMPLATE_TEST_CASE("SVRGBBLogisticRegressionTest", "[SVRG]",
    coot::mat, coot::fmat)
{
  // Run SVRG with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 50; batchSize += 5)
  {
    SVRG_BB optimizer(0.005, batchSize, 300, 0, 1e-5, true, SVRGUpdate(),
        BarzilaiBorweinDecay(0.1));
    LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
        optimizer, 0.015, 0.015);
  }
}

#endif
