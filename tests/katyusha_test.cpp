/**
 * @file katyusha_test.cpp
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

TEMPLATE_TEST_CASE("KatyushaLogisticRegressionTest", "[Katyusha]",
    arma::mat, arma::fmat)
{
  // Run with a couple of batch sizes.
  for (size_t batchSize = 30; batchSize < 45; batchSize += 5)
  {
    Katyusha optimizer(1.0, 10.0, batchSize, 100, 0, 1e-10, true);
    LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(
      optimizer, 0.015, 0.015);
  }
}

TEMPLATE_TEST_CASE("KatyushaProximalLogisticRegressionTest", "[Katyusha]",
    arma::mat, arma::fmat)
{
  // Run with a couple of batch sizes.
  for (size_t batchSize = 30; batchSize < 45; batchSize += 5)
  {
    KatyushaProximal optimizer(1.0, 10.0, batchSize, 100, 0, 1e-10, true);
    LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(
      optimizer, 0.015, 0.015);
  }
}

#if ARMA_VERSION_MAJOR > 9 ||\
    (ARMA_VERSION_MAJOR == 9 && ARMA_VERSION_MINOR >= 400)

/* TEST_CASE("KatyushaLogisticRegressionSpMatTest", "[KatyushaTest]") */
/* { */
/*   // Run with a couple of batch sizes. */
/*   for (size_t batchSize = 30; batchSize < 45; batchSize += 5) */
/*   { */
/*     Katyusha optimizer(1.0, 10.0, batchSize, 100, 0, 1e-10, true); */
/*     LogisticRegressionFunctionTest<arma::sp_mat>(optimizer, 0.015, 0.015); */
/*   } */
/* } */

/* TEST_CASE("KatyushaProximalLogisticRegressionSpMatTest", "[KatyushaTest]") */
/* { */
/*   // Run with a couple of batch sizes. */
/*   for (size_t batchSize = 30; batchSize < 45; batchSize += 5) */
/*   { */
/*     KatyushaProximal optimizer(1.0, 10.0, batchSize, 100, 0, 1e-10, true); */
/*     LogisticRegressionFunctionTest<arma::sp_mat>(optimizer, 0.015, 0.015); */
/*   } */
/* } */

#endif

#ifdef USE_COOT

TEMPLATE_TEST_CASE("KatyushaLogisticRegressionTest", "[Katyusha]",
    coot::mat, coot::fmat)
{
  Katyusha optimizer(1.0, 10.0, 10, 100, 0, 1e-10, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
    optimizer, 0.015, 0.015);
}

TEMPLATE_TEST_CASE("KatyushaProximalLogisticRegressionTest", "[Katyusha]",
    coot::mat, coot::fmat)
{
  KatyushaProximal optimizer(1.0, 10.0, 30, 100, 0, 1e-10, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
    optimizer, 0.015, 0.015);
}

#endif
