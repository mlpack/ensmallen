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

TEMPLATE_TEST_CASE("SARAHLogisticRegressionTest","[SARAH]",
    arma::mat, arma::fmat)
{
  // Run SARAH with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 45; batchSize += 5)
  {
    SARAH optimizer(0.01, batchSize, 250, 0, 1e-5, true);
    LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(
        optimizer, 0.003, 0.006);
  }
}

TEMPLATE_TEST_CASE("SARAHPlusLogisticRegressionTest","[SARAH]",
    arma::mat, arma::fmat)
{
  // Run SARAH_Plus with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 45; batchSize += 5)
  {
    SARAH_Plus optimizer(0.01, batchSize, 250, 0, 1e-5, true);
    LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(
        optimizer, 0.015, 0.015);
  }
}

#ifdef USE_COOT

TEMPLATE_TEST_CASE("SARAHLogisticRegressionTest","[SARAH]",
    coot::mat, coot::fmat)
{
  // Run SARAH with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 45; batchSize += 5)
  {
    SARAH optimizer(0.01, batchSize, 250, 0, 1e-5, true);
    LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
        optimizer, 0.015, 0.015);
  }
}

TEMPLATE_TEST_CASE("SARAHPlusLogisticRegressionTest","[SARAH]",
    coot::mat, coot::fmat)
{
  // Run SARAH_Plus with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 45; batchSize += 5)
  {
    SARAH_Plus optimizer(0.01, batchSize, 250, 0, 1e-5, true);
    LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
        optimizer, 0.015, 0.015);
  }
}

#endif
