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
#include "test_types.hpp"

using namespace ens;
using namespace ens::test;

TEMPLATE_TEST_CASE("SARAH_LogisticRegressionFunction", "[SARAH]",
    ENS_TEST_TYPES)
{
  // Run SARAH with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 45; batchSize += 5)
  {
    SARAH optimizer(0.01, batchSize, 250, 0, 1e-5, true);
    LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(optimizer);
  }
}

TEMPLATE_TEST_CASE("SARAH_Plus_LogisticRegressionFunction", "[SARAH]",
    ENS_TEST_TYPES)
{
  // Run SARAH_Plus with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 45; batchSize += 5)
  {
    SARAH_Plus optimizer(0.01, batchSize, 250, 0, 1e-5, true);
    LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(optimizer);
  }
}

#ifdef USE_COOT

TEMPLATE_TEST_CASE("SARAH_LogisticRegressionFunction", "[SARAH]",
    coot::mat, coot::fmat)
{
  SARAH optimizer(0.01, 45, 250, 0, 1e-5, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      optimizer, 0.015, 0.015);
}

TEMPLATE_TEST_CASE("SARAH_Plus_LogisticRegressionFunction", "[SARAH]",
    coot::mat, coot::fmat)
{
  SARAH_Plus optimizer(0.01, 45, 250, 0, 1e-5, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      optimizer, 0.015, 0.015);
}

#endif
