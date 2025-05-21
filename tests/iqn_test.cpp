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

TEMPLATE_TEST_CASE("IQN_LogisticRegressionFunction", "[IQN]",
    arma::mat, arma::fmat)
{
  // Run on a couple of batch sizes.
  for (size_t batchSize = 1; batchSize < 9; batchSize += 4)
  {
    IQN iqn(0.01, batchSize, 5000, 0.01);
    LogisticRegressionFunctionTest<TestType>(iqn, 0.003, 0.006);
  }
}

#ifdef USE_COOT

TEMPLATE_TEST_CASE("IQN_LogisticRegressionFunction", "[IQN]",
    coot::mat, coot::fmat)
{
  IQN iqn(0.01, 10, 5000, 0.01);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      iqn, 0.003, 0.006);
}

#endif