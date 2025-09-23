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
#if defined(ENS_USE_COOT)
  #include <armadillo>
  #include <bandicoot>
#endif
#include <ensmallen.hpp>
#include "catch.hpp"
#include "test_function_tools.hpp"
#include "test_types.hpp"

using namespace ens;
using namespace ens::test;

// NOTE: IQN cannot use arma::hmat because pinv() is required.

TEMPLATE_TEST_CASE("IQN_LogisticRegressionFunction", "[IQN]", ENS_TEST_TYPES)
{
  // Run on a couple of batch sizes.
  for (size_t batchSize = 1; batchSize < 9; batchSize += 4)
  {
    IQN iqn(0.01, batchSize, 200000, 0.01);
    // It could take a few attempts to converge.
    LogisticRegressionFunctionTest<TestType>(iqn,
        Tolerances<TestType>::LRTrainAcc,
        Tolerances<TestType>::LRTestAcc,
        5);
  }
}
