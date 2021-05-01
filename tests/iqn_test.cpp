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

/**
 * Run IQN on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("IQNLogisticRegressionTest", "[IQNTest]")
{
  // Run on a couple of batch sizes.
  for (size_t batchSize = 1; batchSize < 9; batchSize += 4)
  {
    IQN iqn(0.01, batchSize, 5000, 0.01);
    LogisticRegressionFunctionTest(iqn, 0.013, 0.016);
  }
}

/**
 * Run IQN on logistic regression and make sure the results are acceptable.  Use
 * arma::fmat.
 */
TEST_CASE("IQNLogisticRegressionFMatTest", "[IQNTest]")
{
  // Run on a couple of batch sizes.
  for (size_t batchSize = 1; batchSize < 9; batchSize += 4)
  {
    IQN iqn(0.001, batchSize, 5000, 0.01);
    LogisticRegressionFunctionTest<arma::fmat>(iqn, 0.013, 0.016);
  }
}
