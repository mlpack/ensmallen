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
#include "test_types.hpp"

using namespace ens;
using namespace ens::test;

TEMPLATE_TEST_CASE("BBS_BB_LogisticRegressionFunction", "[BigBatchSGD]",
    ENS_ALL_TEST_TYPES)
{
  // Run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 350; batchSize < 360; batchSize += 5)
  {
    BBS_BB bbsgd(batchSize, 0.001, 0.1, 10000, 1e-8, true, true);
    LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(bbsgd);
  }
}

TEMPLATE_TEST_CASE("BBS_Armijo_LogisticRegressionFunction", "[BigBatchSGD]",
    ENS_ALL_TEST_TYPES)
{
  // Run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 40; batchSize < 50; batchSize += 1)
  {
    BBS_Armijo bbsgd(batchSize, 0.005, 0.1, 10000, 1e-6, true, true);
    LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(bbsgd);
  }
}

#ifdef USE_COOT

TEMPLATE_TEST_CASE("BBS_BB_LogisticRegressionFunction", "[BigBatchSGD]",
    coot::mat, coot::fmat)
{
  BBS_BB bbsgd(350, 0.001, 0.1, 10000, 1e-8, true, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(bbsgd);
}

TEMPLATE_TEST_CASE("BBS_Armijo_LogisticRegressionFunction", "[BigBatchSGD]",
    coot::mat, coot::fmat)
{
  BBS_Armijo bbsgd(40, 0.005, 0.1, 10000, 1e-6, true, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(bbsgd);
}

#endif
