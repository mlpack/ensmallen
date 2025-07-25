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
#include "test_types.hpp"

using namespace ens;
using namespace ens::test;

TEMPLATE_TEST_CASE("SVRG_LogisticRegressionFunction", "[SVRG]",
    ENS_ALL_TEST_TYPES, ENS_SPARSE_TEST_TYPES)
{
  // Run SVRG with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 50; batchSize += 5)
  {
    SVRG optimizer(0.005, batchSize, 300, 0, 1e-5, true);
    LogisticRegressionFunctionTest<TestType>(
        optimizer,
        5 * Tolerances<TestType>::LRTrainAcc,
        5 * Tolerances<TestType>::LRTestAcc);
  }
}

TEMPLATE_TEST_CASE("SVRG_BB_LogisticRegressionFunction", "[SVRG_BB]",
    ENS_ALL_TEST_TYPES, ENS_SPARSE_TEST_TYPES)
{
  // Run SVRG with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 50; batchSize += 5)
  {
    SVRG_BB optimizer(0.005, batchSize, 300, 0, 1e-5, true, SVRGUpdate(),
        BarzilaiBorweinDecay(0.1));
    LogisticRegressionFunctionTest(
        optimizer,
        5 * Tolerances<TestType>::LRTrainAcc,
        5 * Tolerances<TestType>::LRTestAcc);
  }
}

#ifdef USE_COOT

TEMPLATE_TEST_CASE("SVRG_LogisticRegressionFunction", "[SVRG]",
    coot::mat, coot::fmat)
{
  SVRG optimizer(0.005, 50, 300, 0, 1e-5, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      optimizer, 0.015, 0.015);
}

TEMPLATE_TEST_CASE("SVRG_BB_LogisticRegressionFunction", "[SVRG_BB]",
    coot::mat, coot::fmat)
{
  SVRG_BB optimizer(0.005, 50, 300, 0, 1e-5, true, SVRGUpdate(),
      BarzilaiBorweinDecay(0.1));
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      optimizer, 0.015, 0.015);
}

#endif
