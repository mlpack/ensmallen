/**
 * @file swats_test.cpp
 * @author Marcus Edel
 *
 * Test file for the SWATS optimizer.
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

TEMPLATE_TEST_CASE("SWATS_LogisticRegressionFunction", "[SWATS]",
    ENS_ALL_TEST_TYPES)
{
  SWATS optimizer(0.03, 10, 0.9, 0.999, 1e-6, 600000, 1e-9, true);
  // We allow a few trials in case of poor convergence.
  LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(
      optimizer,
      Tolerances<TestType>::LRTrainAcc,
      Tolerances<TestType>::LRTestAcc,
      5);
}

TEMPLATE_TEST_CASE("SWATS_SphereFunction", "[SWATS]", ENS_ALL_TEST_TYPES)
{
  SWATS optimizer(0.2, 2, 0.9, 0.999, 1e-6, 500000, 1e-9, true);
  FunctionTest<SphereFunction, TestType>(
      optimizer,
      Tolerances<TestType>::LargeObj,
      Tolerances<TestType>::LargeCoord);
}

TEMPLATE_TEST_CASE("SWATS_StyblinskiTangFunction", "[SWATS]",
    ENS_ALL_TEST_TYPES, ENS_SPARSE_TEST_TYPES)
{
  SWATS optimizer(0.4, 2, 0.9, 0.999, 1e-6, 500000, 1e-9, true);
  FunctionTest<StyblinskiTangFunction, TestType>(
      optimizer,
      30 * Tolerances<TestType>::LargeObj,
      30 * Tolerances<TestType>::LargeCoord);
}
