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

#include <ensmallen.hpp>
#include "catch.hpp"
#include "test_function_tools.hpp"
#include "test_types.hpp"

using namespace ens;
using namespace ens::test;

TEMPLATE_TEST_CASE("SWATS_LogisticRegressionFunction", "[SWATS]",
    ENS_ALL_TEST_TYPES)
{
  SWATS optimizer(0.003, 10, 0.9, 0.999, 1e-6, 600000, 1e-9, true);
  // We allow a few trials in case of poor convergence.
  LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(
      optimizer,
      Tolerances<TestType>::LRTrainAcc,
      Tolerances<TestType>::LRTestAcc,
      5);
}

TEMPLATE_TEST_CASE("SWATS_SphereFunction", "[SWATS]", ENS_ALL_TEST_TYPES)
{
  SWATS optimizer(0.1, 2, 0.9, 0.999, 1e-6, 500000, 1e-9, true);
  FunctionTest<SphereFunctionType<TestType, arma::Row<size_t>>, TestType>(
      optimizer,
      Tolerances<TestType>::LargeObj,
      Tolerances<TestType>::LargeCoord);
}

TEMPLATE_TEST_CASE("SWATS_StyblinskiTangFunction", "[SWATS]",
    ENS_ALL_TEST_TYPES, ENS_SPARSE_TEST_TYPES)
{
  SWATS optimizer(0.2, 2, 0.9, 0.999, 1e-6, 500000, 1e-9, true);
  FunctionTest<StyblinskiTangFunction<TestType, arma::Row<size_t>>, TestType>(
      optimizer,
      30 * Tolerances<TestType>::LargeObj,
      30 * Tolerances<TestType>::LargeCoord);
}

#ifdef USE_COOT

TEMPLATE_TEST_CASE("SWATS_LogisticRegressionFunction", "[SWATS]",
    coot::mat, coot::fmat)
{
  SWATS optimizer(1e-3, 10, 0.9, 0.999, 1e-6, 600000, 1e-9, true);
  // We allow a few trials in case of poor convergence.
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      optimizer, 0.003, 0.006, 5);
}

TEMPLATE_TEST_CASE("SWATS_SphereFunction", "[SWATS]",
    coot::mat, coot::fmat)
{
  SWATS optimizer(1e-3, 2, 0.9, 0.999, 1e-6, 500000, 1e-9, true);
  FunctionTest<SphereFunctionType<TestType, coot::Row<size_t>>, TestType>(
      optimizer, 1.0, 0.1);
}

TEMPLATE_TEST_CASE("SWATS_StyblinskiTangFunction", "[SWATS]",
    coot::mat, coot::fmat)
{
  SWATS optimizer(1e-3, 2, 0.9, 0.999, 1e-6, 500000, 1e-9, true);
  FunctionTest<StyblinskiTangFunction<TestType, coot::Row<size_t>>,
      TestType>(optimizer, 3.0, 0.3);
}

#endif
