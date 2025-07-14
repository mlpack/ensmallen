/**
 * @file eve_test.cpp
 * @author Marcus Edel
 *
 * Test file for the Eve optimizer.
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

TEMPLATE_TEST_CASE("Eve_LogisticRegressionFunction", "[Eve]", ENS_TEST_TYPES)
{
  Eve optimizer(1e-3, 1, 0.9, 0.999, 0.999, 1e-8, 10000, 500000, 1e-9, true);
  LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(optimizer);
}

TEMPLATE_TEST_CASE("Eve_SphereFunction", "[Eve]", ENS_TEST_TYPES)
{
  Eve optimizer(1e-3, 2, 0.9, 0.999, 0.999, 1e-8, 10000, 500000, 1e-9, true);
  FunctionTest<SphereFunctionType<TestType, arma::Row<size_t>>, TestType>(
      optimizer,
      Tolerances<TestType>::LargeObj,
      Tolerances<TestType>::LargeCoord);
}

TEMPLATE_TEST_CASE("Eve_StyblinskiTangFunction", "[Eve]", ENS_TEST_TYPES,
    ENS_SPARSE_TEST_TYPES)
{
  Eve optimizer(1e-3, 2, 0.9, 0.999, 0.999, 1e-8, 10000, 500000, 1e-9, true);
  FunctionTest<StyblinskiTangFunction<TestType, arma::Row<size_t>>, TestType>(
      optimizer,
      Tolerances<TestType>::LargeObj,
      Tolerances<TestType>::LargeCoord);
}

#ifdef USE_COOT

TEMPLATE_TEST_CASE("Eve_LogisticRegressionFunction", "[Eve]",
    coot::mat)
{
  Eve optimizer(1e-3, 1, 0.9, 0.999, 0.999, 1e-8, 10000, 500000, 1e-9, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      optimizer, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("Eve_SphereFunction", "[Eve]",
    coot::mat, coot::fmat)
{
  Eve optimizer(1e-3, 2, 0.9, 0.999, 0.999, 1e-8, 10000, 500000, 1e-9, true);
  FunctionTest<SphereFunctionType<TestType, coot::Row<size_t>>, TestType>(
      optimizer, 0.5, 0.1);
}

TEMPLATE_TEST_CASE("Eve_StyblinskiTangFunction", "[Eve]",
    coot::mat, coot::fmat)
{
  Eve optimizer(1e-3, 2, 0.9, 0.999, 0.999, 1e-8, 10000, 500000, 1e-9, true);
  FunctionTest<StyblinskiTangFunction<TestType, coot::Row<size_t>>, TestType>(
      optimizer, 0.5, 0.1);
}

#endif
