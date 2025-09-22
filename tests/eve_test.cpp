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

TEMPLATE_TEST_CASE("Eve_LogisticRegressionFunction", "[Eve]",
    ENS_ALL_CPU_TEST_TYPES)
{
  Eve optimizer(0.01, 1, 0.9, 0.999, 0.999, Tolerances<TestType>::Obj, 10000,
      500000, Tolerances<TestType>::Obj / 10, true);
  LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(optimizer);
}

TEMPLATE_TEST_CASE("Eve_SphereFunction", "[Eve]", ENS_ALL_TEST_TYPES)
{
  Eve optimizer(0.02, 2, 0.9, 0.999, 0.999, Tolerances<TestType>::Obj, 10000,
      500000, Tolerances<TestType>::Obj / 10, true);
  FunctionTest<SphereFunction, TestType>(
      optimizer,
      Tolerances<TestType>::LargeObj,
      Tolerances<TestType>::LargeCoord);
}

TEMPLATE_TEST_CASE("Eve_StyblinskiTangFunction", "[Eve]", ENS_ALL_TEST_TYPES,
    ENS_SPARSE_TEST_TYPES)
{
  Eve optimizer(0.16, 2, 0.9, 0.999, 0.999, Tolerances<TestType>::Obj, 10000,
      500000, Tolerances<TestType>::Obj / 10, true);
  FunctionTest<StyblinskiTangFunction, TestType>(
      optimizer,
      Tolerances<TestType>::LargeObj,
      Tolerances<TestType>::LargeCoord);
}
