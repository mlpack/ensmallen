/**
 * @file ftml_test.cpp
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Test file for the FTML optimizer.
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

TEMPLATE_TEST_CASE("FTML_LogisticRegressionFunction", "[FTML]",
    ENS_ALL_TEST_TYPES)
{
  FTML optimizer(0.005, 1, 0.9, 0.999, Tolerances<TestType>::Obj, 100000,
      Tolerances<TestType>::Obj * 10, true);
  LogisticRegressionFunctionTest<TestType>(optimizer);
}

TEMPLATE_TEST_CASE("FTML_SphereFunction", "[FTML]", ENS_ALL_TEST_TYPES)
{
  FTML optimizer(0.06, 2, 0.9, 0.999, Tolerances<TestType>::Obj / 100, 500000,
      Tolerances<TestType>::Obj / 100, true);
  FunctionTest<SphereFunction, TestType>(
      optimizer,
      Tolerances<TestType>::LargeObj,
      Tolerances<TestType>::LargeCoord);
}

TEMPLATE_TEST_CASE("FTML_StyblinskiTangFunction", "[FTML]", ENS_ALL_TEST_TYPES)
{
  FTML optimizer(0.8, 2, 0.9, 0.999, Tolerances<TestType>::Obj, 100000,
      Tolerances<TestType>::Obj / 100, true);
  FunctionTest<StyblinskiTangFunction, TestType>(
      optimizer,
      10 * Tolerances<TestType>::LargeObj,
      Tolerances<TestType>::LargeCoord);
}

// A test with sp_mat is not done, because FTML uses some parts internally that
// assume the objective is dense.
