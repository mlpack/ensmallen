/**
 * @file ada_sqrt_test.cpp
 * @author Marcus Edel
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

TEMPLATE_TEST_CASE("AdaSqrt_LogisticRegressionFunction", "[AdaSqrt]",
    ENS_ALL_TEST_TYPES)
{
  AdaSqrt optimizer(0.25, 32, 10 * Tolerances<TestType>::Obj, 150000, 1e-9,
      true);
  // We allow a few trials for lower precision types because AdaSqrt can have
  // trouble converging in that case.
  LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(
      optimizer,
      Tolerances<TestType>::LRTrainAcc,
      Tolerances<TestType>::LRTestAcc,
      sizeof(typename TestType::elem_type) < 4 ? 5 : 1);
}

#ifdef USE_COOT

TEMPLATE_TEST_CASE("AdaSqrt_LogisticRegressionFunction", "[AdaSqrt]",
    coot::mat, coot::fmat)
{
  AdaSqrt optimizer(0.02, 32, 1e-8, 150000, 1e-9, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      optimizer);
}

#endif
