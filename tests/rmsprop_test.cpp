/**
 * @file rmsprop_test.cpp
 * @author Marcus Edel
 * @author Conrad Sanderson
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

TEMPLATE_TEST_CASE("RMSProp_LogisticRegressionFunction", "[RMSProp]",
    ENS_ALL_TEST_TYPES, ENS_SPARSE_TEST_TYPES)
{
  RMSProp optimizer(0.32);
  LogisticRegressionFunctionTest<TestType>(
      optimizer,
      Tolerances<TestType>::LRTrainAcc,
      Tolerances<TestType>::LRTestAcc,
      // Low-precision may need a few trials because it sometimes diverges
      // (gradient or update is too large).
      (sizeof(typename TestType::elem_type) < 4) ? 5 : 1);
}
