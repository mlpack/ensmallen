/**
 * @file yogi_test.cpp
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

TEMPLATE_TEST_CASE("Yogi_SphereFunction", "[Yogi]", ENS_ALL_TEST_TYPES)
{
  Yogi optimizer(0.5, 2, 0.7, 0.999, 1e-8, 500000, 1e-3, false);
  FunctionTest<SphereFunctionType<TestType, arma::Row<size_t>>, TestType>(
      optimizer,
      10 * Tolerances<TestType>::LargeObj,
      10 * Tolerances<TestType>::LargeCoord);
}

TEMPLATE_TEST_CASE("Yogi_McCormickFunction", "[Yogi]", ENS_ALL_TEST_TYPES)
{
  Yogi optimizer(0.5, 1, 0.7, 0.999, 1e-8, 500000, 1e-5, false);
  FunctionTest<McCormickFunction, TestType>(
      optimizer,
      Tolerances<TestType>::LargeObj,
      Tolerances<TestType>::LargeCoord);
}

TEMPLATE_TEST_CASE("Yogi_LogisticRegressionFunction", "[Yogi]",
    ENS_ALL_TEST_TYPES)
{
  Yogi optimizer;
  // For low-precision, we need to use a very small step size and some other
  // tuning to keep from diverging.
  size_t trials = 1;
  if (sizeof(typename TestType::elem_type) < 4)
  {
    optimizer.StepSize() = 5e-4;
    optimizer.BatchSize() = 16;
    optimizer.Tolerance() = -1.0; // Force maximum number of iterations.
    optimizer.MaxIterations() = 1000000;
    trials = 5;
  }
  LogisticRegressionFunctionTest<TestType>(
      optimizer,
      Tolerances<TestType>::LRTrainAcc,
      Tolerances<TestType>::LRTestAcc,
      trials);
}

#ifdef USE_COOT

TEMPLATE_TEST_CASE("Yogi_SphereFunction", "[Yogi]",
    coot::mat, coot::fmat)
{
  Yogi optimizer(0.5, 2, 0.7, 0.999, 1e-8, 500000, 1e-3, false);
  FunctionTest<SphereFunctionType<TestType, coot::Row<size_t>>, TestType>(
      optimizer, 0.5, 0.1);
}

TEMPLATE_TEST_CASE("Yogi_McCormickFunction", "[Yogi]",
    coot::mat)
{
  Yogi optimizer(0.5, 1, 0.7, 0.999, 1e-8, 500000, 1e-5, false);
  FunctionTest<McCormickFunction, TestType>(optimizer, 0.5, 0.1);
}

TEMPLATE_TEST_CASE("Yogi_LogisticRegressionFunction", "[Yogi]",
    coot::mat)
{
  Yogi optimizer;
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      optimizer, 0.003, 0.006);
}

#endif
