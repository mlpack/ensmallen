/**
 * @file lookahead_test.cpp
 * @author Marcus Edel
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

TEMPLATE_TEST_CASE("LookaheadAdam_SphereFunction", "[Lookahead]",
    ENS_ALL_TEST_TYPES)
{
  Lookahead<> optimizer(2.5, 5, 100000, 1e-5, NoDecay(), false, true);
  optimizer.BaseOptimizer().StepSize() = 0.5;
  optimizer.BaseOptimizer().BatchSize() = 2;
  optimizer.BaseOptimizer().Beta1() = 0.7;
  optimizer.BaseOptimizer().Tolerance() = Tolerances<TestType>::Obj;
  // We allow a few trials.
  FunctionTest<SphereFunction, TestType>(
      optimizer,
      10 * Tolerances<TestType>::LargeObj,
      10 * Tolerances<TestType>::LargeCoord,
      3);
}

TEMPLATE_TEST_CASE("LookaheadAdaGrad_SphereFunction", "[Lookahead]",
    ENS_ALL_TEST_TYPES)
{
  AdaGrad adagrad(0.99, 1, 1e-8, 5, Tolerances<TestType>::Obj / 100, true);
  adagrad.ResetPolicy() = false;
  Lookahead<AdaGrad> optimizer(adagrad, 2.5, 5, 5000000,
      Tolerances<TestType>::Obj / 100, NoDecay(), false, true);
  FunctionTest<SphereFunction, TestType>(
      optimizer,
      10 * Tolerances<TestType>::LargeObj,
      10 * Tolerances<TestType>::LargeCoord,
      3);
}

TEMPLATE_TEST_CASE("LookaheadAdam_LogisticRegressionFunction", "[Lookahead]",
    ENS_ALL_TEST_TYPES)
{
  Adam adam(0.032, 4, 0.9, 0.999, Tolerances<TestType>::Obj, 5,
      Tolerances<TestType>::Obj / 10);
  adam.ResetPolicy() = false;
  Lookahead<Adam> optimizer(adam, 12.5, 25, 100000, Tolerances<TestType>::Obj,
      NoDecay(), false, true);
  LogisticRegressionFunctionTest<TestType>(
      optimizer,
      Tolerances<TestType>::LRTrainAcc,
      Tolerances<TestType>::LRTestAcc);
}
