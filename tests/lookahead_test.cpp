/**
 * @file lookahead_test.cpp
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

TEMPLATE_TEST_CASE("LookaheadAdam_SphereFunction", "[Lookahead]",
    ENS_ALL_TEST_TYPES)
{
  Lookahead<> optimizer(0.5, 5, 100000, 1e-5, NoDecay(), false, true);
  optimizer.BaseOptimizer().StepSize() = 0.5;
  optimizer.BaseOptimizer().BatchSize() = 2;
  optimizer.BaseOptimizer().Beta1() = 0.7;
  optimizer.BaseOptimizer().Tolerance() = Tolerances<TestType>::Obj;
  // We allow a few trials.
  FunctionTest<SphereFunctionType<TestType, arma::Row<size_t>>, TestType>(
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
  Lookahead<AdaGrad> optimizer(adagrad, 0.5, 5, 5000000,
      Tolerances<TestType>::Obj / 100, NoDecay(), false, true);
  FunctionTest<SphereFunctionType<TestType, arma::Row<size_t>>, TestType>(
      optimizer,
      10 * Tolerances<TestType>::LargeObj,
      10 * Tolerances<TestType>::LargeCoord,
      3);
}

TEMPLATE_TEST_CASE("LookaheadAdam_LogisticRegressionFunction", "[Lookahead]",
    ENS_ALL_TEST_TYPES)
{
  Adam adam(0.008, 4, 0.9, 0.999, Tolerances<TestType>::Obj, 5,
      Tolerances<TestType>::Obj / 10);
  adam.ResetPolicy() = false;
  Lookahead<Adam> optimizer(adam, 0.5, 25, 100000, Tolerances<TestType>::Obj,
      NoDecay(), false, true);
  LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(
      optimizer,
      Tolerances<TestType>::LRTrainAcc,
      Tolerances<TestType>::LRTestAcc);
}

#ifdef USE_COOT

TEMPLATE_TEST_CASE("LookaheadAdam_SphereFunction", "[Lookahead]",
    coot::mat)
{
  Lookahead<> optimizer(0.5, 5, 100000, 1e-5, NoDecay(), false, true);
  optimizer.BaseOptimizer().StepSize() = 0.1;
  optimizer.BaseOptimizer().BatchSize() = 2;
  optimizer.BaseOptimizer().Beta1() = 0.7;
  optimizer.BaseOptimizer().Tolerance() = 1e-15;
  // We allow a few trials.
  FunctionTest<SphereFunctionType<TestType, coot::Row<size_t>>, TestType>(
    optimizer, 0.5, 0.2, 3);
}

TEMPLATE_TEST_CASE("LookaheadAdaGrad_SphereFunction", "[Lookahead]",
    coot::mat)
{
  AdaGrad adagrad(0.99, 1, 1e-8, 5, 1e-15, true);
  adagrad.ResetPolicy() = false;
  Lookahead<AdaGrad> optimizer(adagrad, 0.5, 5, 5000000, 1e-15, NoDecay(),
      false, true);
  FunctionTest<SphereFunctionType<TestType, coot::Row<size_t>>, TestType>(
      optimizer, 0.5, 0.2, 3);
}

TEMPLATE_TEST_CASE("LookaheadAdam_LogisticRegressionFunction", "[Lookahead]",
    coot::mat)
{
  Adam adam(0.001, 32, 0.9, 0.999, 1e-8, 5, 1e-19);
  adam.ResetPolicy() = false;
  Lookahead<Adam> optimizer(adam, 0.5, 20, 100000, 1e-15, NoDecay(),
      false, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      optimizer, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("LookaheadAdam_SimpleSphereFunction", "[Lookahead]",
    coot::fmat)
{
  Adam adam(0.001, 1, 0.9, 0.999, 1e-8, 5, 1e-19, false, true);
  adam.ResetPolicy() = false;
  Lookahead<Adam> optimizer(adam, 0.5, 5, 100000, 1e-15, NoDecay(),
      false, true);
  FunctionTest<SphereFunctionType<TestType, coot::Row<size_t>>, TestType>(
      optimizer, 0.5, 0.2, 3);
}

#endif
