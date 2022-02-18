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

using namespace ens;
using namespace ens::test;

/**
 * Test the Lookahead - Adam optimizer on the Sphere function.
 */
TEST_CASE("LookaheadAdamSphereFunctionTest", "[LookaheadTest]")
{
  Lookahead<> optimizer(0.5, 5, 100000, 1e-5, NoDecay(), false, true);
  optimizer.BaseOptimizer().StepSize() = 0.1;
  optimizer.BaseOptimizer().BatchSize() = 2;
  optimizer.BaseOptimizer().Beta1() = 0.7;
  optimizer.BaseOptimizer().Tolerance() = 1e-15;
  // We allow a few trials.
  FunctionTest<SphereFunction>(optimizer, 0.5, 0.2, 3);
}

/**
 * Test the Lookahead - AdaGrad optimizer on the SphereFunction function.
 */
TEST_CASE("LookaheadAdaGradSphereFunction", "[LookaheadTest]")
{
  AdaGrad adagrad(0.99, 1, 1e-8, 5, 1e-15, true);
  Lookahead<AdaGrad> optimizer(adagrad, 0.5, 5, 5000000, 1e-15, NoDecay(),
      false, true);
  FunctionTest<SphereFunction>(optimizer, 0.5, 0.2, 3);
}

/**
 * Run Lookahead - Adam on logistic regression and make sure the results are
 * acceptable.
 */
TEST_CASE("LookaheadAdamLogisticRegressionTest","[LookaheadTest]")
{
  Adam adam(0.001, 32, 0.9, 0.999, 1e-8, 5, 1e-19);
  Lookahead<Adam> optimizer(adam, 0.5, 20, 100000, 1e-15, NoDecay(),
      false, true);
  LogisticRegressionFunctionTest(optimizer, 0.003, 0.006);
}

/**
 * Test the Lookahead - Adam optimizer on the Sphere function (float).
 */
TEST_CASE("LookaheadAdamSimpleSphereFunctionFloat", "[LookaheadTest]")
{
  Adam adam(0.001, 1, 0.9, 0.999, 1e-8, 5, 1e-19, false, true);
  Lookahead<Adam> optimizer(adam, 0.5, 5, 100000, 1e-15, NoDecay(),
      false, true);
  FunctionTest<SphereFunction, arma::fmat>(optimizer, 0.5, 0.2, 3);
}
