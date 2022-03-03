/**
 * @file demon_adam_test.cpp
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
 * Run DemonAdam on logistic regression and make sure the results are
 * acceptable.
 */
TEST_CASE("DemonAdamLogisticRegressionTest", "[DemonAdamTest]")
{
  DemonAdam optimizer(0.2, 32, 0.9, 0.9, 0.999, 1e-8,
      10000, 1e-9, true, true, true);
  LogisticRegressionFunctionTest(optimizer, 0.003, 0.006, 6);
}

/**
 * Test the Adam optimizer on the Sphere function.
 */
TEST_CASE("DemonAdamSphereFunctionTest", "[DemonAdamTest]")
{
  SphereFunction f(2);
  DemonAdam optimizer(0.5, 2, 0.9);
  FunctionTest<SphereFunction, arma::mat>(optimizer, 1.0, 0.1);
}

/**
 * Test the DemonAdam optimizer on the Matyas function.
 */
TEST_CASE("DemonAdamMatyasFunctionTest", "[DemonAdamTest]")
{
  DemonAdam optimizer(0.5, 1, 0.9);
  FunctionTest<MatyasFunction, arma::mat>(optimizer, 0.1, 0.01);
}

/**
 * Test the Adam optimizer on the Sphere function.
 */
TEST_CASE("DemonAdamSphereFunctionTestFloat", "[DemonAdamTest]")
{
  DemonAdam optimizer(0.5, 2, 0.9);
  FunctionTest<SphereFunction, arma::sp_mat>(optimizer, 1.0, 0.1);
}

/**
 * Test the DemonAdam optimizer on the Matyas function.
 */
TEST_CASE("DemonAdamMatyasFunctionTestFloat", "[DemonAdamTest]")
{
  DemonAdam optimizer(0.5, 1, 0.9);
  FunctionTest<MatyasFunction, arma::fmat>(optimizer, 0.1, 0.01);
}

/**
 * Run DemonAdam (AdaMax update) on logistic regression and make sure the
 * results are acceptable.
 */
TEST_CASE("DemonAdaMaxLogisticRegressionTest", "[DemonAdamTest]")
{
  DemonAdamType<AdaMaxUpdate> optimizer(0.5, 10, 0.9, 0.9, 0.999, 1e-8,
      10000, 1e-9, true, true, true);
  LogisticRegressionFunctionTest(optimizer, 0.003, 0.006, 3);
}
