/**
 * @file spsa_test.cpp
 * @author N Rajiv Vaidyanathan
 * @author Marcus Edel
 *
 * Test file for the SPSA optimizer.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <ensmallen.hpp>
#include "catch.hpp"
#include "test_function_tools.hpp"

using namespace arma;
using namespace ens;
using namespace ens::test;

/**
 * Test the SPSA optimizer on the Sphere function.
 */
TEST_CASE("SPSASphereFunctionTest", "[SPSATest]")
{
  SPSA optimizer(0.1, 0.102, 0.16, 0.3, 100000, 0);
  FunctionTest<SphereFunction>(optimizer, 1.0, 0.1);
}

/**
 * Test the SPSA optimizer on the Sphere function using arma::fmat.
 */
TEST_CASE("SPSASphereFunctionFMatTest", "[SPSATest]")
{
  SPSA optimizer(0.1, 0.102, 0.16, 0.3, 100000, 0);
  FunctionTest<SphereFunction, arma::fmat>(optimizer, 1.0, 0.1);
}

/**
 * Test the SPSA optimizer on the Sphere function using arma::sp_mat.
 */
TEST_CASE("SPSASphereFunctionSpMatTest", "[SPSATest]")
{
  SPSA optimizer(0.1, 0.102, 0.16, 0.3, 100000, 0);
  FunctionTest<SphereFunction, arma::sp_mat>(optimizer, 1.0, 0.1);
}

/**
 * Test the SPSA optimizer on the Matyas function.
 */
TEST_CASE("SPSAMatyasFunctionTest", "[SPSATest]")
{
  SPSA optimizer(0.1, 0.102, 0.16, 0.3, 100000, 0);
  FunctionTest<MatyasFunction>(optimizer, 0.1, 0.01);
}

/**
 * Run SPSA on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("SPSALogisticRegressionTest", "[SPSATest]")
{
  // We allow 10 trials, because SPSA is definitely not guaranteed to
  // converge.
  SPSA optimizer(0.5, 0.102, 0.002, 0.3, 5000, 1e-8);
  LogisticRegressionFunctionTest(optimizer, 0.003, 0.006, 10);
}
