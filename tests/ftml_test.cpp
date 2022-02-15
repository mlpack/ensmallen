/**
 * @file ftml_test.cpp
 * @author Ryan Curtin
 *
 * Test file for the FTML optimizer.
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
 * Run FTML on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("FTMLLogisticRegressionTest", "[FTMLTest]")
{
  FTML optimizer(0.001, 1, 0.9, 0.999, 1e-8, 100000, 1e-5, true);
  LogisticRegressionFunctionTest(optimizer, 0.003, 0.006);
}

/**
 * Test the FTML optimizer on the Sphere function.
 */
TEST_CASE("FTMLSphereFunctionTest", "[FTMLTest]")
{
  FTML optimizer(0.001, 2, 0.9, 0.999, 1e-8, 500000, 1e-9, true);
  FunctionTest<SphereFunction>(optimizer, 0.5, 0.1);
}

/**
 * Test the FTML optimizer on the Styblinski-Tang function.
 */
TEST_CASE("FTMLStyblinskiTangFunctionTest", "[FTMLTest]")
{
  FTML optimizer(0.001, 2, 0.9, 0.999, 1e-8, 100000, 1e-5, true);
  FunctionTest<StyblinskiTangFunction>(optimizer, 0.5, 0.1);
}

/**
 * Test the FTML optimizer on the Styblinski-Tang function using arma::fmat as
 * the objective type.
 */
TEST_CASE("FTMLStyblinskiTangFunctionFMatTest", "[FTMLTest]")
{
  FTML optimizer(0.001, 2, 0.9, 0.999, 1e-8, 100000, 1e-5, true);
  FunctionTest<StyblinskiTangFunction, arma::fmat>(optimizer, 0.5, 0.1);
}

// A test with sp_mat is not done, because FTML uses some parts internally that
// assume the objective is dense.
