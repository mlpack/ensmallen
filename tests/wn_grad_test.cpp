/**
 * @file wn_grad_test.cpp
 * @author Marcus Edel
 *
 * Test file for the WNGrad optimizer.
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
 * Run WNGrad on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("WNGradLogisticRegressionTest","[WNGradTest]")
{
  WNGrad optimizer(0.56, 1, 500000, 1e-9, true);
  LogisticRegressionFunctionTest(optimizer, 0.003, 0.006);
}

/**
 * Test the WNGrad optimizer on the Sphere function.
 */
TEST_CASE("WNGradSphereFunctionTest","[WNGradTest]")
{
  WNGrad optimizer(0.56, 2, 500000, 1e-9, true);
  FunctionTest<SphereFunction>(optimizer, 1.0, 0.1);
}

/**
 * Test the WNGrad optimizer on the StyblinskiTangFunction.
 */
TEST_CASE("WNGradStyblinskiTangFunctionTest","[WNGradTest]")
{
  WNGrad optimizer(0.56, 2, 500000, 1e-9, true);
  FunctionTest<StyblinskiTangFunction>(optimizer, 0.3, 0.03);
}

/**
 * Test the WNGrad optimizer on the StyblinskiTangFunction.  Use arma::fmat.
 */
TEST_CASE("WNGradStyblinskiTangFunctionFMatTest", "[WNGradTest]")
{
  WNGrad optimizer(0.56, 2, 500000, 1e-9, true);
  FunctionTest<StyblinskiTangFunction, arma::fmat>(optimizer, 3.0, 0.3);
}

/**
 * Test the WNGrad optimizer on the StyblinskiTangFunction.  Use arma::sp_mat.
 */
TEST_CASE("WNGradStyblinskiTangFunctionSpMatTest", "[WNGradTest]")
{
  WNGrad optimizer(0.56, 2, 500000, 1e-9, true);
  FunctionTest<StyblinskiTangFunction, arma::sp_mat>(optimizer, 0.3, 0.03);
}
