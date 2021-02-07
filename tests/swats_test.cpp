/**
 * @file swats_test.cpp
 * @author Marcus Edel
 *
 * Test file for the SWATS optimizer.
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
 * Run SWATS on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("SWATSLogisticRegressionTestFunction", "[SWATSTest]")
{
  SWATS optimizer(1e-3, 10, 0.9, 0.999, 1e-6, 600000, 1e-9, true);
  // We allow a few trials in case of poor convergence.
  LogisticRegressionFunctionTest(optimizer, 0.003, 0.006, 5);
}

/**
 * Test the SWATS optimizer on the Sphere function.
 */
TEST_CASE("SWATSSphereFunctionTest", "[SWATSTest]")
{
  SWATS optimizer(1e-3, 2, 0.9, 0.999, 1e-6, 500000, 1e-9, true);
  FunctionTest<SphereFunction>(optimizer, 1.0, 0.1);
}

/**
 * Test the SWATS optimizer on the Styblinski-Tang function.
 */
TEST_CASE("SWATSStyblinskiTangFunctionTest", "[SWATSTest]")
{
  SWATS optimizer(1e-3, 2, 0.9, 0.999, 1e-6, 500000, 1e-9, true);
  FunctionTest<StyblinskiTangFunction>(optimizer, 0.3, 0.03);
}

/**
 * Test the SWATS optimizer on the Styblinski-Tang function.  Use arma::fmat.
 */
TEST_CASE("SWATSStyblinskiTangFunctionFMatTest", "[SWATSTest]")
{
  SWATS optimizer(1e-3, 2, 0.9, 0.999, 1e-6, 500000, 1e-9, true);
  FunctionTest<StyblinskiTangFunction, arma::fmat>(optimizer, 3.0, 0.3);
}

#if ARMA_VERSION_MAJOR > 9 ||\
    (ARMA_VERSION_MAJOR == 9 && ARMA_VERSION_MINOR >= 400)

/**
 * Test the SWATS optimizer on the Styblinski-Tang function.  Use arma::sp_mat.
 */
TEST_CASE("SWATSStyblinskiTangFunctionSpMatTest", "[SWATSTest]")
{
  SWATS optimizer(1e-3, 2, 0.9, 0.999, 1e-6, 500000, 1e-9, true);
  FunctionTest<StyblinskiTangFunction, arma::sp_mat>(optimizer, 0.3, 0.03);
}

#endif
