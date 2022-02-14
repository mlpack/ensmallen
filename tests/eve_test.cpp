/**
 * @file eve_test.cpp
 * @author Marcus Edel
 *
 * Test file for the Eve optimizer.
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
 * Run Eve on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("EveLogisticRegressionTest","[EveTest]")
{
  Eve optimizer(1e-3, 1, 0.9, 0.999, 0.999, 1e-8, 10000, 500000, 1e-9, true);
  LogisticRegressionFunctionTest(optimizer, 0.003, 0.006);
}

/**
 * Test the Eve optimizer on the Sphere function.
 */
TEST_CASE("EveSphereFunctionTest","[EveTest]")
{
  Eve optimizer(1e-3, 2, 0.9, 0.999, 0.999, 1e-8, 10000, 500000, 1e-9, true);
  FunctionTest<SphereFunction>(optimizer, 0.5, 0.1);
}

/**
 * Test the Eve optimizer on the Styblinski-Tang function.
 */
TEST_CASE("EveStyblinskiTangFunctionTest","[EveTest]")
{
  Eve optimizer(1e-3, 2, 0.9, 0.999, 0.999, 1e-8, 10000, 500000, 1e-9, true);
  FunctionTest<StyblinskiTangFunction>(optimizer, 0.5, 0.1);
}

/**
 * Test the Eve optimizer on the Styblinski-Tang function using arma::fmat as
 * the objective type.
 */
TEST_CASE("EveStyblinskiTangFunctionFMatTest","[EveTest]")
{
  Eve optimizer(1e-3, 2, 0.9, 0.999, 0.999, 1e-8, 10000, 500000, 1e-9, true);
  FunctionTest<StyblinskiTangFunction, arma::fmat>(optimizer, 0.5, 0.1);
}

#if ARMA_VERSION_MAJOR > 9 ||\
    (ARMA_VERSION_MAJOR == 9 && ARMA_VERSION_MINOR >= 400)

/**
 * Test the Eve optimizer on the Styblinski-Tang function, using arma::sp_mat as
 * the objective type.
 */
TEST_CASE("EveStyblinskiTangFunctionSpMatTest","[EveTest]")
{
  Eve optimizer(1e-3, 2, 0.9, 0.999, 0.999, 1e-8, 10000, 500000, 1e-9, true);
  FunctionTest<StyblinskiTangFunction, arma::sp_mat>(optimizer, 0.5, 0.1);
}

#endif
