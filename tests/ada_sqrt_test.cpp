/**
 * @file ada_sqrt_test.cpp
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
 * Run AdaSqrt on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("AdaSqrtLogisticRegressionTest", "[AdaSqrtTest]")
{
  AdaSqrt optimizer(0.01, 32, 1e-8, 5000000, 1e-9, true);
  LogisticRegressionFunctionTest(optimizer, 0.003, 0.006);
}

/**
 * Run AdaSqrt on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("AdaSqrtLogisticRegressionTestFMat", "[AdaSqrtTest]")
{
  AdaSqrt optimizer(0.01, 32, 1e-8, 5000000, 1e-9, true);
  LogisticRegressionFunctionTest<arma::fmat>(optimizer, 0.003, 0.006);
}
