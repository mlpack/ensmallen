/**
 * @file ada_grad_test.cpp
 * @author Abhinav Moudgil
 * @author Marcus Edel
 * @author Conrad Sanderson
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
 * Run AdaGrad on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("AdaGradLogisticRegressionTest", "[AdaGradTest]")
{
  AdaGrad adagrad(0.99, 32, 1e-8, 5000000, 1e-9, true);
  LogisticRegressionFunctionTest(adagrad, 0.003, 0.006);
}

/**
 * Run AdaGrad on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("AdaGradLogisticRegressionTestFMat", "[AdaGradTest]")
{
  AdaGrad adagrad(0.99, 32, 1e-8, 5000000, 1e-9, true);
  LogisticRegressionFunctionTest<arma::fmat>(adagrad, 0.003, 0.006);
}
