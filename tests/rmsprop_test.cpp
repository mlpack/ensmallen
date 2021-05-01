/**
 * @file rmsprop_test.cpp
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
 * Run RMSProp on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("RMSPropLogisticRegressionTest", "[rmsprop]")
{
  RMSProp optimizer;
  LogisticRegressionFunctionTest(optimizer, 0.003, 0.006);
}

/**
 * Run RMSProp on logistic regression and make sure the results are acceptable.
 * Use arma::fmat.
 */
TEST_CASE("RMSPropLogisticRegressionFMatTest", "[rmsprop]")
{
  RMSProp optimizer;
  LogisticRegressionFunctionTest<arma::fmat>(optimizer, 0.003, 0.006);
}

#if ARMA_VERSION_MAJOR > 9 ||\
    (ARMA_VERSION_MAJOR == 9 && ARMA_VERSION_MINOR >= 400)

/**
 * Run RMSProp on logistic regression and make sure the results are acceptable.
 * Use arma::sp_mat.
 */
TEST_CASE("RMSPropLogisticRegressionSpMatTest", "[rmsprop]")
{
  RMSProp optimizer;
  LogisticRegressionFunctionTest<arma::sp_mat>(optimizer, 0.003, 0.006);
}

#endif
