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

TEMPLATE_TEST_CASE("RMSPropLogisticRegressionTest", "[RMSProp]",
    arma::mat, arma::fmat, arma::sp_mat)
{
  RMSProp optimizer;
  LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(optimizer, 0.003, 0.006);
}

#ifdef USE_COOT

TEMPLATE_TEST_CASE("RMSPropLogisticRegressionTest", "[RMSProp]",
    coot::mat, coot::fmat)
{
  RMSProp optimizer;
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(optimizer, 0.003, 0.006);
}

#endif

