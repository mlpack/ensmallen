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
#if defined(ENS_USE_COOT)
  #include <armadillo>
  #include <bandicoot>
#endif
#include <ensmallen.hpp>
#include "catch.hpp"
#include "test_function_tools.hpp"

using namespace ens;
using namespace ens::test;

TEMPLATE_TEST_CASE("RMSProp_LogisticRegressionFunction", "[RMSProp]",
    arma::mat, arma::fmat, arma::sp_mat)
{
  RMSProp optimizer(0.32);
  LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(
      optimizer, 0.003, 0.006);
}

#ifdef ENS_HAVE_COOT

TEMPLATE_TEST_CASE("RMSProp_LogisticRegressionFunction", "[RMSProp]",
    coot::mat, coot::fmat)
{
  RMSProp optimizer(0.32);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      optimizer, 0.003, 0.006);
}

#endif

