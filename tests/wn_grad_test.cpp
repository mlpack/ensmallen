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

TEMPLATE_TEST_CASE("WNGradLogisticRegressionTest", "[WNGrad]",
    arma::mat, arma::fmat)
{
  WNGrad optimizer(0.56, 1, 500000, 1e-9, true);
  LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(
      optimizer, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("WNGradSphereFunctionTest", "[WNGrad]",
    arma::mat, arma::fmat)
{
  WNGrad optimizer(0.56, 2, 500000, 1e-9, true);
  FunctionTest<SphereFunction<TestType, arma::Row<size_t>>, TestType>(
      optimizer, 1.0, 0.1);
}

TEMPLATE_TEST_CASE("WNGradStyblinskiTangFunctionTest", "[WNGrad]",
    arma::mat, arma::fmat)
{
  WNGrad optimizer(0.56, 2, 500000, 1e-9, true);
  FunctionTest<StyblinskiTangFunction<TestType, arma::Row<size_t>>, TestType>(
      optimizer, 0.3, 0.03);
}

TEST_CASE("WNGradStyblinskiTangFunctionSpMatTest", "[WNGrad]")
{
  WNGrad optimizer(0.56, 2, 500000, 1e-9, true);
  FunctionTest<StyblinskiTangFunction<>, arma::sp_mat>(optimizer, 0.3, 0.03);
}

#ifdef ENS_HAS_COOT

TEMPLATE_TEST_CASE("WNGradLogisticRegressionTest", "[WNGrad]",
    coot::mat, coot::fmat)
{
  WNGrad optimizer(0.56, 1, 500000, 1e-9, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      optimizer, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("WNGradSphereFunctionTest", "[WNGrad]",
    coot::mat, coot::fmat)
{
  WNGrad optimizer(0.56, 2, 500000, 1e-9, true);
  FunctionTest<SphereFunction<TestType, coot::Row<size_t>>, TestType>(
      optimizer, 1.0, 0.1);
}

TEMPLATE_TEST_CASE("WNGradStyblinskiTangFunctionTest", "[WNGrad]",
    coot::mat, coot::fmat)
{
  WNGrad optimizer(0.56, 2, 500000, 1e-9, true);
  FunctionTest<StyblinskiTangFunction<TestType, coot::Row<size_t>>, TestType>(
      optimizer, 0.3, 0.03);
}

#endif
