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
#if defined(ENS_USE_COOT)
  #include <armadillo>
  #include <bandicoot>
#endif
#include <ensmallen.hpp>
#include "catch.hpp"
#include "test_function_tools.hpp"

using namespace ens;
using namespace ens::test;

TEMPLATE_TEST_CASE("WNGrad_LogisticRegressionFunction", "[WNGrad]",
    arma::mat, arma::fmat)
{
  WNGrad optimizer(0.56, 1, 500000, 1e-9, true);
  LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(
      optimizer, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("WNGrad_SphereFunction", "[WNGrad]",
    arma::mat, arma::fmat)
{
  WNGrad optimizer(1.12, 2, 500000, 1e-9, true);
  FunctionTest<SphereFunction, TestType>(optimizer, 1.0, 0.1);
}

TEMPLATE_TEST_CASE("WNGrad_StyblinskiTangFunction", "[WNGrad]",
    arma::mat, arma::fmat, arma::sp_mat)
{
  WNGrad optimizer(1.12, 2, 500000, 1e-9, true);
  FunctionTest<StyblinskiTangFunction, TestType>(optimizer, 0.3, 0.03);
}

#ifdef ENS_HAVE_COOT

TEMPLATE_TEST_CASE("WNGrad_LogisticRegressionFunction", "[WNGrad]",
    coot::mat, coot::fmat)
{
  WNGrad optimizer(0.56, 1, 500000, 1e-9, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      optimizer, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("WNGrad_SphereFunction", "[WNGrad]",
    coot::mat, coot::fmat)
{
  WNGrad optimizer(1.12, 2, 500000, 1e-9, true);
  FunctionTest<SphereFunction, TestType>(optimizer, 1.0, 0.1);
}

TEMPLATE_TEST_CASE("WNGrad_StyblinskiTangFunction", "[WNGrad]",
    coot::mat, coot::fmat)
{
  WNGrad optimizer(1.12, 2, 500000, 1e-9, true);
  FunctionTest<StyblinskiTangFunction, TestType>(optimizer, 0.3, 0.03);
}

#endif
