/**
 * @file ftml_test.cpp
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Test file for the FTML optimizer.
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

TEMPLATE_TEST_CASE("FTML_LogisticRegressionFunction", "[FTML]",
    arma::mat, arma::fmat)
{
  FTML optimizer(0.001, 1, 0.9, 0.999, 1e-8, 100000, 1e-5, true);
  LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(
      optimizer, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("FTML_SphereFunction", "[FTML]",
    arma::mat, arma::fmat)
{
  FTML optimizer(0.002, 2, 0.9, 0.999, 1e-8, 500000, 1e-9, true);
  FunctionTest<SphereFunction, TestType>(optimizer, 0.5, 0.1);
}

TEMPLATE_TEST_CASE("FTML_StyblinskiTangFunction", "[FTML]",
    arma::mat, arma::fmat)
{
  FTML optimizer(0.002, 2, 0.9, 0.999, 1e-8, 100000, 1e-5, true);
  FunctionTest<StyblinskiTangFunction, TestType>(optimizer, 0.5, 0.1);
}

#ifdef ENS_HAVE_COOT

TEMPLATE_TEST_CASE("FTML_LogisticRegressionFunction", "[FTML]",
    coot::mat, coot::fmat)
{
  FTML optimizer(0.001, 1, 0.9, 0.999, 1e-8, 100000, 1e-5, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      optimizer, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("FTML_SphereFunction", "[FTML]",
    coot::mat, coot::fmat)
{
  FTML optimizer(0.002, 2, 0.9, 0.999, 1e-8, 500000, 1e-9, true);
  FunctionTest<SphereFunction, TestType>(optimizer, 0.5, 0.1);
}

TEMPLATE_TEST_CASE("FTML_StyblinskiTangFunction", "[FTML]",
    coot::mat, coot::fmat)
{
  FTML optimizer(0.002, 2, 0.9, 0.999, 1e-8, 100000, 1e-5, true);
  FunctionTest<StyblinskiTangFunction, TestType>(optimizer, 0.5, 0.1);
}

#endif

// A test with sp_mat is not done, because FTML uses some parts internally that
// assume the objective is dense.
