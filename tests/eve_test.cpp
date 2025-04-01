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

TEMPLATE_TEST_CASE("EveLogisticRegressionTest", "[Eve]",
    arma::mat, arma::fmat)
{
  Eve optimizer(1e-3, 1, 0.9, 0.999, 0.999, 1e-8, 10000, 500000, 1e-9, true);
  LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(
      optimizer, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("EveSphereFunctionTest", "[Eve]",
    arma::mat, arma::fmat)
{
  Eve optimizer(1e-3, 2, 0.9, 0.999, 0.999, 1e-8, 10000, 500000, 1e-9, true);
  FunctionTest<SphereFunction<TestType, arma::Row<size_t>>, TestType>(
      optimizer, 0.5, 0.1);
}

TEMPLATE_TEST_CASE("EveStyblinskiTangFunctionTest", "[Eve]",
    arma::mat, arma::fmat)
{
  Eve optimizer(1e-3, 2, 0.9, 0.999, 0.999, 1e-8, 10000, 500000, 1e-9, true);
  FunctionTest<StyblinskiTangFunction<TestType, arma::Row<size_t>>, TestType>(
      optimizer, 0.5, 0.1);
}

/**
 * Test the Eve optimizer on the Styblinski-Tang function, using arma::sp_mat as
 * the objective type.
 */
TEST_CASE("EveStyblinskiTangFunctionSpMatTest","[Eve]")
{
  Eve optimizer(1e-3, 2, 0.9, 0.999, 0.999, 1e-8, 10000, 500000, 1e-9, true);
  FunctionTest<StyblinskiTangFunction<>, arma::sp_mat>(optimizer, 0.5, 0.1);
}

#ifdef ENS_HAS_COOT

TEMPLATE_TEST_CASE("EveLogisticRegressionTest", "[Eve]",
    coot::mat, coot::fmat)
{
  Eve optimizer(1e-3, 1, 0.9, 0.999, 0.999, 1e-8, 10000, 500000, 1e-9, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      optimizer, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("EveSphereFunctionTest", "[Eve]",
    coot::mat, coot::fmat)
{
  Eve optimizer(1e-3, 2, 0.9, 0.999, 0.999, 1e-8, 10000, 500000, 1e-9, true);
  FunctionTest<SphereFunction<TestType, coot::Row<size_t>>, TestType>(
      optimizer, 0.5, 0.1);
}

TEMPLATE_TEST_CASE("EveStyblinskiTangFunctionTest", "[Eve]",
    coot::mat, coot::fmat)
{
  Eve optimizer(1e-3, 2, 0.9, 0.999, 0.999, 1e-8, 10000, 500000, 1e-9, true);
  FunctionTest<StyblinskiTangFunction<TestType, coot::Row<size_t>>, TestType>(
      optimizer, 0.5, 0.1);
}

#endif
