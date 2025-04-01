/**
 * @file swats_test.cpp
 * @author Marcus Edel
 *
 * Test file for the SWATS optimizer.
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

TEMPLATE_TEST_CASE("SWATSLogisticRegressionTestFunction", "[SWATS]",
    arma::mat, arma::fmat)
{
  SWATS optimizer(1e-3, 10, 0.9, 0.999, 1e-6, 600000, 1e-9, true);
  // We allow a few trials in case of poor convergence.
  LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(
      optimizer, 0.003, 0.006, 5);
}

TEMPLATE_TEST_CASE("SWATSSphereFunctionTest", "[SWATS]",
    arma::mat, arma::fmat)
{
  SWATS optimizer(1e-3, 2, 0.9, 0.999, 1e-6, 500000, 1e-9, true);
  FunctionTest<SphereFunction<TestType, arma::Row<size_t>>, TestType>(
      optimizer, 1.0, 0.1);
}

TEMPLATE_TEST_CASE("SWATSStyblinskiTangFunctionFMatTest", "[SWATS]",
    arma::mat, arma::fmat)
{
  SWATS optimizer(1e-3, 2, 0.9, 0.999, 1e-6, 500000, 1e-9, true);
  FunctionTest<StyblinskiTangFunction<TestType, arma::Row<size_t>>,
    TestType>(optimizer, 3.0, 0.3);
}

/**
 * Test the SWATS optimizer on the Styblinski-Tang function.  Use arma::sp_mat.
 */
TEST_CASE("SWATSStyblinskiTangFunctionSpMatTest", "[SWATSTest]")
{
  SWATS optimizer(1e-3, 2, 0.9, 0.999, 1e-6, 500000, 1e-9, true);
  FunctionTest<StyblinskiTangFunction<>, arma::sp_mat>(optimizer, 0.3, 0.03);
}

#ifdef ENS_HAS_COOT

TEMPLATE_TEST_CASE("SWATSLogisticRegressionTestFunction", "[SWATS]",
    coot::mat, coot::fmat)
{
  SWATS optimizer(1e-3, 10, 0.9, 0.999, 1e-6, 600000, 1e-9, true);
  // We allow a few trials in case of poor convergence.
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      optimizer, 0.003, 0.006, 5);
}

/* TEMPLATE_TEST_CASE("SWATSSphereFunctionTest", "[SWATS]", */
/*     coot::mat, coot::fmat) */
/* { */
/*   SWATS optimizer(1e-3, 2, 0.9, 0.999, 1e-6, 500000, 1e-9, true); */
/*   FunctionTest<SphereFunction<TestType, arma::Row<size_t>, TestType>( */
/*       optimizer, 1.0, 0.1); */
/* } */

/* TEMPLATE_TEST_CASE("SWATSStyblinskiTangFunctionFMatTest", "[SWATS]", */
/*     coot::mat, coot::fmat) */
/* { */
/*   SWATS optimizer(1e-3, 2, 0.9, 0.999, 1e-6, 500000, 1e-9, true); */
/*   FunctionTest<StyblinskiTangFunction<TestType, arma::Row<size_t>, TestType>(optimizer, 3.0, 0.3); */
/* } */

#endif
