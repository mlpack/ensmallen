/**
 * @file ada_bound_test.cpp
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

TEMPLATE_TEST_CASE("AdaBoundSphereFunctionTest", "[AdaBound]",
    arma::mat, arma::fmat)
{
  AdaBound optimizer(0.001, 2, 0.1, 1e-3, 0.9, 0.999, 1e-8, 500000,
      1e-3, false);
  FunctionTest<SphereFunction<TestType, arma::Row<size_t>>, TestType>(
      optimizer, 0.5, 0.1);
}

TEMPLATE_TEST_CASE("AMSBoundSphereFunctionTest", "[AdaBound]",
    arma::mat, arma::fmat)
{
  AMSBound optimizer(0.001, 2, 0.1, 1e-3, 0.9, 0.999, 1e-8, 500000,
      1e-3, false);
  FunctionTest<SphereFunction<TestType, arma::Row<size_t>>, TestType>(
      optimizer, 0.5, 0.1);
}

/**
 * Test the AdaBound optimizer on the Sphere function with arma::sp_mat.
 */
TEST_CASE("AdaBoundSphereFunctionTestSpMat", "[AdaBoundTest]")
{
  AdaBound optimizer(0.001, 2, 0.1, 1e-3, 0.9, 0.999, 1e-8, 500000,
      1e-3, false);
  FunctionTest<SphereFunction<>, arma::sp_mat>(optimizer, 0.5, 0.1);
}

TEST_CASE("AdaBoundSphereFunctionTestSpMatDenseGradient", "[AdaBound]")
{
  SphereFunction<> f(2);
  AdaBound optimizer(0.001, 2, 0.1, 1e-3, 0.9, 0.999, 1e-8, 500000,
      1e-3, false);

  arma::sp_mat coordinates = arma::conv_to<arma::sp_mat>::from(
      f.GetInitialPoint());
  optimizer.Optimize<decltype(f), arma::sp_mat, arma::mat>(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
}

TEST_CASE("AMSBoundSphereFunctionTestSpMat", "[AdaBound]")
{
  AMSBound optimizer(0.001, 2, 0.1, 1e-3, 0.9, 0.999, 1e-8, 500000,
      1e-3, false);
  FunctionTest<SphereFunction<>, arma::sp_mat>(optimizer, 0.5, 0.1);
}

TEST_CASE("AMSBoundSphereFunctionTestSpMatDenseGradient", "[AdaBound]")
{
  SphereFunction<> f(2);
  AMSBound optimizer(0.001, 2, 0.1, 1e-3, 0.9, 0.999, 1e-8, 500000,
      1e-3, false);

  arma::sp_mat coordinates = f.GetInitialPoint<arma::sp_mat>();
  optimizer.Optimize<decltype(f), arma::sp_mat, arma::mat>(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
}

#endif

#ifdef USE_COOT

TEMPLATE_TEST_CASE("AdaBoundSphereFunctionTest", "[AdaBound]",
    coot::mat, coot::fmat)
{
  AdaBound optimizer(0.001, 2, 0.1, 1e-3, 0.9, 0.999, 1e-8, 500000,
      1e-3, false);
  FunctionTest<SphereFunction<TestType, coot::Row<size_t>>, TestType>(
      optimizer, 0.5, 0.1);
}

TEMPLATE_TEST_CASE("AMSBoundSphereFunctionTest", "[AdaBound]",
    coot::mat, coot::fmat)
{
  AMSBound optimizer(0.001, 2, 0.1, 1e-3, 0.9, 0.999, 1e-8, 500000,
      1e-3, false);
  FunctionTest<SphereFunction<TestType, coot::Row<size_t>>, TestType>(
      optimizer, 0.5, 0.1);
}
#endif
