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

/**
 * Test the AdaBound optimizer on the Sphere function.
 */
TEST_CASE("AdaBoundSphereFunctionTest", "[AdaBoundTest]")
{
  AdaBound optimizer(0.001, 2, 0.1, 1e-3, 0.9, 0.999, 1e-8, 500000,
      1e-3, false);
  FunctionTest<SphereFunction>(optimizer, 0.5, 0.1);
}

/**
 * Test the AdaBound optimizer on the Sphere function with arma::fmat.
 */
TEST_CASE("AdaBoundSphereFunctionTestFMat", "[AdaBoundTest]")
{
  AdaBound optimizer(0.001, 2, 0.1, 1e-3, 0.9, 0.999, 1e-8, 500000,
      1e-3, false);
  FunctionTest<SphereFunction, arma::fmat>(optimizer, 0.5, 0.1);
}

/**
 * Test the AMSBound optimizer on the Sphere function.
 */
TEST_CASE("AMSBoundSphereFunctionTest", "[AdaBoundTest]")
{
  AMSBound optimizer(0.001, 2, 0.1, 1e-3, 0.9, 0.999, 1e-8, 500000,
      1e-3, false);
  FunctionTest<SphereFunction, arma::mat>(optimizer, 0.5, 0.1);
}

/**
 * Test the AMSBound optimizer on the Sphere function with arma::fmat.
 */
TEST_CASE("AMSBoundphereFunctionTestFMat", "[AdaBoundTest]")
{
  AMSBound optimizer(0.001, 2, 0.1, 1e-3, 0.9, 0.999, 1e-8, 500000,
      1e-3, false);
  FunctionTest<SphereFunction, arma::fmat>(optimizer, 0.5, 0.1);
}

#if ARMA_VERSION_MAJOR > 9 ||\
    (ARMA_VERSION_MAJOR == 9 && ARMA_VERSION_MINOR >= 400)

/**
 * Test the AdaBound optimizer on the Sphere function with arma::sp_mat.
 */
TEST_CASE("AdaBoundSphereFunctionTestSpMat", "[AdaBoundTest]")
{
  AdaBound optimizer(0.001, 2, 0.1, 1e-3, 0.9, 0.999, 1e-8, 500000,
      1e-3, false);
  FunctionTest<SphereFunction, arma::sp_mat>(optimizer, 0.5, 0.1);
}

/**
 * Test the AdaBound optimizer on the Sphere function with arma::sp_mat but a
 * dense (arma::mat) gradient.
 */
TEST_CASE("AdaBoundSphereFunctionTestSpMatDenseGradient", "[AdaBoundTest]")
{
  SphereFunction f(2);
  AdaBound optimizer(0.001, 2, 0.1, 1e-3, 0.9, 0.999, 1e-8, 500000,
      1e-3, false);

  arma::sp_mat coordinates = f.GetInitialPoint<arma::sp_mat>();
  optimizer.Optimize<decltype(f), arma::sp_mat, arma::mat>(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
}

/**
 * Test the AMSBound optimizer on the Sphere function with arma::sp_mat.
 */
TEST_CASE("AMSBoundSphereFunctionTestSpMat", "[AdaBoundTest]")
{
  AMSBound optimizer(0.001, 2, 0.1, 1e-3, 0.9, 0.999, 1e-8, 500000,
      1e-3, false);
  FunctionTest<SphereFunction, arma::sp_mat>(optimizer, 0.5, 0.1);
}

/**
 * Test the AMSBound optimizer on the Sphere function with arma::sp_mat but a
 * dense (arma::mat) gradient.
 */
TEST_CASE("AMSBoundSphereFunctionTestSpMatDenseGradient", "[AdaBoundTest]")
{
  SphereFunction f(2);
  AMSBound optimizer(0.001, 2, 0.1, 1e-3, 0.9, 0.999, 1e-8, 500000,
      1e-3, false);

  arma::sp_mat coordinates = f.GetInitialPoint<arma::sp_mat>();
  optimizer.Optimize<decltype(f), arma::sp_mat, arma::mat>(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
}

#endif
