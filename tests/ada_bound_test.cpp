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

TEMPLATE_TEST_CASE("AdaBound_SphereFunction", "[AdaBound]",
    arma::mat, arma::fmat, arma::sp_mat)
{
  AdaBound optimizer(0.001, 2, 0.1, 1e-3, 0.9, 0.999, 1e-8, 500000,
      1e-3, false);
  FunctionTest<SphereFunction, TestType>(optimizer, 0.5, 0.1);
}

TEMPLATE_TEST_CASE("AMSBound_SphereFunction", "[AdaBound]",
    arma::mat, arma::fmat, arma::sp_mat)
{
  AMSBound optimizer(0.001, 2, 0.1, 1e-3, 0.9, 0.999, 1e-8, 500000,
      1e-3, false);
  FunctionTest<SphereFunction, TestType>(optimizer, 0.5, 0.1);
}

TEMPLATE_TEST_CASE("AdaBound_SphereFunctionSpMatDenseGradient", "[AdaBound]",
    arma::sp_mat)
{
  typedef typename TestType::elem_type ElemType;

  SphereFunction f(2);
  AdaBound optimizer(0.001, 2, 0.1, 1e-3, 0.9, 0.999, 1e-8, 500000,
      1e-3, false);

  TestType coordinates = arma::conv_to<TestType>::from(
      f.GetInitialPoint());
  optimizer.Optimize<decltype(f), TestType, arma::Mat<ElemType> >(
      f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
}

TEMPLATE_TEST_CASE("AMSBound_SphereFunctionSpMatDenseGradient", "[AdaBound]",
    arma::sp_mat)
{
  typedef typename TestType::elem_type ElemType;

  SphereFunction f(2);
  AMSBound optimizer(0.001, 2, 0.1, 1e-3, 0.9, 0.999, 1e-8, 500000,
      1e-3, false);

  arma::sp_mat coordinates = f.GetInitialPoint<TestType>();
  optimizer.Optimize<decltype(f), TestType, arma::Mat<ElemType> >(
      f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
}

#ifdef ENS_USE_COOT

TEMPLATE_TEST_CASE("AdaBound_SphereFunction", "[AdaBound]",
    coot::mat, coot::fmat)
{
  AdaBound optimizer(0.001, 2, 0.1, 1e-3, 0.9, 0.999, 1e-8, 500000,
      1e-3, false);
  FunctionTest<SphereFunction, TestType>(optimizer, 0.5, 0.1);
}

TEMPLATE_TEST_CASE("AMSBoundSphereFunctionTest", "[AdaBound]",
    coot::mat, coot::fmat)
{
  AMSBound optimizer(0.001, 2, 0.1, 1e-3, 0.9, 0.999, 1e-8, 500000,
      1e-3, false);
  FunctionTest<SphereFunction, TestType>(optimizer, 0.5, 0.1);
}

#endif
