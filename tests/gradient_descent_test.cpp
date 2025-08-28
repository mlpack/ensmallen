/**
 * @file gradient_descent_test.cpp
 * @author Sumedh Ghaisas
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

TEMPLATE_TEST_CASE("GradientDescent_GDTestFunction", "[GradientDescent]",
   arma::mat, arma::fmat)
{
  GradientDescent s(0.01, 5000000, 1e-9);
  FunctionTest<GDTestFunction, TestType>(s, 0.1, 0.01);
}

TEMPLATE_TEST_CASE("GradientDescent_RosenbrockFunction", "[GradientDescent]",
    arma::mat, arma::fmat)
{
  GradientDescent s(0.001, 0, 1e-15);
  FunctionTest<RosenbrockFunction, TestType>(s, 0.01, 0.001);
}

#ifdef ENS_HAVE_COOT

TEMPLATE_TEST_CASE("GradientDescent_GDTestFunction", "[GradientDescent]",
   coot::mat, coot::fmat)
{
  GradientDescent s(0.01, 5000000, 1e-9);
  FunctionTest<GDTestFunction, TestType>(s, 0.1, 0.01);
}

TEMPLATE_TEST_CASE("GradientDescent_RosenbrockFunction", "[GradientDescent]",
    coot::mat, coot::fmat)
{
  GradientDescent s(0.001, 0, 1e-15);
  FunctionTest<RosenbrockFunction, TestType>(s, 0.01, 0.001);
}

#endif
