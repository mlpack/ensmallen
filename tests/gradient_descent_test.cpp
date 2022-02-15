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

#include <ensmallen.hpp>
#include "catch.hpp"
#include "test_function_tools.hpp"

using namespace ens;
using namespace ens::test;

TEST_CASE("SimpleGDTestFunction", "[GradientDescentTest]")
{
  GradientDescent s(0.01, 5000000, 1e-9);
  FunctionTest<GDTestFunction>(s, 0.1, 0.01);
}

TEST_CASE("GDRosenbrockTest", "[GradientDescentTest]")
{
  GradientDescent s(0.001, 0, 1e-15);
  FunctionTest<RosenbrockFunction>(s, 0.01, 0.001);
}

TEST_CASE("GDRosenbrockFMatTest", "[GradientDescentTest]")
{
  GradientDescent s(0.001, 0, 1e-15);
  FunctionTest<RosenbrockFunction, arma::fmat>(s, 0.1, 0.01);
}
