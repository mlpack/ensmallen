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

using namespace ens;
using namespace ens::test;

TEST_CASE("SimpleGDTestFunction", "[GradientDescentTest]")
{
  GDTestFunction f;
  GradientDescent s(0.01, 5000000, 1e-9);

  arma::mat coordinates = f.GetInitialPoint<arma::mat>();
  double result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(0.0).margin(1e-4));
  REQUIRE(coordinates(0) == Approx(0.0).margin(1e-2));
  REQUIRE(coordinates(1) == Approx(0.0).margin(1e-2));
  REQUIRE(coordinates(2) == Approx(0.0).margin(1e-2));
}

TEST_CASE("GDRosenbrockTest", "[GradientDescentTest]")
{
  // Create the Rosenbrock function.
  RosenbrockFunction f;

  GradientDescent s(0.001, 0, 1e-15);

  arma::mat coordinates = f.GetInitialPoint<arma::mat>();
  double result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(0.0).margin(1e-10));
  for (size_t j = 0; j < 2; ++j)
    REQUIRE(coordinates(j) == Approx(1.0).epsilon(1e-5));
}

TEST_CASE("GDRosenbrockFMatTest", "[GradientDescentTest]")
{
  // Create the Rosenbrock function.
  RosenbrockFunction f;

  GradientDescent s(0.001, 0, 1e-15);

  arma::fmat coordinates = f.GetInitialPoint<arma::fmat>();
  float result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(0.0).margin(1e-5));
  for (size_t j = 0; j < 2; ++j)
    REQUIRE(coordinates(j) == Approx(1.0).epsilon(1e-3));
}
