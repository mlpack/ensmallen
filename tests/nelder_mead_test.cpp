/**
 * @file nelder_mead_test.cpp
 * @author Marcus Edel
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

/**
 * Tests the Nelder-Mead optimizer using the Rosenbrock Function.
 */
TEST_CASE("NMRosenbrockFunctionTest", "[NelderMeadTest]")
{
  RosenbrockFunction f;
  NelderMead nm(300, 1e-15);

  arma::mat coords = f.GetInitialPoint();
  nm.Optimize(f, coords);

  double finalValue = f.Evaluate(coords);

  REQUIRE(finalValue == Approx(0.0).margin(1e-5));
  REQUIRE(coords(0) == Approx(1.0).epsilon(1e-7));
  REQUIRE(coords(1) == Approx(1.0).epsilon(1e-7));
}

/**
 * Test the Nelder-Mead optimizer using an arma::fmat with the Rosenbrock
 * function.
 */
TEST_CASE("NMRosenbrockFunctionFloatTest", "[NelderMeadTest]")
{
  RosenbrockFunction f;
  NelderMead nm(300, 1e-15);

  arma::fmat coords = f.GetInitialPoint<arma::fvec>();
  nm.Optimize(f, coords);

  float finalValue = f.Evaluate(coords);

  REQUIRE(finalValue == Approx(0.0f).margin(1e-3));
  REQUIRE(coords(0) == Approx(1.0f).epsilon(1e-4));
  REQUIRE(coords(1) == Approx(1.0f).epsilon(1e-4));
}

/**
 * Tests the Nelder–Mead optimizer using the Colville Function.
 */
TEST_CASE("NMColvilleFunctionTest", "[NelderMeadTest]")
{
  ColvilleFunction f;
  NelderMead nm(500, 1e-15);

  arma::mat coords = f.GetInitialPoint();
  nm.Optimize(f, coords);

  REQUIRE(coords(0) == Approx(1.0).epsilon(1e-7));
  REQUIRE(coords(1) == Approx(1.0).epsilon(1e-7));
}

/**
 * Tests the Nelder–Mead optimizer using the Wood Function.
 */
TEST_CASE("NMWoodFunctionTest", "[NelderMeadTest]")
{
  WoodFunction f;
  NelderMead nm(800, 1e-15);

  arma::mat coords = f.GetInitialPoint();
  nm.Optimize(f, coords);

  double finalValue = f.Evaluate(coords);

  REQUIRE(finalValue == Approx(0.0).margin(1e-5));
  REQUIRE(coords(0) == Approx(1.0).epsilon(1e-7));
  REQUIRE(coords(1) == Approx(1.0).epsilon(1e-7));
  REQUIRE(coords(2) == Approx(1.0).epsilon(1e-7));
  REQUIRE(coords(3) == Approx(1.0).epsilon(1e-7));
}
