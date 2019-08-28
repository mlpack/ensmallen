/**
 * @file aug_lagrangian_test.cpp
 * @author Ryan Curtin
 * @author Marcus Edel
 * @author Conrad Sanderson
 *
 * Test of the AugmentedLagrangian class using the test functions defined in
 * aug_lagrangian_test_functions.hpp.
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
 * Tests the Augmented Lagrangian optimizer using the
 * AugmentedLagrangianTestFunction class.
 */
TEST_CASE("AugLagrangianTestFunctionTest", "[AugLagrangianTest]")
{
  // The choice of 10 memory slots is arbitrary.
  AugLagrangianTestFunction f;
  AugLagrangian aug;

  arma::vec coords = f.GetInitialPoint();

  if (!aug.Optimize(f, coords))
    FAIL("Optimization reported failure.");

  double finalValue = f.Evaluate(coords);

  REQUIRE(finalValue == Approx(70.0).epsilon(1e-7));
  REQUIRE(coords(0) == Approx(1.0).epsilon(1e-7));
  REQUIRE(coords(1) == Approx(4.0).epsilon(1e-7));
}

/**
 * Tests the Augmented Lagrangian optimizer using the Gockenbach function.
 */
TEST_CASE("GockenbachFunctionTest", "[AugLagrangianTest]")
{
  GockenbachFunction f;
  AugLagrangian aug;

  arma::mat coords = f.GetInitialPoint<arma::mat>();

  if (!aug.Optimize(f, coords))
    FAIL("Optimization reported failure.");

  double finalValue = f.Evaluate(coords);

  // Higher tolerance for smaller values.
  REQUIRE(finalValue == Approx(29.633926).epsilon(1e-7));
  REQUIRE(coords(0) == Approx(0.12288178).epsilon(1e-5));
  REQUIRE(coords(1) == Approx(-1.10778185).epsilon(1e-7));
  REQUIRE(coords(2) == Approx(0.015099932).epsilon(1e-5));
}

/**
 * Tests the Augmented Lagrangian optimizer using the Gockenbach function.  Uses
 * arma::fmat.
 */
TEST_CASE("GockenbachFunctionFMatTest", "[AugLagrangianTest]")
{
  GockenbachFunction f;
  AugLagrangian aug;

  arma::fmat coords = f.GetInitialPoint<arma::fmat>();

  if (!aug.Optimize(f, coords))
    FAIL("Optimization reported failure.");

  float finalValue = f.Evaluate(coords);

  // Higher tolerance for smaller values.
  REQUIRE(finalValue == Approx(29.633926).epsilon(1e-3));
  REQUIRE(coords(0) == Approx(0.12288178).epsilon(0.1));
  REQUIRE(coords(1) == Approx(-1.10778185).epsilon(1e-3));
  REQUIRE(coords(2) == Approx(0.015099932).epsilon(0.1));
}

/**
 * Tests the Augmented Lagrangian optimizer using the Gockenbach function.  Uses
 * arma::sp_mat.
 */
TEST_CASE("GockenbachFunctionSpMatTest", "[AugLagrangianTest]")
{
  GockenbachFunction f;
  AugLagrangian aug;

  arma::sp_mat coords = f.GetInitialPoint<arma::sp_mat>();

  if (!aug.Optimize(f, coords))
    FAIL("Optimization reported failure.");

  double finalValue = f.Evaluate(coords);

  // Higher tolerance for smaller values.
  REQUIRE(finalValue == Approx(29.633926).epsilon(1e-7));
  REQUIRE(coords(0) == Approx(0.12288178).epsilon(1e-5));
  REQUIRE(coords(1) == Approx(-1.10778185).epsilon(1e-7));
  REQUIRE(coords(2) == Approx(0.015099932).epsilon(1e-5));
}
