/**
 * @file cne_test.cpp
 * @author Marcus Edel
 * @author Kartik Nighania
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

/**
 * Optimize the Sphere function using CNE.
 */
TEST_CASE("CNESphereFunctionTest", "[CNETest]")
{
  SphereFunction f(2);
  CNE optimizer(200, 1000, 0.2, 0.2, 0.2, 1e-5);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates[0] == Approx(0.0).margin(0.1));
  REQUIRE(coordinates[1] == Approx(0.0).margin(0.1));
}

/**
 * Test the CNE optimizer on the Wood function.
 */
TEST_CASE("CNEStyblinskiTangFunctionTest", "[AdamTest]")
{
  StyblinskiTangFunction f(2);
  CNE optimizer(200, 1000, 0.2, 0.2, 0.2, 1e-5);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates[0] == Approx(-2.9).epsilon(0.01)); // 1% error tolerance.
  REQUIRE(coordinates[1] == Approx(-2.9).epsilon(0.01)); // 1% error tolerance.
}

/**
 * Test the CNE optimizer on the Matyas function.
 */
TEST_CASE("CNEMatyasFunctionTest", "[AdamTest]")
{
  MatyasFunction f;
  CNE optimizer(200, 1000, 0.2, 0.2, 0.2, 1e-5);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  // 3% error tolerance.
  REQUIRE(coordinates[0] == Approx(0.0).margin(0.03));
  REQUIRE(coordinates[1] == Approx(0.0).margin(0.03));
}

/**
 * Test the CNE optimizer on the Easom function.
 */
TEST_CASE("CNEEasomFunctionTest", "[AdamTest]")
{
  EasomFunction f;
  CNE optimizer(200, 1000, 0.2, 0.2, 0.2, 1e-5);

  arma::mat coordinates = arma::mat("2.9; 2.9");
  optimizer.Optimize(f, coordinates);

  // 5% error tolerance.
  REQUIRE((std::trunc(100.0 * coordinates[0]) / 100.0) ==
      Approx(3.14).epsilon(0.005));
  REQUIRE((std::trunc(100.0 * coordinates[1]) / 100.0) ==
      Approx(3.14).epsilon(0.005));
}

/**
 * Test the CNE optimizer on the Booth function.
 */
TEST_CASE("CNEBoothFunctionTest", "[AdamTest]")
{
  BoothFunction f;
  CNE optimizer(200, 1000, 0.2, 0.2, 0.2, 1e-5);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  // 2% tolerance.
  REQUIRE(coordinates[0] == Approx(1.0).epsilon(0.02));
  REQUIRE(coordinates[1] == Approx(3.0).epsilon(0.02));
}
