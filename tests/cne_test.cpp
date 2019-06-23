/**
 * @file cne_test.cpp
 * @author Marcus Edel
 * @author Kartik Nighania
 * @author Conrad Sanderson
 * @author Suryoday Basak
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
using namespace std;

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

/**
 * Test the CNE optimizer on Cross-in-Tray Function.
 */
TEST_CASE("CNECrossInTrayFunctionTest", "[CNETest]")
{
  CrossInTrayFunction f;
  CNE optimizer(500, 2000, 0.3, 0.3, 0.3, 1e-7);

  arma::mat coordinates = arma::mat("3; 3");
  optimizer.Optimize(f, coordinates);

  REQUIRE(abs(coordinates[0]) == Approx(1.34941).margin(0.1));
  REQUIRE(abs(coordinates[1]) == Approx(1.34941).margin(0.1));
}

/**
 * Test the CNE optimizer on Schaffer function N.4.
 */
TEST_CASE("CNESchafferFunctionN4Test", "[CNETest]")
{
  SchafferFunctionN4 f;
  CNE optimizer(1000, 3000, 0.8, 0.7, 0.4, 1e-9);

  arma::mat coordinates = arma::mat("0; 10");
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates[0] == Approx(0).margin(0.1));
  REQUIRE(abs(coordinates[1]) == Approx(1.25313).margin(0.1));
}

/**
 * Test the CNE optimizer on the Ackley Function.
 */
TEST_CASE("CNEAckleyFunctionTest", "[CNETest]")
{
  AckleyFunction f;
  CNE optimizer(500, 2000, 0.3, 0.3, 0.3, 1e-7);

  arma::mat coordinates = arma::mat("3; 3");
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates[0] == Approx(0).margin(0.1));
  REQUIRE(coordinates[1] == Approx(0).margin(0.1));
}

/**
 * Test the CNE optimizer on the Beale Function.
 */
TEST_CASE("CNEBealeFunctionTest", "[CNETest]")
{
  BealeFunction f;
  CNE optimizer(500, 2000, 0.3, 0.3, 0.3, 1e-7);

  arma::mat coordinates = arma::mat("3; 3");
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates[0] == Approx(3).margin(0.1));
  REQUIRE(coordinates[1] == Approx(0.5).margin(0.1));
}

/**
 * Test the CNE optimizer on the Goldstein-Price Function.
 */
TEST_CASE("CNEGoldsteinPriceFunctionTest", "[CNETest]")
{
  GoldsteinPriceFunction f;
  CNE optimizer(500, 2000, 0.3, 0.3, 0.3, 1e-7);

  arma::mat coordinates = arma::mat("1; 0");
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates[0] == Approx(0).margin(0.1));
  REQUIRE(coordinates[1] == Approx(-1).margin(0.1));
}

/**
 * Test the CNE optimizer on the Levi Function.
 */
TEST_CASE("CNELevyFunctionN13Test", "[CNETest]")
{
  LevyFunctionN13 f;
  CNE optimizer(500, 2000, 0.3, 0.3, 0.3, 1e-7);

  arma::mat coordinates = arma::mat("3; 3");
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates[0] == Approx(1).margin(0.1));
  REQUIRE(coordinates[1] == Approx(1).margin(0.1));
}

/**
 * Test the CNE optimizer on the Himmelblau Function.
 */
TEST_CASE("CNEHimmelblauFunctionTest", "[CNETest]")
{
  HimmelblauFunction f;
  CNE optimizer(500, 2000, 0.3, 0.3, 0.3, 1e-7);

  arma::mat coordinates = arma::mat("2; 1");
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates[0] == Approx(3.0).margin(0.1));
  REQUIRE(coordinates[1] == Approx(2.0).margin(0.1));
}

/**
 * Test the CNE optimizer on the Three-hump Camel Function.
 */
TEST_CASE("CNEThreeHumpCamelFunctionTest", "[CNETest]")
{
  ThreeHumpCamelFunction f;
  CNE optimizer(500, 2000, 0.3, 0.3, 0.3, 1e-7);

  arma::mat coordinates = arma::mat("2; 2");
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates[0] == Approx(0).margin(0.1));
  REQUIRE(coordinates[1] == Approx(0).margin(0.1));
}

/**
 * Test the CNE optimizer on Schaffer Function N.2.
 */
TEST_CASE("CNESchafferFunctionN2Test", "[CNETest]")
{
  SchafferFunctionN2 f;
  CNE optimizer(500, 2000, 0.3, 0.3, 0.3, 1e-12);

  arma::mat coordinates = arma::mat("10; 10");
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates[0] == Approx(0).margin(0.1));
  REQUIRE(coordinates[1] == Approx(0).margin(0.1));
}

