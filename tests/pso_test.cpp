/**
 * @file pso_test.cpp
 * @author Suryoday Basak
 * @author Chintan Soni
 *
 * Test file for PSO optimizer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <ensmallen.hpp>
#include "catch.hpp"

using namespace ens;
using namespace ens::test;

TEST_CASE("LBestPSOSphereFunctionTest", "[PSOTest]")
{
  SphereFunction f(4);
  LBestPSO s;

  arma::vec coords = f.GetInitialPoint();
  if (!s.Optimize(f, coords))
    FAIL("LBest PSO optimization reported failure for Sphere Function.");
  
  double finalValue = f.Evaluate(coords);
  //BOOST_REQUIRE_SMALL(result, 1e-5);
  REQUIRE(finalValue <= 1e-5);
  for (size_t j = 0; j < 4; ++j)
    //BOOST_REQUIRE_SMALL(coordinates[j], 1e-3);
	REQUIRE(coords[j] <= 1e-3); //work on this
}

TEST_CASE("LBestPSORosenbrockTest","[PSOTest]")
{
  RosenbrockFunction f;

  //Setting bounds for the initial swarm population.
  arma::vec lowerBound(2);
  arma::vec upperBound(2);
  lowerBound.fill(50);
  upperBound.fill(60); 

  LBestPSO s(64, lowerBound, upperBound);
  arma::vec coordinates = f.GetInitialPoint();

  const double result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(0.0).margin(1e-5));
  REQUIRE(coordinates[0] == Approx(1.0).epsilon(1e-4));
  REQUIRE(coordinates[1] == Approx(1.0).epsilon(1e-4));
}

/**
 * Test the CNE optimizer on Cross-in-Tray Function.
 */
TEST_CASE("LBestPSOCrossInTrayFunctionTest", "[PSOTest]")
{
  CrossInTrayFunction f;

  //Setting bounds for the initial swarm population.
  arma::vec lowerBound(2);
  arma::vec upperBound(2);
  lowerBound.fill(8);
  upperBound.fill(9);

  LBestPSO s(64, lowerBound, upperBound);
  arma::mat coordinates = arma::mat("10; 10");
  const double result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(-2.06261).margin(0.01));
  REQUIRE(abs(coordinates[0]) == Approx(1.34941).margin(0.01));
  REQUIRE(abs(coordinates[1]) == Approx(1.34941).margin(0.01));
}

/**
 * Test the CNE optimizer on the Ackley Function.
 */
TEST_CASE("LBestPSOAckleyFunctionTest", "[PSOTest]")
{
  AckleyFunction f;

  //Setting bounds for the initial swarm population.
  arma::vec lowerBound(2);
  arma::vec upperBound(2);
  lowerBound.fill(4);
  upperBound.fill(5);

  LBestPSO s(64, lowerBound, upperBound);
  arma::mat coordinates = arma::mat("5; 5");
  const double result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(0).margin(0.01));
  REQUIRE(coordinates[0] == Approx(0).margin(0.01));
  REQUIRE(coordinates[1] == Approx(0).margin(0.01));
}

/**
 * Test the CNE optimizer on the Beale Function.
 */
TEST_CASE("LBestPSOBealeFunctionTest", "[PSOTest]")
{
  BealeFunction f;

  //Setting bounds for the initial swarm population.
  arma::vec lowerBound(2);
  arma::vec upperBound(2);
  lowerBound.fill(3);
  upperBound.fill(4);

  LBestPSO s(64, lowerBound, upperBound);

  arma::mat coordinates = arma::mat("4.5; 4.5");
  const double result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(0).margin(0.01));
  REQUIRE(coordinates[0] == Approx(3).margin(0.01));
  REQUIRE(coordinates[1] == Approx(0.5).margin(0.01));
}

