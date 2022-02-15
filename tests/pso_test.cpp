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
using namespace std;

/**
 * Test the PSO optimizer on the Sphere Function.  Use arma::mat.
 */
TEST_CASE("LBestPSOSphereFunctionTest", "[PSOTest]")
{
  SphereFunction f(4);
  LBestPSO s;

  arma::mat coords = f.GetInitialPoint<arma::mat>();
  if (!s.Optimize(f, coords))
    FAIL("LBest PSO optimization reported failure for Sphere Function.");

  double finalValue = f.Evaluate(coords);
  REQUIRE(finalValue <= 1e-5);
  for (size_t j = 0; j < 4; ++j)
    REQUIRE(coords(j) <= 1e-3);
}

/**
 * Test the PSO optimizer on the Sphere Function.  Use arma::fmat.
 */
TEST_CASE("LBestPSOSphereFunctionFMatTest", "[PSOTest]")
{
  SphereFunction f(4);
  LBestPSO s;

  arma::fmat coords = f.GetInitialPoint<arma::fmat>();
  if (!s.Optimize(f, coords))
    FAIL("LBest PSO optimization reported failure for Sphere Function.");

  double finalValue = f.Evaluate(coords);
  REQUIRE(finalValue <= 1e-5);
  for (size_t j = 0; j < 4; ++j)
    REQUIRE(coords(j) <= 1e-3);
}

/**
 * Test the PSO optimizer on the Rosenbrock Function.  Use arma::mat.
 */
TEST_CASE("LBestPSORosenbrockTest","[PSOTest]")
{
  RosenbrockFunction f;

  // Setting bounds for the initial swarm population.
  arma::vec lowerBound(2);
  arma::vec upperBound(2);
  lowerBound.fill(50);
  upperBound.fill(60);

  // We allow a few trials.
  for (size_t trial = 0; trial < 3; ++trial)
  {
    LBestPSO s(250, lowerBound, upperBound, 3000, 600, 1e-30, 2.05, 2.05);
    arma::vec coordinates = f.GetInitialPoint();

    const double result = s.Optimize(f, coordinates);

    if (trial != 2)
    {
      if (result != Approx(0.0).margin(0.03))
        continue;
      if (coordinates(0) != Approx(1.0).margin(0.02))
        continue;
      if (coordinates(1) != Approx(1.0).margin(0.02))
        continue;
    }

    REQUIRE(result == Approx(0.0).margin(0.03));
    REQUIRE(coordinates(0) == Approx(1.0).margin(0.02));
    REQUIRE(coordinates(1) == Approx(1.0).margin(0.02));
  }
}

/**
 * Test the PSO optimizer on the Rosenbrock Function.  Use arma::fmat.
 */
TEST_CASE("LBestPSORosenbrockFMatTest","[PSOTest]")
{
  RosenbrockFunction f;

  // Setting bounds for the initial swarm population.
  arma::vec lowerBound(2);
  arma::vec upperBound(2);
  lowerBound.fill(50);
  upperBound.fill(60);

  // We allow a few trials.
  for (size_t trial = 0; trial < 5; ++trial)
  {
    LBestPSO s(250, lowerBound, upperBound, 5000, 600, 1e-30, 2.05, 2.05);
    arma::fmat coordinates = f.GetInitialPoint<arma::fmat>();

    const double result = s.Optimize(f, coordinates);

    if (trial != 4)
    {
      if (result != Approx(0.0).margin(0.03))
        continue;
      if (coordinates(0) != Approx(1.0).margin(0.03))
        continue;
      if (coordinates(1) != Approx(1.0).margin(0.03))
        continue;
    }

    REQUIRE(result == Approx(0.0).margin(0.03));
    REQUIRE(coordinates(0) == Approx(1.0).margin(0.03));
    REQUIRE(coordinates(1) == Approx(1.0).margin(0.03));
  }
}

/**
 * Test the PSO optimizer on the Rosenbrock function with lowerBound and
 * upperbound of type double.
 */
TEST_CASE("LBestPSORosenbrockDoubleTest","[PSOTest]")
{
  RosenbrockFunction f;

  // Setting bounds for the initial swarm population.
  double lowerBound = 50;
  double upperBound = 60;

  // We allow a few trials.
  for (size_t trial = 0; trial < 3; ++trial)
  {
    LBestPSO s(250, lowerBound, upperBound, 5000, 400, 1e-30, 2.05, 2.05);
    arma::vec coordinates = f.GetInitialPoint();

    const double result = s.Optimize(f, coordinates);

    if (trial != 2)
    {
      if (result != Approx(0.0).margin(1e-3))
        continue;
      if (coordinates(0) != Approx(1.0).epsilon(1e-2))
        continue;
      if (coordinates(1) != Approx(1.0).epsilon(1e-2))
        continue;
    }

    REQUIRE(result == Approx(0.0).margin(0.005));
    REQUIRE(coordinates(0) == Approx(1.0).margin(0.005));
    REQUIRE(coordinates(1) == Approx(1.0).margin(0.005));
  }
}

/**
 * Test the PSO optimizer on Cross-in-Tray Function.
 */
TEST_CASE("LBestPSOCrossInTrayFunctionTest", "[PSOTest]")
{
  CrossInTrayFunction f;

  // Setting bounds for the initial swarm population.
  arma::vec lowerBound(2);
  arma::vec upperBound(2);
  lowerBound.fill(-1);
  upperBound.fill(1);

  // We allow many trials---sometimes this can have trouble converging.
  for (size_t trial = 0; trial < 15; ++trial)
  {
    LBestPSO s(500, lowerBound, upperBound, 6000, 400, 1e-30, 2.05, 2.05);
    arma::mat coordinates = arma::mat("10; 10");
    const double result = s.Optimize(f, coordinates);

    if (trial != 14)
    {
      if (std::isinf(result) || std::isnan(result))
        continue;
      if (result != Approx(-2.06261).margin(0.01))
        continue;
      if (abs(coordinates(0)) != Approx(1.34941).margin(0.01))
        continue;
      if (abs(coordinates(1)) != Approx(1.34941).margin(0.01))
        continue;
    }

    REQUIRE(result == Approx(-2.06261).margin(0.01));
    REQUIRE(abs(coordinates(0)) == Approx(1.34941).margin(0.01));
    REQUIRE(abs(coordinates(1)) == Approx(1.34941).margin(0.01));
  }
}

/**
 * Test the PSO optimizer on the Ackley Function.
 */
TEST_CASE("LBestPSOAckleyFunctionTest", "[PSOTest]")
{
  AckleyFunction f;

  // Setting bounds for the initial swarm population.
  arma::vec lowerBound(2);
  arma::vec upperBound(2);
  lowerBound.fill(4);
  upperBound.fill(5);

  LBestPSO s(64, lowerBound, upperBound);
  arma::mat coordinates = arma::mat("5; 5");
  const double result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(0).margin(0.01));
  REQUIRE(coordinates(0) == Approx(0).margin(0.01));
  REQUIRE(coordinates(1) == Approx(0).margin(0.01));
}

/**
 * Test the PSO optimizer on the Beale Function.
 */
TEST_CASE("LBestPSOBealeFunctionTest", "[PSOTest]")
{
  BealeFunction f;

  // Setting bounds for the initial swarm population.
  arma::vec lowerBound(2);
  arma::vec upperBound(2);
  lowerBound.fill(3);
  upperBound.fill(4);

  LBestPSO s(64, lowerBound, upperBound);

  arma::mat coordinates = arma::mat("4.5; 4.5");
  const double result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(0).margin(0.01));
  REQUIRE(coordinates(0) == Approx(3).margin(0.01));
  REQUIRE(coordinates(1) == Approx(0.5).margin(0.01));
}

/**
 * Test the PSO optimizer on the Goldstein-Price Function.
 */
TEST_CASE("LBestPSOGoldsteinPriceFunctionTest", "[PSOTest]")
{
  GoldsteinPriceFunction f;

  // Setting bounds for the initial swarm population.
  arma::vec lowerBound(2);
  arma::vec upperBound(2);
  lowerBound.fill(1.6);
  upperBound.fill(2);

  // Allow a few trials in case of failure.
  for (size_t trial = 0; trial < 10; ++trial)
  {
    LBestPSO s(64, lowerBound, upperBound);

    arma::mat coordinates = arma::mat("1; 0");
    s.Optimize(f, coordinates);

    if (trial != 9)
    {
      if (coordinates(0) != Approx(0).margin(0.01))
        continue;
      if (coordinates(1) != Approx(-1).margin(0.01))
        continue;
    }

    REQUIRE(coordinates(0) == Approx(0).margin(0.01));
    REQUIRE(coordinates(1) == Approx(-1).margin(0.01));
  }
}

/**
 * Test the PSO optimizer on the Levi Function.
 */
TEST_CASE("LBestPSOLevyFunctionN13Test", "[PSOTest]")
{
  LevyFunctionN13 f;

  // Setting bounds for the initial swarm population.
  arma::vec lowerBound(2);
  arma::vec upperBound(2);
  lowerBound.fill(-10);
  upperBound.fill(-9);

  LBestPSO s(64, lowerBound, upperBound);

  arma::mat coordinates = arma::mat("3; 3");
  s.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(1).margin(0.01));
  REQUIRE(coordinates(1) == Approx(1).margin(0.01));
}

/**
 * Test the PSO optimizer on the Himmelblau Function.
 */
TEST_CASE("LBestPSOHimmelblauFunctionTest", "[PSOTest]")
{
  HimmelblauFunction f;

  // Setting bounds for the initial swarm population.
  arma::vec lowerBound(2);
  arma::vec upperBound(2);
  lowerBound.fill(0);
  upperBound.fill(1);

  LBestPSO s(64, lowerBound, upperBound);

  arma::mat coordinates = arma::mat("2; 1");
  s.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(3.0).margin(0.01));
  REQUIRE(coordinates(1) == Approx(2.0).margin(0.01));
}

/**
 * Test the PSO optimizer on the Three-hump Camel Function.
 */
TEST_CASE("LBestPSOThreeHumpCamelFunctionTest", "[PSOTest]")
{
  ThreeHumpCamelFunction f;

  // Setting bounds for the initial swarm population.
  arma::vec lowerBound(2);
  arma::vec upperBound(2);
  lowerBound.fill(-5);
  upperBound.fill(-4);

  LBestPSO s(64, lowerBound, upperBound);

  arma::mat coordinates = arma::mat("2; 2");
  s.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0).margin(0.01));
  REQUIRE(coordinates(1) == Approx(0).margin(0.01));
}

/**
 * Test the PSO optimizer on Schaffer Function N.2.
 */
TEST_CASE("LBestPSOSchafferFunctionN2Test", "[PSOTest]")
{
  SchafferFunctionN2 f;

  // Setting bounds for the initial swarm population.
  arma::vec lowerBound(2);
  arma::vec upperBound(2);
  lowerBound.fill(40);
  upperBound.fill(50);

  LBestPSO s(500, lowerBound, upperBound);
  arma::mat coordinates = arma::mat("10; 10");
  s.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0).margin(0.01));
  REQUIRE(coordinates(1) == Approx(0).margin(0.01));
}

// TODO: With future improvements in metaheuristic optimisers in ensmallen,
// try to optimize this function.
/**
 * Test the PSO optimizer on Schaffer function N.4.
 */
/*
TEST_CASE("LBestPSOScafferFunctionN4Test", "[PSOTest]")
{
  SchafferFunctionN4 f;

  // Setting bounds for the initial swarm population.
  arma::vec lowerBound(2);
  arma::vec upperBound(2);
  lowerBound.fill(-0.01);
  upperBound.fill(1.5);

  LBestPSO s(25000, lowerBound, upperBound, 4000, 40, 1e-40, 1.5, 1.0);
  arma::mat coordinates = arma::mat("0; 10");
  const double result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(0.292579).margin(0.1));
  REQUIRE(coordinates(0) == Approx(0).margin(0.1));
  REQUIRE(abs(coordinates(1)) == Approx(1.25313).margin(0.1));
}
*/
