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

TEMPLATE_TEST_CASE("LBestPSO_SphereFunction", "[PSO]", arma::mat)
{
  typedef typename TestType::elem_type ElemType;

  SphereFunction f(4);
  LBestPSO s;

  TestType coords = f. template GetInitialPoint<TestType>();
  if (!s.Optimize(f, coords))
    FAIL("LBest PSO optimization reported failure for Sphere Function.");

  ElemType finalValue = f.Evaluate(coords);
  REQUIRE(finalValue <= 1e-5);
  for (size_t j = 0; j < 4; ++j)
    REQUIRE(coords(j) <= 1e-3);
}

TEMPLATE_TEST_CASE("LBestPSO_RosenbrockFunction", "[PSO]",
    arma::mat, arma::fmat)
{
  typedef typename TestType::elem_type ElemType;

  RosenbrockFunction f;

  // Setting bounds for the initial swarm population.
  double lowerBound = 50;
  double upperBound = 60;

  // We allow a few trials.
  for (size_t trial = 0; trial < 3; ++trial)
  {
    LBestPSO s(250, lowerBound, upperBound, 5000, 600, 1e-30, 2.05, 2.05);
    TestType coordinates = f.GetInitialPoint<TestType>();

    const ElemType result = s.Optimize(f, coordinates);

    if (trial != 4)
    {
      if (result != Approx(0.0).margin(0.03))
        continue;
      if (coordinates(0) != Approx(1.0).epsilon(0.03))
        continue;
      if (coordinates(1) != Approx(1.0).epsilon(0.03))
        continue;
    }

    REQUIRE(result == Approx(0.0).margin(0.03));
    REQUIRE(coordinates(0) == Approx(1.0).margin(0.03));
    REQUIRE(coordinates(1) == Approx(1.0).margin(0.03));
  }
}

TEMPLATE_TEST_CASE("LBestPSO_CrossInTrayFunction", "[PSO]", arma::mat)
{
  typedef typename TestType::elem_type ElemType;

  CrossInTrayFunction f;

  // Setting bounds for the initial swarm population.
  arma::Col<ElemType> lowerBound(2);
  arma::Col<ElemType> upperBound(2);
  lowerBound.fill(-1);
  upperBound.fill(1);

  // We allow many trials---sometimes this can have trouble converging.
  for (size_t trial = 0; trial < 15; ++trial)
  {
    LBestPSO s(500, lowerBound, upperBound, 6000, 400, 1e-30, 2.05, 2.05);
    TestType coordinates = TestType("10; 10");
    const ElemType result = s.Optimize(f, coordinates);

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

TEMPLATE_TEST_CASE("LBestPSO_AckleyFunction", "[PSO]", arma::mat)
{
  typedef typename TestType::elem_type ElemType;

  AckleyFunction f;

  // Setting bounds for the initial swarm population.
  arma::Col<ElemType> lowerBound(2);
  arma::Col<ElemType> upperBound(2);
  lowerBound.fill(4);
  upperBound.fill(5);

  LBestPSO s(64, lowerBound, upperBound);
  TestType coordinates = TestType("5; 5");
  const ElemType result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(0).margin(0.01));
  REQUIRE(coordinates(0) == Approx(0).margin(0.01));
  REQUIRE(coordinates(1) == Approx(0).margin(0.01));
}

TEMPLATE_TEST_CASE("LBestPSO_BealeFunction", "[PSO]", arma::mat)
{
  typedef typename TestType::elem_type ElemType;

  BealeFunction f;

  // Setting bounds for the initial swarm population.
  arma::Col<ElemType> lowerBound(2);
  arma::Col<ElemType> upperBound(2);
  lowerBound.fill(3);
  upperBound.fill(4);

  LBestPSO s(64, lowerBound, upperBound);

  TestType coordinates = TestType("4.5; 4.5");
  const ElemType result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(0).margin(0.01));
  REQUIRE(coordinates(0) == Approx(3).margin(0.01));
  REQUIRE(coordinates(1) == Approx(0.5).margin(0.01));
}

TEMPLATE_TEST_CASE("LBestPSO_GoldsteinPriceFunction", "[PSO]", arma::mat)
{
  typedef typename TestType::elem_type ElemType;

  GoldsteinPriceFunction f;

  // Setting bounds for the initial swarm population.
  arma::Col<ElemType> lowerBound(2);
  arma::Col<ElemType> upperBound(2);
  lowerBound.fill(1.6);
  upperBound.fill(2);

  // Allow a few trials in case of failure.
  for (size_t trial = 0; trial < 10; ++trial)
  {
    LBestPSO s(64, lowerBound, upperBound);

    TestType coordinates = TestType("1; 0");
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

TEMPLATE_TEST_CASE("LBestPSO_LevyFunctionN13", "[PSO]", arma::mat)
{
  typedef typename TestType::elem_type ElemType;

  LevyFunctionN13 f;

  // Setting bounds for the initial swarm population.
  arma::Col<ElemType> lowerBound(2);
  arma::Col<ElemType> upperBound(2);
  lowerBound.fill(-10);
  upperBound.fill(-9);

  LBestPSO s(64, lowerBound, upperBound);

  TestType coordinates = TestType("3; 3");
  s.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(1).margin(0.01));
  REQUIRE(coordinates(1) == Approx(1).margin(0.01));
}

TEMPLATE_TEST_CASE("LBestPSO_HimmelblauFunction", "[PSO]", arma::mat)
{
  typedef typename TestType::elem_type ElemType;

  HimmelblauFunction f;

  // Setting bounds for the initial swarm population.
  arma::Col<ElemType> lowerBound(2);
  arma::Col<ElemType> upperBound(2);
  lowerBound.fill(0);
  upperBound.fill(1);

  LBestPSO s(64, lowerBound, upperBound);

  TestType coordinates = TestType("2; 1");
  s.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(3.0).margin(0.01));
  REQUIRE(coordinates(1) == Approx(2.0).margin(0.01));
}

TEMPLATE_TEST_CASE("LBestPSO_ThreeHumpCamelFunction", "[PSO]", arma::mat)
{
  typedef typename TestType::elem_type ElemType;

  ThreeHumpCamelFunction f;

  // Setting bounds for the initial swarm population.
  arma::Col<ElemType> lowerBound(2);
  arma::Col<ElemType> upperBound(2);
  lowerBound.fill(-5);
  upperBound.fill(-4);

  LBestPSO s(64, lowerBound, upperBound);

  TestType coordinates = TestType("2; 2");
  s.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0).margin(0.01));
  REQUIRE(coordinates(1) == Approx(0).margin(0.01));
}

TEMPLATE_TEST_CASE("LBestPSO_SchafferFunctionN2", "[PSO]", arma::mat)
{
  typedef typename TestType::elem_type ElemType;

  SchafferFunctionN2 f;

  // Setting bounds for the initial swarm population.
  arma::Col<ElemType> lowerBound(2);
  arma::Col<ElemType> upperBound(2);
  lowerBound.fill(40);
  upperBound.fill(50);

  LBestPSO s(500, lowerBound, upperBound);
  TestType coordinates = TestType("10; 10");
  s.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0).margin(0.01));
  REQUIRE(coordinates(1) == Approx(0).margin(0.01));
}

#ifdef USE_COOT

TEMPLATE_TEST_CASE("LBestPSO_SchafferFunctionN2", "[PSO]",
    coot::mat, coot::fmat)
{
  typedef typename TestType::elem_type ElemType;

  SchafferFunctionN2 f;

  // Setting bounds for the initial swarm population.
  coot::Col<ElemType> lowerBound(2);
  coot::Col<ElemType> upperBound(2);
  lowerBound.fill(40);
  upperBound.fill(50);

  PSOType<TestType> s(500, lowerBound, upperBound);
  TestType coordinates = TestType("10; 10");
  s.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0).margin(0.01));
  REQUIRE(coordinates(1) == Approx(0).margin(0.01));
}

#endif
