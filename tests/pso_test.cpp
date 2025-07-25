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
#include "test_function_tools.hpp"
#include "test_types.hpp"

using namespace ens;
using namespace ens::test;
using namespace std;

TEMPLATE_TEST_CASE("LBestPSO_SphereFunction", "[PSO]", ENS_ALL_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  SphereFunctionType<TestType> f(4);
  // TODO(PR): remove MatType parameter from PSOType and hold an arma::vec for
  // the bounds internally, but convert to a MatType at the start of Optimize()?
  PSOType<TestType> s;

  TestType coords = f.template GetInitialPoint<TestType>();
  const ElemType finalValue = s.Optimize(f, coords);

  REQUIRE(finalValue <= (ElemType) Tolerances<TestType>::Obj);
  for (size_t j = 0; j < 4; ++j)
    REQUIRE(coords(j) <= (ElemType) Tolerances<TestType>::Coord);
}

TEMPLATE_TEST_CASE("LBestPSO_RosenbrockFunction", "[PSO]", ENS_ALL_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  RosenbrockFunction f;

  // Setting bounds for the initial swarm population.
  double lowerBound = 50;
  double upperBound = 60;

  const ElemType objTol = Tolerances<TestType>::LargeObj;
  const ElemType coordTol = Tolerances<TestType>::LargeCoord;

  // We allow a few trials.
  for (size_t trial = 0; trial < 3; ++trial)
  {
    PSOType<TestType> s(250, lowerBound, upperBound, 5000, 600,
        Tolerances<TestType>::Obj / 100, 2.05, 2.05);
    TestType coordinates = f.GetInitialPoint<TestType>();

    const ElemType result = s.Optimize(f, coordinates);

    if (trial != 4)
    {
      if (result != Approx(ElemType(0)).margin(objTol))
        continue;
      if (coordinates(0) != Approx(ElemType(1)).epsilon(coordTol))
        continue;
      if (coordinates(1) != Approx(ElemType(1)).epsilon(coordTol))
        continue;
    }

    REQUIRE(result == Approx(ElemType(0)).margin(objTol));
    REQUIRE(coordinates(0) == Approx(ElemType(1)).margin(coordTol));
    REQUIRE(coordinates(1) == Approx(ElemType(1)).margin(coordTol));

    break;
  }
}

TEMPLATE_TEST_CASE("LBestPSO_CrossInTrayFunction", "[PSO]", ENS_ALL_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  CrossInTrayFunction f;

  // Setting bounds for the initial swarm population.
  arma::Col<ElemType> lowerBound(2);
  arma::Col<ElemType> upperBound(2);
  lowerBound.fill(-1);
  upperBound.fill(1);

  // We allow many trials---sometimes this can have trouble converging.
  for (size_t trial = 0; trial < 3; ++trial)
  {
    PSOType<TestType> s(500, lowerBound, upperBound, 6000, 400,
        Tolerances<TestType>::Obj, 2.05, 2.05);
    TestType coordinates = TestType("10; 10");
    const ElemType result = s.Optimize(f, coordinates);
    const ElemType objTol = Tolerances<TestType>::LargeObj;
    const ElemType coordTol = Tolerances<TestType>::LargeCoord;

    if (trial != 2)
    {
      if (std::isinf(result) || std::isnan(result))
        continue;
      if (result != Approx(ElemType(-2.06261)).margin(objTol))
        continue;
      if (abs(coordinates(0)) != Approx(ElemType(1.34941)).margin(coordTol))
        continue;
      if (abs(coordinates(1)) != Approx(ElemType(1.34941)).margin(coordTol))
        continue;
    }

    REQUIRE(result == Approx(ElemType(-2.06261)).margin(objTol));
    REQUIRE(abs(coordinates(0)) == Approx(ElemType(1.34941)).margin(coordTol));
    REQUIRE(abs(coordinates(1)) == Approx(ElemType(1.34941)).margin(coordTol));

    break;
  }
}

TEMPLATE_TEST_CASE("LBestPSO_AckleyFunction", "[PSO]", ENS_ALL_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  AckleyFunction f;

  // Setting bounds for the initial swarm population.
  arma::Col<ElemType> lowerBound(2);
  arma::Col<ElemType> upperBound(2);
  lowerBound.fill(4);
  upperBound.fill(5);

  PSOType<TestType> s(64, lowerBound, upperBound);
  TestType coordinates = TestType("5; 5");
  const ElemType result = s.Optimize(f, coordinates);

  const ElemType objTol = Tolerances<TestType>::LargeObj;
  const ElemType coordTol = Tolerances<TestType>::LargeCoord;

  REQUIRE(result == Approx(ElemType(0)).margin(objTol));
  REQUIRE(coordinates(0) == Approx(ElemType(0)).margin(coordTol));
  REQUIRE(coordinates(1) == Approx(ElemType(0)).margin(coordTol));
}

TEMPLATE_TEST_CASE("LBestPSO_BealeFunction", "[PSO]", ENS_ALL_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  BealeFunction f;

  // Setting bounds for the initial swarm population.
  arma::Col<ElemType> lowerBound(2);
  arma::Col<ElemType> upperBound(2);
  lowerBound.fill(3);
  upperBound.fill(4);

  PSOType<TestType> s(64, lowerBound, upperBound);

  TestType coordinates = TestType("4.5; 4.5");
  const ElemType result = s.Optimize(f, coordinates);

  const ElemType objTol = Tolerances<TestType>::LargeObj;
  const ElemType coordTol = Tolerances<TestType>::LargeCoord;

  REQUIRE(result == Approx(ElemType(0)).margin(objTol));
  REQUIRE(coordinates(0) == Approx(ElemType(3)).margin(coordTol));
  REQUIRE(coordinates(1) == Approx(ElemType(0.5)).margin(coordTol));
}

TEMPLATE_TEST_CASE("LBestPSO_GoldsteinPriceFunction", "[PSO]",
    ENS_ALL_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  GoldsteinPriceFunction f;

  // Setting bounds for the initial swarm population.
  arma::Col<ElemType> lowerBound(2);
  arma::Col<ElemType> upperBound(2);
  lowerBound.fill(ElemType(1.6));
  upperBound.fill(ElemType(2));

  // Allow a few trials in case of failure.
  for (size_t trial = 0; trial < 3; ++trial)
  {
    PSOType<TestType> s(64, lowerBound, upperBound);

    TestType coordinates = TestType("1; 0");
    s.Optimize(f, coordinates);

    const ElemType coordTol = Tolerances<TestType>::LargeCoord;

    if (trial != 2)
    {
      if (coordinates(0) != Approx(ElemType(0)).margin(coordTol))
        continue;
      if (coordinates(1) != Approx(ElemType(-1)).margin(coordTol))
        continue;
    }

    REQUIRE(coordinates(0) == Approx(ElemType(0)).margin(coordTol));
    REQUIRE(coordinates(1) == Approx(ElemType(-1)).margin(coordTol));

    break;
  }
}

TEMPLATE_TEST_CASE("LBestPSO_LevyFunctionN13", "[PSO]", ENS_ALL_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  LevyFunctionN13 f;

  // Setting bounds for the initial swarm population.
  arma::Col<ElemType> lowerBound(2);
  arma::Col<ElemType> upperBound(2);
  lowerBound.fill(-10);
  upperBound.fill(-9);

  PSOType<TestType> s(64, lowerBound, upperBound);

  TestType coordinates = TestType("3; 3");
  s.Optimize(f, coordinates);

  const ElemType coordTol = Tolerances<TestType>::LargeCoord;

  REQUIRE(coordinates(0) == Approx(ElemType(1)).margin(coordTol));
  REQUIRE(coordinates(1) == Approx(ElemType(1)).margin(coordTol));
}

TEMPLATE_TEST_CASE("LBestPSO_HimmelblauFunction", "[PSO]", ENS_ALL_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  HimmelblauFunction f;

  // Setting bounds for the initial swarm population.
  arma::Col<ElemType> lowerBound(2);
  arma::Col<ElemType> upperBound(2);
  lowerBound.fill(0);
  upperBound.fill(1);

  // This optimization could take a couple trials to get right.
  TestType coordinates;
  const double coordTol = Tolerances<TestType>::LargeCoord;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    PSOType<TestType> s(64, lowerBound, upperBound);

    coordinates = TestType("2; 1");
    s.Optimize(f, coordinates);

    if (coordinates(0) == Approx(ElemType(3)).margin(coordTol))
      break;
    if (coordinates(1) == Approx(ElemType(2)).margin(coordTol))
      break;
  }

  REQUIRE(coordinates(0) == Approx(ElemType(3)).margin(coordTol));
  REQUIRE(coordinates(1) == Approx(ElemType(2)).margin(coordTol));
}

TEMPLATE_TEST_CASE("LBestPSO_ThreeHumpCamelFunction", "[PSO]",
    ENS_ALL_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  ThreeHumpCamelFunction f;

  // Setting bounds for the initial swarm population.
  arma::Col<ElemType> lowerBound(2);
  arma::Col<ElemType> upperBound(2);
  lowerBound.fill(-5);
  upperBound.fill(-4);

  PSOType<TestType> s(64, lowerBound, upperBound);

  TestType coordinates = TestType("2; 2");
  s.Optimize(f, coordinates);

  const ElemType coordTol = Tolerances<TestType>::LargeCoord;

  REQUIRE(coordinates(0) == Approx(ElemType(0)).margin(coordTol));
  REQUIRE(coordinates(1) == Approx(ElemType(0)).margin(coordTol));
}

TEMPLATE_TEST_CASE("LBestPSO_SchafferFunctionN2", "[PSO]", ENS_ALL_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  SchafferFunctionN2 f;

  // Setting bounds for the initial swarm population.
  arma::Col<ElemType> lowerBound(2);
  arma::Col<ElemType> upperBound(2);
  lowerBound.fill(40);
  upperBound.fill(50);

  PSOType<TestType> s(500, lowerBound, upperBound);
  TestType coordinates = TestType("10; 10");
  s.Optimize(f, coordinates);

  ElemType coordTol = Tolerances<TestType>::LargeCoord;
  // Low-precision will need a larger tolerance.
  if (sizeof(ElemType) < 4)
    coordTol *= 5;

  REQUIRE(coordinates(0) == Approx(ElemType(0)).margin(coordTol));
  REQUIRE(coordinates(1) == Approx(ElemType(0)).margin(coordTol));
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
