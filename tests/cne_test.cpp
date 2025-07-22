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

TEMPLATE_TEST_CASE("CNE_LogisticRegressionFunction", "[CNE]",
    ENS_ALL_TEST_TYPES)
{
  CNE opt(300, 150, 0.2, 0.2, 0.2, -1);
  LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(opt);
}

// The CrossInTray function doesn't optimize well with FP16.
TEMPLATE_TEST_CASE("CNE_CrossInTrayFunction", "[CNE]", ENS_ALL_TEST_TYPES)
{
  CrossInTrayFunction f;
  CNE optimizer(450, 1500, 0.3, 0.3, 0.3, -1);

  TestType coordinates = TestType("0.8; 1.8");
  optimizer.Optimize(f, coordinates);

  REQUIRE(abs(coordinates(0)) == Approx(1.34941).margin(
      10 * Tolerances<TestType>::LargeCoord));
  REQUIRE(abs(coordinates(1)) == Approx(1.34941).margin(
      10 * Tolerances<TestType>::LargeCoord));
}

TEMPLATE_TEST_CASE("CNE_AckleyFunction", "[CNE]", ENS_ALL_TEST_TYPES)
{
  CNE optimizer(450, 1500, 0.3, 0.3, 0.3, -1);
  FunctionTest<AckleyFunction>(optimizer,
      50 * Tolerances<TestType>::LargeObj,
      10 * Tolerances<TestType>::LargeCoord);
}

TEMPLATE_TEST_CASE("CNE_BealeFunction", "[CNE]", ENS_ALL_TEST_TYPES)
{
  CNE optimizer(450, 1500, 0.3, 0.3, 0.3, -1);
  FunctionTest<BealeFunction>(optimizer,
      50 * Tolerances<TestType>::LargeObj,
      10 * Tolerances<TestType>::LargeCoord);
}

TEMPLATE_TEST_CASE("CNE_GoldsteinPriceFunction", "[CNE]", ENS_ALL_TEST_TYPES)
{
  CNE optimizer(450, 1500, 0.3, 0.3, 0.1, -1);
  FunctionTest<GoldsteinPriceFunction>(optimizer,
      50 * Tolerances<TestType>::LargeObj,
      10 * Tolerances<TestType>::LargeCoord);
}

TEMPLATE_TEST_CASE("CNE_LevyFunctionN13", "[CNE]", ENS_ALL_TEST_TYPES)
{
  CNE optimizer(450, 1500, 0.3, 0.3, 0.02, -1);
  FunctionTest<LevyFunctionN13>(optimizer,
      50 * Tolerances<TestType>::LargeObj,
      10 * Tolerances<TestType>::LargeCoord);
}

TEMPLATE_TEST_CASE("CNE_HimmelblauFunction", "[CNE]", ENS_ALL_TEST_TYPES)
{
  HimmelblauFunction f;
  CNE optimizer(650, 3000, 0.3, 0.3, 0.3, 1e-7);

  // Allow multiple trials.
  TestType coordinates;
  const double coordTol1 = Tolerances<TestType>::LargeCoord * 12;
  const double coordTol2 = Tolerances<TestType>::LargeCoord * 8;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    coordinates = TestType("2; 1");
    optimizer.Optimize(f, coordinates);

    if (coordinates(0) == Approx(3.0).margin(coordTol1) &&
        coordinates(1) == Approx(2.0).margin(coordTol2))
      break;
  }

  REQUIRE(coordinates(0) == Approx(3.0).margin(coordTol1));
  REQUIRE(coordinates(1) == Approx(2.0).margin(coordTol2));
}

TEMPLATE_TEST_CASE("CNE_ThreeHumpCamelFunction", "[CNE]", ENS_ALL_TEST_TYPES)
{
  CNE optimizer(450, 1500, 0.3, 0.3, 0.3, -1);
  FunctionTest<ThreeHumpCamelFunction>(optimizer,
      50 * Tolerances<TestType>::LargeObj,
      10 * Tolerances<TestType>::LargeCoord);
}

// TODO: The CNE optimizer with the given parameter occasionally fails to find a
// solution for the Schaffer N4 function, so the function should be tested
// against another optimizer (PSO).
/**
 * Test the CNE optimizer on Schaffer function N.4.
 */
TEMPLATE_TEST_CASE("CNE_SchafferFunctionN4", "[CNE]", ENS_ALL_TEST_TYPES)
{
  SchafferFunctionN4 f;
  CNE optimizer(500, 1600, 0.3, 0.3, 0.3, -1);

  // We allow a few trials.
  const double coordTol = 5 * Tolerances<TestType>::LargeCoord;
  for (size_t trial = 0; trial < 5; ++trial)
  {
    TestType coordinates = TestType("0.5; 2");
    optimizer.Optimize(f, coordinates);

    if (trial != 4)
    {
      if (coordinates(0) != Approx(0).margin(coordTol))
        continue;
      if (abs(coordinates(1)) != Approx(1.25313).margin(coordTol))
        continue;
    }

    REQUIRE(coordinates(0) == Approx(0).margin(coordTol));
    REQUIRE(abs(coordinates(1)) == Approx(1.25313).margin(coordTol));

    // The test was successfull or reached the maximum number of trials.
    break;
  }
}

TEMPLATE_TEST_CASE("CNE_SchafferFunctionN2", "[CNE]", ENS_ALL_TEST_TYPES)
{
  // We allow a few trials in case convergence is not achieved.
  CNE optimizer(500, 1600, 0.3, 0.3, 0.3, -1);
  FunctionTest<SchafferFunctionN2>(optimizer,
      50 * Tolerances<TestType>::LargeObj,
      10 * Tolerances<TestType>::LargeCoord, 7);
}

#ifdef USE_COOT

TEMPLATE_TEST_CASE("CNE_LogisticRegressionFunction", "[CNE]",
    coot::mat, coot::fmat)
{
  CNE opt(300, 150, 0.2, 0.2, 0.2, -1);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(opt);
}

#endif
