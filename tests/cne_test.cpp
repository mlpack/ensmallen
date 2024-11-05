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

TEMPLATE_TEST_CASE("CNELogisticRegressionTest", "[CNE]",
    arma::mat, arma::fmat)
{
  CNE opt(300, 150, 0.2, 0.2, 0.2, -1);
  LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(
      opt, 0.003, 0.006, 1);
}

TEST_CASE("CNECrossInTrayFunctionTest", "[CNE]")
{
  CrossInTrayFunction f;
  CNE optimizer(450, 1500, 0.3, 0.3, 0.3, -1);

  arma::mat coordinates = arma::mat("0.8; 1.8");
  optimizer.Optimize(f, coordinates);

  REQUIRE(abs(coordinates(0)) == Approx(1.34941).margin(0.1));
  REQUIRE(abs(coordinates(1)) == Approx(1.34941).margin(0.1));
}

TEST_CASE("CNEAckleyFunctionTest", "[CNE]")
{
  CNE optimizer(450, 1500, 0.3, 0.3, 0.3, -1);
  FunctionTest<AckleyFunction>(optimizer, 0.5, 0.1);
}

TEST_CASE("CNEBealeFunctionTest", "[CNE]")
{
  CNE optimizer(450, 1500, 0.3, 0.3, 0.3, -1);
  FunctionTest<BealeFunction>(optimizer, 0.5, 0.1);
}

TEST_CASE("CNEGoldsteinPriceFunctionTest", "[CNE]")
{
  CNE optimizer(450, 1500, 0.3, 0.3, 0.1, -1);
  FunctionTest<GoldsteinPriceFunction>(optimizer, 0.5, 0.1);
}

TEST_CASE("CNELevyFunctionN13Test", "[CNE]")
{
  CNE optimizer(450, 1500, 0.3, 0.3, 0.02, -1);
  FunctionTest<LevyFunctionN13>(optimizer, 0.5, 0.1);
}

TEST_CASE("CNEHimmelblauFunctionTest", "[CNE]")
{
  HimmelblauFunction f;
  CNE optimizer(650, 3000, 0.3, 0.3, 0.3, 1e-7);

  // Allow multiple trials.
  arma::mat coordinates;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    coordinates = arma::mat("2; 1");
    optimizer.Optimize(f, coordinates);

    if (coordinates(0) == Approx(3.0).margin(0.6) &&
        coordinates(1) == Approx(2.0).margin(0.2))
      break;
  }

  REQUIRE(coordinates(0) == Approx(3.0).margin(0.6));
  REQUIRE(coordinates(1) == Approx(2.0).margin(0.2));
}

TEST_CASE("CNEThreeHumpCamelFunctionTest", "[CNE]")
{
  CNE optimizer(450, 1500, 0.3, 0.3, 0.3, -1);
  FunctionTest<ThreeHumpCamelFunction>(optimizer, 0.5, 0.1);
}

// TODO: The CNE optimizer with the given parameter occasionally fails to find a
// solution for the Schaffer N4 function, so the function should be tested
// against another optimizer (PSO).
/**
 * Test the CNE optimizer on Schaffer function N.4.
 */
TEST_CASE("CNESchafferFunctionN4Test", "[CNE]")
{
  SchafferFunctionN4 f;
  CNE optimizer(500, 1600, 0.3, 0.3, 0.3, -1);

  // We allow a few trials.
  for (size_t trial = 0; trial < 5; ++trial)
  {
    arma::mat coordinates = arma::mat("0.5; 2");
    optimizer.Optimize(f, coordinates);

    if (trial != 4)
    {
      if (coordinates(0) != Approx(0).margin(0.1))
        continue;
      if (abs(coordinates(1)) != Approx(1.25313).margin(0.1))
        continue;
    }

    REQUIRE(coordinates(0) == Approx(0).margin(0.1));
    REQUIRE(abs(coordinates(1)) == Approx(1.25313).margin(0.1));

    // The test was successfull or reached the maximum number of trials.
    break;
  }
}

TEST_CASE("CNESchafferFunctionN2Test", "[CNE]")
{
  // We allow a few trials in case convergence is not achieved.
  CNE optimizer(500, 1600, 0.3, 0.3, 0.3, -1);
  FunctionTest<SchafferFunctionN2>(optimizer, 0.5, 0.1, 7);
}

#ifdef USE_COOT

// TEMPLATE_TEST_CASE("CNELogisticRegressionTest", "[CNE]",
//     coot::mat, coot::fmat)
// {
//   CNE opt(300, 150, 0.2, 0.2, 0.2, -1);
//   LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
//       opt, 0.003, 0.006, 1);
// }

#endif