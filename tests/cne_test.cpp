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
 * Train and test a logistic regression function using CNE optimizer.
 */
TEST_CASE("CNELogisticRegressionTest", "[CNETest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

  CNE opt(300, 150, 0.2, 0.2, 0.2, -1);
  arma::mat coordinates = lr.GetInitialPoint();
  opt.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}

/**
 * Train and test a logistic regression function using CNE optimizer.  Use
 * arma::fmat.
 */
TEST_CASE("CNELogisticRegressionFMatTest", "[CNETest]")
{
  arma::fmat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<arma::fmat> lr(shuffledData, shuffledResponses, 0.5);

  CNE opt(300, 150, 0.2, 0.2, 0.2, -1);
  arma::fmat coordinates = lr.GetInitialPoint();
  opt.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}

/**
 * Test the CNE optimizer on Cross-in-Tray Function.
 */
TEST_CASE("CNECrossInTrayFunctionTest", "[CNETest]")
{
  CrossInTrayFunction f;
  CNE optimizer(450, 1500, 0.3, 0.3, 0.3, -1);

  arma::mat coordinates = arma::mat("0.8; 1.8");
  optimizer.Optimize(f, coordinates);

  REQUIRE(abs(coordinates(0)) == Approx(1.34941).margin(0.1));
  REQUIRE(abs(coordinates(1)) == Approx(1.34941).margin(0.1));
}

/**
 * Test the CNE optimizer on the Ackley Function.
 */
TEST_CASE("CNEAckleyFunctionTest", "[CNETest]")
{
  AckleyFunction f;
  CNE optimizer(450, 1500, 0.3, 0.3, 0.3, -1);

  arma::mat coordinates = arma::mat("3; 3");
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0).margin(0.1));
}

/**
 * Test the CNE optimizer on the Beale Function.
 */
TEST_CASE("CNEBealeFunctionTest", "[CNETest]")
{
  BealeFunction f;
  CNE optimizer(450, 1500, 0.3, 0.3, 0.3, -1);

  arma::mat coordinates = arma::mat("3; 3");
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(3).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.5).margin(0.1));
}

/**
 * Test the CNE optimizer on the Goldstein-Price Function.
 */
TEST_CASE("CNEGoldsteinPriceFunctionTest", "[CNETest]")
{
  GoldsteinPriceFunction f;
  CNE optimizer(450, 1500, 0.3, 0.3, 0.1, -1);

  arma::mat coordinates = arma::mat("0.5; -0.5");
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(-1).margin(0.1));
}

/**
 * Test the CNE optimizer on the Levi Function.
 */
TEST_CASE("CNELevyFunctionN13Test", "[CNETest]")
{
  LevyFunctionN13 f;
  CNE optimizer(450, 1500, 0.3, 0.3, 0.02, -1);

  arma::mat coordinates = arma::mat("1.5; 0.5");
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(1).margin(0.1));
  REQUIRE(coordinates(1) == Approx(1).margin(0.1));
}

/**
 * Test the CNE optimizer on the Himmelblau Function.
 */
TEST_CASE("CNEHimmelblauFunctionTest", "[CNETest]")
{
  HimmelblauFunction f;
  CNE optimizer(450, 1500, 0.3, 0.3, 0.3, 1e-7);

  arma::mat coordinates = arma::mat("2; 1");
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(3.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(2.0).margin(0.1));
}

/**
 * Test the CNE optimizer on the Three-hump Camel Function.
 */
TEST_CASE("CNEThreeHumpCamelFunctionTest", "[CNETest]")
{
  ThreeHumpCamelFunction f;
  CNE optimizer(450, 1500, 0.3, 0.3, 0.3, -1);

  arma::mat coordinates = arma::mat("1; 1");
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0).margin(0.1));
}

// TODO: The CNE optimizer with the given parameter occasionally fails to find a
// solution for the Schaffer N4 function, so the function should be tested
// against another optimizer (PSO).
/**
 * Test the CNE optimizer on Schaffer function N.4.
 */
TEST_CASE("CNESchafferFunctionN4Test", "[CNETest]")
{
  SchafferFunctionN4 f;
  CNE optimizer(500, 1600, 0.3, 0.3, 0.3, -1);

  arma::mat coordinates = arma::mat("0.5; 2");
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0).margin(0.1));
  REQUIRE(abs(coordinates(1)) == Approx(1.25313).margin(0.1));
}

/**
 * Test the CNE optimizer on Schaffer Function N.2.
 */
TEST_CASE("CNESchafferFunctionN2Test", "[CNETest]")
{
  SchafferFunctionN2 f;
  CNE optimizer(500, 1600, 0.3, 0.3, 0.3, -1);

  arma::mat coordinates = arma::mat("0.5; -0.5");
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0).margin(0.1));
}
