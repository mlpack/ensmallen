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
using namespace std;

/**
 * Train and test a logistic regression function using CNE optimizer
 */
TEST_CASE("CNELogisticRegressionTest", "[CNETest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

  CNE opt(200, 1000, 0.2, 0.2, 0.2, 1e-5);
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
 * Test the CNE optimizer on Cross-in-Tray Function.
 */
TEST_CASE("CNECrossInTrayFunctionTest", "[CNETest]")
{
  CrossInTrayFunction f;
  CNE optimizer(500, 2000, 0.3, 0.3, 0.3, 1e-7);

  //arma::mat coordinates = arma::mat("1; 1");
  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(abs(coordinates[0]) == Approx(1.34941).margin(0.1));
  REQUIRE(abs(coordinates[1]) == Approx(1.34941).margin(0.1));
}

/**
 * Test the CNE optimizer on the Holder-table Function.
 */
TEST_CASE("CNEHolderTableFunctionTest", "[CNETest]")
{
  HolderTableFunction f;
  CNE optimizer(500, 2000, 0.3, 0.3, 0.3, 1e-7);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(abs(coordinates[0]) == Approx(8.05502).margin(0.1));
  REQUIRE(abs(coordinates[1]) == Approx(9.66459).margin(0.1));
}

/**
 * Test the CNE optimizer on Schaffer function N.4.
 */
TEST_CASE("CNESchafferFunctionN4Test", "[CNETest]")
{
  SchafferFunctionN4 f;
  CNE optimizer(500, 2000, 0.3, 0.3, 0.3, 1e-7);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates[0] == Approx(0).margin(0.1));
  REQUIRE(abs(coordinates[1]) == Approx(1.25313).margin(0.1));
}
