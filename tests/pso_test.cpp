/**
 * @file pso_test.cpp
 * @author Adeel Ahmad
 *
 * Test file for PSO.
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
 * Simple test case for PSO.
 */
TEST_CASE("SimpleTest", "[PSO_TEST]")
{
  PSO optimizer;
  PSOTestFunction f;

  arma::mat iterate;
  iterate << 0.5828 << 0.0654 << 0.8817;

  double result = optimizer.Optimize(f, iterate);

  REQUIRE(result == Approx(0.0).margin(1e-5));
}

/**
 * Test for minimizing the Rosenbrock function.
 */
TEST_CASE("RosenbrockFunctionPSOTest", "[PSO_TEST]")
{
  PSO optimizer;
  RosenbrockFunction f;

  arma::mat iterate;
  iterate << 0.5828 << 0.0654 << 0.8817 << 0.3461 << arma::endr
          << 0.2351 << 0.0034 << 0.1641 << 0.7621;

  double result = optimizer.Optimize(f, iterate);

  REQUIRE(result == Approx(0.0).margin(1e-4));
}

/**
 * Test for minimizing Matyas function.
 */
TEST_CASE("MatyasFunctionPSOTest", "[PSO_TEST]")
{
  PSO optimizer;
  MatyasFunction f;

  arma::mat iterate;
  iterate << 0.5828 << 0.0654 << 0.8817 << 0.3461 << arma::endr
          << 0.2351 << 0.0034 << 0.1641 << 0.7621;

  double result = optimizer.Optimize(f, iterate);

  REQUIRE(result == Approx(0.0).margin(1e-5));
}

/**
 * Test for minimizing the Booth function.
 */
TEST_CASE("BoothFunctionPSOTest", "[PSO_TEST]")
{
  PSO optimizer(60, 0.9, 0.5, 0.3, 200, 1e-3);
  BoothFunction f;

  arma::mat iterate;
  iterate << 1 << 3;

  double result = optimizer.Optimize(f, iterate);

  REQUIRE(result == Approx(0.0).margin(1e-3));
}

/**
 * Test for the McCormick function.
 */
TEST_CASE("McCormickTest", "[PSO_TEST]")
{
  PSO optimizer;
  McCormickFunction f;

  arma::mat iterate;
  iterate << -0.54719 << -1.54719;

  double result = optimizer.Optimize(f, iterate);

  REQUIRE(result == Approx(-1.9132229549).margin(1e-5));
}

/**
 * Test for the Eggholder function.
 */
TEST_CASE("EggholderFunctionPSOTest", "[PSO_TEST]")
{
  PSO optimizer;
  EggholderFunction f;

  arma::mat iterate;
  iterate << 512 << 404.2319;

  double result = optimizer.Optimize(f, iterate);

  REQUIRE(result == Approx(-959.6407).margin(1e-5));
}

/**
 * Test for the Easom function.
 */
TEST_CASE("EasomFunctionPSOTest", "[PSO_TEST]")
{
  PSO optimizer;
  EasomFunction f;

  arma::mat iterate;
  iterate << 3.14 << 3.14;

  double result = optimizer.Optimize(f, iterate);

  REQUIRE(result == Approx(-1).margin(1e-3));
}

/**
 * Test for the Colville function.
 */
TEST_CASE("ColvilleFunctionPSOTest", "[PSO_TEST]")
{
  PSO optimizer;
  ColvilleFunction f;

  arma::mat iterate;
  iterate << 1 << 1 << 1 << 1;

  double result = optimizer.Optimize(f, iterate);

  REQUIRE(result == Approx(0).margin(1e-4));
}

/**
 * Test for the Bukin function.
 */
TEST_CASE("BukinFunctionPSOTest", "[PSO_TEST]")
{
  PSO optimizer;
  BukinFunction f;

  arma::mat iterate;
  iterate << -10 << 1;

  double result = optimizer.Optimize(f, iterate);

  REQUIRE(result == Approx(0.0).margin(1e-4));
}

/**
 * Test for logistic regression.
 */
TEST_CASE("LogisticRegressionPSOTest", "[PSO_TEST]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  ConstrictionPSO pso(40, 0.9, 2.05, 2.05, 20000, 1e-3);
  LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

  arma::mat coordinates = lr.GetInitialPoint();
  pso.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.3)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses, coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.6)); // 0.6% error tolerance.
}
