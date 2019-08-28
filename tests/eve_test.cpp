/**
 * @file eve_test.cpp
 * @author Marcus Edel
 *
 * Test file for the Eve optimizer.
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
 * Test the Eve optimizer on the simple SGD function.
 */
TEST_CASE("EveSGDFunction","[EveTest]")
{
  SGDTestFunction f;
  Eve optimizer(1e-3, 1, 0.9, 0.999, 0.999, 1e-8, 10, 400000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(2) == Approx(0.0).margin(0.1));
}

/**
 * Run Eve on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("EveLogisticRegressionTest","[EveTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

  Eve optimizer(1e-3, 1, 0.9, 0.999, 0.999, 1e-8, 10000, 500000, 1e-9, true);
  arma::mat coordinates = lr.GetInitialPoint();
  optimizer.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}

/**
 * Test the Eve optimizer on the Sphere function.
 */
TEST_CASE("EveSphereFunctionTest","[EveTest]")
{
  SphereFunction f(2);
  Eve optimizer(1e-3, 2, 0.9, 0.999, 0.999, 1e-8, 10000, 500000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
}

/**
 * Test the Eve optimizer on the Styblinski-Tang function.
 */
TEST_CASE("EveStyblinskiTangFunctionTest","[EveTest]")
{
  StyblinskiTangFunction f(2);
  Eve optimizer(1e-3, 2, 0.9, 0.999, 0.999, 1e-8, 10000, 500000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(-2.9).epsilon(0.01));
  REQUIRE(coordinates(1) == Approx(-2.9).epsilon(0.01));
}

/**
 * Test the Eve optimizer on the Styblinski-Tang function using arma::fmat as
 * the objective type.
 */
TEST_CASE("EveStyblinskiTangFunctionFMatTest","[EveTest]")
{
  StyblinskiTangFunction f(2);
  Eve optimizer(1e-3, 2, 0.9, 0.999, 0.999, 1e-8, 10000, 500000, 1e-9, true);

  arma::fmat coordinates = f.GetInitialPoint<arma::fmat>();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(-2.9).epsilon(0.01));
  REQUIRE(coordinates(1) == Approx(-2.9).epsilon(0.01));
}

#if ARMA_VERSION_MAJOR > 9 ||\
    (ARMA_VERSION_MAJOR == 9 && ARMA_VERSION_MINOR >= 400)

/**
 * Test the Eve optimizer on the Styblinski-Tang function, using arma::sp_mat as
 * the objective type.
 */
TEST_CASE("EveStyblinskiTangFunctionSpMatTest","[EveTest]")
{
  StyblinskiTangFunction f(2);
  Eve optimizer(1e-3, 2, 0.9, 0.999, 0.999, 1e-8, 10000, 500000, 1e-9, true);

  arma::sp_mat coordinates = f.GetInitialPoint<arma::sp_mat>();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(-2.9).epsilon(0.01));
  REQUIRE(coordinates(1) == Approx(-2.9).epsilon(0.01));
}

#endif
