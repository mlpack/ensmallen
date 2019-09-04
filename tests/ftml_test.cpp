/**
 * @file ftml_test.cpp
 * @author Ryan Curtin
 *
 * Test file for the FTML optimizer.
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
 * Test the FTML optimizer on the simple SGD function.
 */
TEST_CASE("FTMLSGDFunction", "[FTMLTest]")
{
  SGDTestFunction f;
  FTML optimizer(0.005, 1, 0.9, 0.999, 1e-8, 1000000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(2) == Approx(0.0).margin(0.1));
}

/**
 * Run FTML on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("FTMLLogisticRegressionTest", "[FTMLTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

  FTML optimizer(0.001, 1, 0.9, 0.999, 1e-8, 100000, 1e-5, true);
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
 * Test the FTML optimizer on the Sphere function.
 */
TEST_CASE("FTMLSphereFunctionTest", "[FTMLTest]")
{
  SphereFunction f(2);
  FTML optimizer(0.001, 2, 0.9, 0.999, 1e-8, 500000, 1e-9, true);
  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
}

/**
 * Test the FTML optimizer on the Styblinski-Tang function.
 */
TEST_CASE("FTMLStyblinskiTangFunctionTest", "[FTMLTest]")
{
  StyblinskiTangFunction f(2);
  FTML optimizer(0.001, 2, 0.9, 0.999, 1e-8, 100000, 1e-5, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(-2.9).epsilon(0.01));
  REQUIRE(coordinates(1) == Approx(-2.9).epsilon(0.01));
}

/**
 * Test the FTML optimizer on the Styblinski-Tang function using arma::fmat as
 * the objective type.
 */
TEST_CASE("FTMLStyblinskiTangFunctionFMatTest", "[FTMLTest]")
{
  StyblinskiTangFunction f(2);
  FTML optimizer(0.001, 2, 0.9, 0.999, 1e-8, 100000, 1e-5, true);

  arma::fmat coordinates = f.GetInitialPoint<arma::fmat>();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(-2.9).epsilon(0.01));
  REQUIRE(coordinates(1) == Approx(-2.9).epsilon(0.01));
}

// A test with sp_mat is not done, because FTML uses some parts internally that
// assume the objective is dense.
