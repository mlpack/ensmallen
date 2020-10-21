/**
 * @file ada_belief_test.cpp
 * @author Marcus Edel
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
 * Test the AdaBelief optimizer on the Sphere function.
 */
TEST_CASE("AdaBeliefSphereFunctionTest", "[AdaBeliefTest]")
{
  SphereFunction f(2);
  AdaBelief optimizer(0.5, 2, 0.7, 0.999, 1e-12, 500000, 1e-3, false);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
}

/**
 * Test the AdaBelief optimizer on the Sphere function with arma::fmat.
 */
TEST_CASE("AdaBeliefSphereFunctionTestFMat", "[AdaBeliefTest]")
{
  SphereFunction f(2);
  AdaBelief optimizer(0.5, 2, 0.7, 0.999, 1e-12, 500000, 1e-3, false);

  arma::fmat coordinates = f.GetInitialPoint<arma::fmat>();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
}

/**
 * Test the AdaBelief optimizer on the McCormick function.
 */
TEST_CASE("AdaBeliefMcCormickFunctionTest", "[AdaBeliefTest]")
{
  McCormickFunction f;
  AdaBelief optimizer(0.5, 1, 0.7, 0.999, 1e-12, 500000, 1e-5, false);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  // 3% error tolerance.
  REQUIRE(coordinates(0) == Approx(-0.547).epsilon(0.03));
  REQUIRE(coordinates(1) == Approx(-1.547).epsilon(0.03));
}

/**
 * Tests the AdaBelief optimizer using a simple test function.
 */
TEST_CASE("SimpleAdaBeliefTestFunction", "[AdaBeliefTest]")
{
  SGDTestFunction f;
  AdaBelief optimizer(1e-3, 1, 0.9, 0.999, 1e-12, 500000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.3));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.3));
  REQUIRE(coordinates(2) == Approx(0.0).margin(0.3));
}

/**
 * Run AdaBelief on logistic regression and make sure the results are
 * acceptable.
 */
TEST_CASE("AdaBeliefLogisticRegressionTest", "[AdaBeliefTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

  AdaBelief optimizer;
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
 * Run AdaBelief on logistic regression and make sure the results are
 * acceptable, using arma::fmat.
 */
TEST_CASE("AdaBeliefLogisticRegressionFMatTest", "[AdaBeliefTest]")
{
  arma::fmat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<arma::fmat> lr(shuffledData, shuffledResponses, 0.5);

  AdaBelief optimizer;
  arma::fmat coordinates = lr.GetInitialPoint();
  optimizer.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const float acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.03)); // 3% error tolerance.

  const float testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.06)); // 6% error tolerance.
}

