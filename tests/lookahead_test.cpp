/**
 * @file lookahead_test.cpp
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
 * Test the Lookahead - Adam optimizer on the Sphere function.
 */
TEST_CASE("LookaheadAdamSphereFunctionTest", "[LookaheadTest]")
{
  SphereFunction f(2);

  Lookahead<> optimizer(0.5, 5, 100000, 1e-5, NoDecay(), false, true);
  optimizer.BaseOptimizer().StepSize() = 0.1;
  optimizer.BaseOptimizer().BatchSize() = 2;
  optimizer.BaseOptimizer().Beta1() = 0.7;
  optimizer.BaseOptimizer().Tolerance() = 1e-15;

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
}

/**
 * Test the Lookahead - Adam optimizer on the SGDTest function.
 */
TEST_CASE("LookaheadAdamSimpleSGDTestFunction", "[LookaheadTest]")
{
  SGDTestFunction f;

  Adam adam(0.001, 1, 0.9, 0.999, 1e-8, 5, 1e-19, false, true);
  Lookahead<Adam> optimizer(adam, 0.5, 5, 100000, 1e-15, NoDecay(),
      false, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(1e-3));
  REQUIRE(coordinates(1) == Approx(0.0).margin(1e-3));
  REQUIRE(coordinates(2) == Approx(0.0).margin(1e-3));
}

/**
 * Test the Lookahead - AdaGrad optimizer on the SphereFunction function.
 */
TEST_CASE("LookaheadAdaGradSphereFunction", "[LookaheadTest]")
{
  SphereFunction f(2);

  AdaGrad adagrad(0.99, 1, 1e-8, 5, 1e-15, true);
  Lookahead<AdaGrad> optimizer(adagrad, 0.5, 5, 5000000, 1e-15, NoDecay(),
      false, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
}

/**
 * Run Lookahead - Adam on logistic regression and make sure the results are
 * acceptable.
 */
TEST_CASE("LookaheadAdamLogisticRegressionTest","[LookaheadTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

  Adam adam(0.001, 32, 0.9, 0.999, 1e-8, 5, 1e-19);
  Lookahead<Adam> optimizer(adam, 0.5, 20, 100000, 1e-15, NoDecay(),
      false, true);

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
 * Test the Lookahead - Adam optimizer on the SGDTest function (float).
 */
TEST_CASE("LookaheadAdamSimpleSGDTestFunctionFloat", "[LookaheadTest]")
{
  SGDTestFunction f;

  Adam adam(0.001, 1, 0.9, 0.999, 1e-8, 5, 1e-19, false, true);
  Lookahead<Adam> optimizer(adam, 0.5, 5, 100000, 1e-15, NoDecay(),
      false, true);

  arma::fmat coordinates = f.GetInitialPoint<arma::fmat>();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(1e-3));
  REQUIRE(coordinates(1) == Approx(0.0).margin(1e-3));
  REQUIRE(coordinates(2) == Approx(0.0).margin(1e-3));
}
