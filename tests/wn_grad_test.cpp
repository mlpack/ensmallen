/**
 * @file wn_grad_test.cpp
 * @author Marcus Edel
 *
 * Test file for the WNGrad optimizer.
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
 * Run WNGrad on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("WNGradLogisticRegressionTest","[WNGradTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

  WNGrad optimizer(0.56, 1, 500000, 1e-9, true);
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
 * Test the WNGrad optimizer on the Sphere function.
 */
TEST_CASE("WNGradSphereFunctionTest","[WNGradTest]")
{
  SphereFunction f(2);
  WNGrad optimizer(0.56, 2, 500000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
}

/**
 * Test the WNGrad optimizer on the StyblinskiTangFunction.
 */
TEST_CASE("WNGradStyblinskiTangFunctionTest","[WNGradTest]")
{
  StyblinskiTangFunction f(2);
  WNGrad optimizer(0.56, 2, 500000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(-2.9).epsilon(0.01)); // 1% error tolerance.
  REQUIRE(coordinates(1) == Approx(-2.9).epsilon(0.01)); // 1% error tolerance.
}

/**
 * Test the WNGrad optimizer on the StyblinskiTangFunction.  Use arma::fmat.
 */
TEST_CASE("WNGradStyblinskiTangFunctionFMatTest", "[WNGradTest]")
{
  StyblinskiTangFunction f(2);
  WNGrad optimizer(0.56, 2, 500000, 1e-9, true);

  arma::fmat coordinates = f.GetInitialPoint<arma::fmat>();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(-2.9).epsilon(0.01)); // 1% error tolerance.
  REQUIRE(coordinates(1) == Approx(-2.9).epsilon(0.01)); // 1% error tolerance.
}

/**
 * Test the WNGrad optimizer on the StyblinskiTangFunction.  Use arma::sp_mat.
 */
TEST_CASE("WNGradStyblinskiTangFunctionSpMatTest", "[WNGradTest]")
{
  StyblinskiTangFunction f(2);
  WNGrad optimizer(0.56, 2, 500000, 1e-9, true);

  arma::sp_mat coordinates = f.GetInitialPoint<arma::sp_mat>();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(-2.9).epsilon(0.01)); // 1% error tolerance.
  REQUIRE(coordinates(1) == Approx(-2.9).epsilon(0.01)); // 1% error tolerance.
}
