/**
 * @file spsa_test.cpp
 * @author N Rajiv Vaidyanathan
 * @author Marcus Edel
 *
 * Test file for the SPSA optimizer.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <ensmallen.hpp>
#include "catch.hpp"
#include "test_function_tools.hpp"

using namespace arma;
using namespace ens;
using namespace ens::test;

/**
 * Test the SPSA optimizer on the Sphere function.
 */
TEST_CASE("SPSASphereFunctionTest", "[SPSATest]")
{
  SphereFunction f(2);
  SPSA optimizer(0.1, 0.102, 0.16, 0.3, 100000, 0);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
}

/**
 * Test the SPSA optimizer on the Sphere function using arma::fmat.
 */
TEST_CASE("SPSASphereFunctionFMatTest", "[SPSATest]")
{
  SphereFunction f(2);
  SPSA optimizer(0.1, 0.102, 0.16, 0.3, 100000, 0);

  arma::fmat coordinates = f.GetInitialPoint<arma::fmat>();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0f).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0f).margin(0.1));
}

/**
 * Test the SPSA optimizer on the Sphere function using arma::sp_mat.
 */
TEST_CASE("SPSASphereFunctionSpMatTest", "[SPSATest]")
{
  SphereFunction f(2);
  SPSA optimizer(0.1, 0.102, 0.16, 0.3, 100000, 0);

  arma::sp_mat coordinates = f.GetInitialPoint<arma::sp_mat>();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
}

/**
 * Test the SPSA optimizer on the Matyas function.
 */
TEST_CASE("SPSAMatyasFunctionTest", "[SPSATest]")
{
  MatyasFunction f;
  SPSA optimizer(0.1, 0.102, 0.16, 0.3, 100000, 0);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  // 3% error tolerance.
  REQUIRE((std::trunc(100.0 * coordinates(0)) / 100.0) ==
      Approx(0.0).epsilon(0.003));
  REQUIRE((std::trunc(100.0 * coordinates(1)) / 100.0) ==
      Approx(0.0).epsilon(0.003));
}

/**
 * Run SPSA on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("SPSALogisticRegressionTest", "[SPSATest]")
{
  arma::mat data, testData, shuffledData;
  bool success = false;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  for (size_t trial = 0; trial < 6; ++trial)
  {
    LogisticRegressionTestData(data, testData, shuffledData,
        responses, testResponses, shuffledResponses);
    LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

    SPSA optimizer(0.5, 0.102, 0.002, 0.3, 5000, 1e-8);
    arma::mat coordinates = lr.GetInitialPoint();
    optimizer.Optimize(lr, coordinates);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses, coordinates);
    const double testAcc = lr.ComputeAccuracy(testData, testResponses,
        coordinates);
    if (acc == Approx(100.0).epsilon(0.003) &&
        testAcc == Approx(100.0).epsilon(0.006))
      success = true;

    if (success)
      break;
  }

  REQUIRE(success == true);
}
