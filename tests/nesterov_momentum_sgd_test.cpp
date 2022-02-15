/**
 * @file nesterov_momentum_sgd_test.cpp
 * @author Sourabh Varshney
 * @author Marcus Edel
 * @author Conrad Sanderson
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <ensmallen.hpp>
#include "catch.hpp"

using namespace ens;
using namespace ens::test;

/**
 * Tests the Nesterov Momentum SGD update policy.
 */
TEST_CASE("NesterovMomentumSGDSpeedUpTestFunction", "[NesterovMomentumSGDTest]")
{
  SGDTestFunction f;
  NesterovMomentumUpdate nesterovMomentumUpdate(0.9);
  NesterovMomentumSGD s(0.0003, 1, 2500000, 1e-9, true, nesterovMomentumUpdate,
      NoDecay(), true, true);

  arma::mat coordinates = f.GetInitialPoint();
  double result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(-1.0).margin(0.01));
  REQUIRE(coordinates(0) == Approx(0.0).margin(3e-3));
  REQUIRE(coordinates(1) == Approx(0.0).margin(1e-6));
  REQUIRE(coordinates(2) == Approx(0.0).margin(1e-6));
}

/**
 * Tests the Nesterov Momentum SGD with Generalized Rosenbrock Test.
 */
TEST_CASE("NesterovMomentumSGDGeneralizedRosenbrockTest", "[NesterovMomentumSGDTest]")
{
  // Loop over several variants.
  for (size_t i = 10; i < 50; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction f(i);
    NesterovMomentumUpdate nesterovMomentumUpdate(0.9);
    NesterovMomentumSGD s(0.0001, 1, 0, 1e-15, true, nesterovMomentumUpdate,
        NoDecay(), true, true);

    arma::mat coordinates = f.GetInitialPoint();
    double result = s.Optimize(f, coordinates);

    REQUIRE(result == Approx(0.0).margin(1e-4));
    for (size_t j = 0; j < i; ++j)
      REQUIRE(coordinates(j) == Approx(1.0).epsilon(0.003));
  }
}

/**
 * Tests the Nesterov Momentum SGD with Generalized Rosenbrock Test.  Uses
 * arma::fmat.
 */
TEST_CASE("NesterovMomentumSGDGeneralizedRosenbrockFMatTest",
          "[NesterovMomentumSGDTest]")
{
  // Loop over several variants.
  for (size_t i = 10; i < 50; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction f(i);
    NesterovMomentumUpdate nesterovMomentumUpdate(0.9);
    NesterovMomentumSGD s(0.00015, 1, 0, 1e-10, true, nesterovMomentumUpdate);

    size_t trial = 0;
    float result = std::numeric_limits<float>::max();
    arma::fmat coordinates;
    while (trial++ < 8 && result > 0.1)
    {
      coordinates = f.GetInitialPoint<arma::fmat>();
      result = s.Optimize(f, coordinates);
    }

    REQUIRE(result == Approx(0.0).margin(0.02));
    for (size_t j = 0; j < i; ++j)
      REQUIRE(coordinates(j) == Approx(1.0).margin(0.05));
  }
}

/**
 * Tests the Nesterov Momentum SGD with Generalized Rosenbrock Test.  Uses
 * arma::sp_mat.
 */
TEST_CASE("NesterovMomentumSGDGeneralizedRosenbrockSpMatTest",
          "[NesterovMomentumSGDTest]")
{
  // Loop over several variants.
  for (size_t i = 10; i < 50; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction f(i);
    NesterovMomentumUpdate nesterovMomentumUpdate(0.9);
    NesterovMomentumSGD s(0.0001, 1, 0, 1e-15, true, nesterovMomentumUpdate);

    arma::sp_mat coordinates = f.GetInitialPoint<arma::sp_mat>();
    double result = s.Optimize(f, coordinates);

    REQUIRE(result == Approx(0.0).margin(1e-4));
    for (size_t j = 0; j < i; ++j)
      REQUIRE(coordinates(j) == Approx(1.0).epsilon(0.003));
  }
}
