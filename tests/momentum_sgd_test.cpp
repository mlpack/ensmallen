/**
 * @file momentum_sgd_test.cpp
 * @author Ryan Curtin
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

TEST_CASE("MomentumSGDSpeedUpTestFunction", "[MomentumSGDTest]")
{
  SGDTestFunction f;
  MomentumUpdate momentumUpdate(0.7);
  MomentumSGD s(0.0003, 1, 2500000, 1e-9, true, momentumUpdate, NoDecay(), true,
      true);

  arma::mat coordinates = f.GetInitialPoint();
  double result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(-1.0).epsilon(0.0015));
  REQUIRE(coordinates(0) == Approx(0.0).margin(0.015));
  REQUIRE(coordinates(1) == Approx(0.0).margin(1e-6));
  REQUIRE(coordinates(2) == Approx(0.0).margin(1e-6));

  // Compare with SGD with vanilla update.
  SGDTestFunction f1;
  VanillaUpdate vanillaUpdate;
  StandardSGD s1(0.0003, 1, 2500000, 1e-9, true, vanillaUpdate, NoDecay(), true,
      true);

  arma::mat coordinates1 = f1.GetInitialPoint();
  double result1 = s1.Optimize(f1, coordinates1);

  // Result doesn't converge in 2500000 iterations.
  REQUIRE((result1 + 1.0) > 0.05);
  REQUIRE(coordinates1(0) >= 0.015);
  REQUIRE(coordinates1(1) == Approx(0.0).margin(1e-6));
  REQUIRE(coordinates1(2) == Approx(0.0).margin(1e-6));

  REQUIRE(result < result1);
}

TEST_CASE("MomentumSGDGeneralizedRosenbrockTest", "[MomentumSGDTest]")
{
  // Loop over several variants.
  for (size_t i = 10; i < 50; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction f(i);
    MomentumUpdate momentumUpdate(0.4);
    MomentumSGD s(0.0008, 1, 2500000, 1e-15, true, momentumUpdate, NoDecay(),
        true, true);

    arma::mat coordinates = f.GetInitialPoint();
    double result = s.Optimize(f, coordinates);

    REQUIRE(result == Approx(0.0).margin(1e-4));
    for (size_t j = 0; j < i; ++j)
      REQUIRE(coordinates(j) == Approx(1.0).epsilon(1e-5));
  }
}

// Use arma::fmat.
TEST_CASE("MomentumSGDGeneralizedRosenbrockFMatTest", "[MomentumSGDTest]")
{
  // Loop over several variants.
  for (size_t i = 10; i < 50; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction f(i);
    MomentumUpdate momentumUpdate(0.1);
    MomentumSGD s(0.0002, 1, 10000000, 1e-15, true, momentumUpdate);

    arma::fmat coordinates = f.GetInitialPoint<arma::fmat>();
    float result = s.Optimize(f, coordinates);

    REQUIRE(result == Approx(0.0).margin(1e-2));
    for (size_t j = 0; j < i; ++j)
      REQUIRE(coordinates(j) == Approx(1.0).epsilon(1e-3));
  }
}

// Use arma::sp_mat.
TEST_CASE("MomentumSGDGeneralizedRosenbrockSpMatTest", "[MomentumSGDTest]")
{
  // Loop over several variants.
  for (size_t i = 10; i < 50; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction f(i);
    MomentumUpdate momentumUpdate(0.4);
    MomentumSGD s(0.0008, 1, 2500000, 1e-15, true, momentumUpdate);

    arma::sp_mat coordinates = f.GetInitialPoint<arma::sp_mat>();
    double result = s.Optimize(f, coordinates);

    REQUIRE(result == Approx(0.0).margin(1e-4));
    for (size_t j = 0; j < i; ++j)
      REQUIRE(coordinates(j) == Approx(1.0).epsilon(1e-5));
  }
}
