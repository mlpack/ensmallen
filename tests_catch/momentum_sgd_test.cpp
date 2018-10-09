// Copyright (c) 2018 ensmallen developers.
// 
// Licensed under the 3-clause BSD license (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.opensource.org/licenses/BSD-3-Clause

#include <ensmallen.hpp>
#include "catch.hpp"

using namespace std;
using namespace arma;
using namespace ens;
using namespace ens::test;

// #include <mlpack/core.hpp>
// #include <mlpack/core/optimizers/sgd/sgd.hpp>
// #include <mlpack/core/optimizers/sgd/update_policies/gradient_clipping.hpp>
// #include <mlpack/core/optimizers/sgd/update_policies/momentum_update.hpp>
// #include <mlpack/core/optimizers/problems/generalized_rosenbrock_function.hpp>
// #include <mlpack/core/optimizers/problems/sgd_test_function.hpp>
// 
// using namespace mlpack;
// using namespace mlpack::optimization;
// using namespace mlpack::optimization::test;

TEST_CASE("MomentumSGDSpeedUpTestFunction", "[MomentumSGDTest]")
{
  SGDTestFunction f;
  MomentumUpdate momentumUpdate(0.7);
  MomentumSGD s(0.0003, 1, 2500000, 1e-9, true, momentumUpdate);

  arma::mat coordinates = f.GetInitialPoint();
  double result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(-1.0).epsilon(0.0015));
  REQUIRE(coordinates[0] == Approx(0.0).margin(0.015));
  REQUIRE(coordinates[1] == Approx(0.0).margin(1e-6));
  REQUIRE(coordinates[2] == Approx(0.0).margin(1e-6));

  // Compare with SGD with vanilla update.
  SGDTestFunction f1;
  StandardSGD s1(0.0003, 1, 2500000, 1e-9, true);

  arma::mat coordinates1 = f.GetInitialPoint();
  double result1 = s1.Optimize(f1, coordinates1);

  // Result doesn't converge in 2500000 iterations.
  REQUIRE((result1 + 1.0) > 0.05);
  REQUIRE(coordinates1[0] >= 0.015);
  REQUIRE(coordinates1[1] == Approx(0.0).margin(1e-6));
  REQUIRE(coordinates1[2] == Approx(0.0).margin(1e-6));

  REQUIRE(result < result1);
}

TEST_CASE("GeneralizedRosenbrockTest", "[MomentumSGDTest]")
{
  // Loop over several variants.
  for (size_t i = 10; i < 50; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction f(i);
    MomentumUpdate momentumUpdate(0.4);
    MomentumSGD s(0.0008, 1, 0, 1e-15, true, momentumUpdate);

    arma::mat coordinates = f.GetInitialPoint();
    double result = s.Optimize(f, coordinates);

    REQUIRE(result == Approx(0.0).margin(1e-4));
    for (size_t j = 0; j < i; ++j)
      REQUIRE(coordinates[j] == Approx(1.0).epsilon(1e-5));
  }
}
