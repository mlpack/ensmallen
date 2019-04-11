/**
 * @file momentum_sgd_test.cpp
 * @author Ryan Curtin
 * @author Marcus Edel
 * @author Conrad Sanderson
 * @Gaurav Sharma
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

  arma::mat coordinates1 = f1.GetInitialPoint();
  double result1 = s1.Optimize(f1, coordinates1);

  // Result doesn't converge in 2500000 iterations.
  REQUIRE((result1 + 1.0) > 0.05);
  REQUIRE(coordinates1[0] >= 0.015);
  REQUIRE(coordinates1[1] == Approx(0.0).margin(1e-6));
  REQUIRE(coordinates1[2] == Approx(0.0).margin(1e-6));

  REQUIRE(result < result1);

  // Use time decay.
  SGDTestFunction f2;
  TimeDecay timeDecay(1e-12);
  SGD<MomentumUpdate, TimeDecay> s2(0.0003, 1, 25000000, 1e-9, true, momentumUpdate, timeDecay);

  arma::mat coordinates2 = f2.GetInitialPoint();
  double result2 = s2.Optimize(f2, coordinates2);

  REQUIRE(result2 == Approx(-1.0).epsilon(0.0015));
  REQUIRE(coordinates2[0] == Approx(0.0).margin(1e-5));
  REQUIRE(coordinates2[1] == Approx(0.0).margin(1e-6));
  REQUIRE(coordinates2[2] == Approx(0.0).margin(1e-6));

  // Use exponential decay.
  SGDTestFunction f3;
  ExponentialDecay expDecay(0.0003, 1e-7);
  SGD<MomentumUpdate, ExponentialDecay> s3(0.0003, 1, 2500000, 1e-9, true, momentumUpdate, expDecay);

  arma::mat coordinates3 = f3.GetInitialPoint();
  double result3 = s3.Optimize(f3, coordinates3);

  REQUIRE(result3 == Approx(-1.0).epsilon(0.0015));
  REQUIRE(coordinates3[0] == Approx(0.0).margin(1e-4));
  REQUIRE(coordinates3[1] == Approx(0.0).margin(1e-6));
  REQUIRE(coordinates3[2] == Approx(0.0).margin(1e-6));

  // Use drop decay.
  SGDTestFunction f4;
  DropDecay dropDecay(0.0003, 0.0003 / 25000000, 2500000);
  SGD<MomentumUpdate, DropDecay> s4(0.0003, 1, 25000000, 1e-9, true, momentumUpdate, dropDecay);

  arma::mat coordinates4 = f4.GetInitialPoint();
  double result4 = s4.Optimize(f4, coordinates4);

  REQUIRE(result4 == Approx(-1.0).epsilon(0.0015));
  REQUIRE(coordinates4[0] == Approx(0.0).margin(1e-5));
  REQUIRE(coordinates4[1] == Approx(0.0).margin(1e-6));
  REQUIRE(coordinates4[2] == Approx(0.0).margin(1e-6));
  
}

TEST_CASE("MomentumSGDGeneralizedRosenbrockTest", "[MomentumSGDTest]")
{
  // Loop over several variants.
  for (size_t i = 10; i < 50; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction f(i);
    MomentumUpdate momentumUpdate(0.4);
    MomentumSGD s(0.0008, 1, 2500000, 1e-15, true, momentumUpdate);

    arma::mat coordinates = f.GetInitialPoint();
    double result = s.Optimize(f, coordinates);

    REQUIRE(result == Approx(0.0).margin(1e-4));
    for (size_t j = 0; j < i; ++j)
      REQUIRE(coordinates[j] == Approx(1.0).epsilon(1e-5));
  }
}
