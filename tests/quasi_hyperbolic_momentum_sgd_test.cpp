/**
 * @file quasi_hyperbolic_momentum_sgd_test.cpp
 * @author Niteya Shah
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
/*
* Tests the Quasi Hyperbolic Momentum SGD update policy.
*/
TEST_CASE("QHSGDTestFunction", "[QHMomentumSGDTest]")
{
  SGDTestFunction f;
  QHUpdate update(0.9, 0.9);
  QHSGD s(0.005, 1, 2500000, 1e-9, true, update);

  arma::mat coordinates = f.GetInitialPoint();
  double result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(-1.0).epsilon(0.0025));
  REQUIRE(coordinates[0] == Approx(0.0).margin(3e-3));
  REQUIRE(coordinates[1] == Approx(0.0).margin(1e-6));
  REQUIRE(coordinates[2] == Approx(0.0).margin(1e-6));
}

/*
* Tests the Quasi hyperbolic SGD with Generalized Rosenbrock Test.
*/
TEST_CASE("QHSGDSGDGeneralizedRosenbrockTest", "[QHMomentumSGDTest]")
{  // Loop over several variants.
  for (size_t i = 10; i < 50; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction f(i);
    QHUpdate update(0.99, 0.999);
    QHSGD s(0.0005, 1, 2500000, 1e-15, true, update);

    arma::mat coordinates = f.GetInitialPoint();
    double result = s.Optimize(f, coordinates);

    REQUIRE(result == Approx(0.0).margin(1e-4));
    for (size_t j = 0; j < i; ++j)
      REQUIRE(coordinates[j] == Approx(1.0).epsilon(1e-4));
  }
}
