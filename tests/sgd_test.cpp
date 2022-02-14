/**
 * @file sgd_test.cpp
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
#include "test_function_tools.hpp"

using namespace std;
using namespace arma;
using namespace ens;
using namespace ens::test;

TEST_CASE("GeneralizedRosenbrockTest", "[SGDTest]")
{
  // Loop over several variants.
  for (size_t i = 10; i < 50; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction f(i);

    VanillaUpdate vanillaUpdate;
    StandardSGD s(0.001, 1, 0, 1e-15, true, vanillaUpdate, NoDecay(), true,
        true);

    arma::mat coordinates = f.GetInitialPoint();
    double result = s.Optimize(f, coordinates);

    REQUIRE(result == Approx(0.0).margin(1e-10));
    for (size_t j = 0; j < i; ++j)
      REQUIRE(coordinates(j) == Approx(1.0).epsilon(1e-5));
  }
}

TEST_CASE("GeneralizedRosenbrockTestFloat", "[SGDTest]")
{
  // Loop over several variants.
  for (size_t i = 10; i < 50; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction f(i);

    // Allow a few trials.
    for (size_t trial = 0; trial < 5; ++trial)
    {
      StandardSGD s(0.001, 1, 0, 1e-15, true);

      arma::fmat coordinates = f.GetInitialPoint<arma::fmat>();
      float result = s.Optimize(f, coordinates);

      if (trial != 4)
      {
        if (result != Approx(0.0).margin(1e-5))
          continue;
        for (size_t j = 0; j < i; ++j)
        {
          if (coordinates(j) != Approx(1.0).epsilon(1e-3))
            continue;
        }
      }

      REQUIRE(result == Approx(0.0).margin(1e-5));
      for (size_t j = 0; j < i; ++j)
        REQUIRE(coordinates(j) == Approx(1.0).epsilon(1e-3));
      break;
    }
  }
}
