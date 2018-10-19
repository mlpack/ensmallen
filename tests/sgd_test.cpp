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

using namespace std;
using namespace arma;
using namespace ens;
using namespace ens::test;

// #include <mlpack/core.hpp>
// #include <mlpack/core/optimizers/sgd/sgd.hpp>
// #include <mlpack/core/optimizers/problems/generalized_rosenbrock_function.hpp>
// #include <mlpack/core/optimizers/problems/sgd_test_function.hpp>
// 
// using namespace mlpack;
// using namespace mlpack::optimization;
// using namespace mlpack::optimization::test;

TEST_CASE("SimpleSGDTestFunction","[SGDTest]")
{
  SGDTestFunction f;
  StandardSGD s(0.0003, 1, 5000000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  double result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(-1.0).epsilon(0.0005));
  REQUIRE(coordinates[0] == Approx(0.0).margin(1e-3));
  REQUIRE(coordinates[1] == Approx(0.0).margin(1e-7));
  REQUIRE(coordinates[2] == Approx(0.0).margin(1e-7));
}

TEST_CASE("GeneralizedRosenbrockTest","[SGDTest]")
{
  // Loop over several variants.
  for (size_t i = 10; i < 50; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction f(i);

    StandardSGD s(0.001, 1, 0, 1e-15, true);

    arma::mat coordinates = f.GetInitialPoint();
    double result = s.Optimize(f, coordinates);

    REQUIRE(result == Approx(0.0).margin(1e-10));
    for (size_t j = 0; j < i; ++j)
      REQUIRE(coordinates[j] == Approx(1.0).epsilon(1e-5));
  }
}
