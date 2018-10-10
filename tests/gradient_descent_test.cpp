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

// #include <ensmallen.hpp>
// #include <ensmallen/problems/problems.hpp>
// 
// using namespace ens;
// using namespace ens::test;

TEST_CASE("SimpleGDTestFunction", "[GradientDescentTest]")
{
  GDTestFunction f;
  GradientDescent s(0.01, 5000000, 1e-9);

  arma::vec coordinates = f.GetInitialPoint();
  double result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(0.0).margin(1e-4));
  REQUIRE(coordinates[0] == Approx(0.0).margin(1e-2));
  REQUIRE(coordinates[1] == Approx(0.0).margin(1e-2));
  REQUIRE(coordinates[2] == Approx(0.0).margin(1e-2));
}

TEST_CASE("RosenbrockTest", "[GradientDescentTest]")
{
  // Create the Rosenbrock function.
  RosenbrockFunction f;

  GradientDescent s(0.001, 0, 1e-15);

  arma::mat coordinates = f.GetInitialPoint();
  double result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(0.0).margin(1e-10));
  for (size_t j = 0; j < 2; ++j)
    REQUIRE(coordinates[j] == Approx(1.0).epsilon(1e-5));
}
