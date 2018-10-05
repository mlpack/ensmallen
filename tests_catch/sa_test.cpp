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


// #include <mlpack/core.hpp>
// #include <mlpack/core/optimizers/sa/sa.hpp>
// #include <mlpack/core/optimizers/sa/exponential_schedule.hpp>
// #include <mlpack/core/optimizers/problems/generalized_rosenbrock_function.hpp>
// #include <mlpack/core/optimizers/problems/rosenbrock_function.hpp>
// #include <mlpack/core/optimizers/problems/rastrigin_function.hpp>
// 
// #include <mlpack/core/metrics/ip_metric.hpp>
// #include <mlpack/core/metrics/lmetric.hpp>
// #include <mlpack/core/metrics/mahalanobis_distance.hpp>

// using namespace mlpack;
// using namespace mlpack::optimization;
// using namespace mlpack::optimization::test;
// using namespace mlpack::metric;

// The Generalized-Rosenbrock function is a simple function to optimize.
TEST_CASE("GeneralizedRosenbrockTest","[SATest]")
{
  size_t dim = 10;
  GeneralizedRosenbrockFunction f(dim);

  double iteration = 0;
  double result = DBL_MAX;
  arma::mat coordinates;
  while (result > 1e-6)
  {
    ExponentialSchedule schedule;
    // The convergence is very sensitive to the choices of maxMove and initMove.
    SA<ExponentialSchedule> sa(schedule, 1000000, 1000., 1000, 100, 1e-10, 3,
        1.5, 0.5, 0.3);
    coordinates = f.GetInitialPoint();
    result = sa.Optimize(f, coordinates);
    ++iteration;

    REQUIRE(iteration < 4); // No more than three tries.
  }

  // 0.1% tolerance for each coordinate.
  REQUIRE(result == Approx(0.0).margin(1e-6));
  for (size_t j = 0; j < dim; ++j)
      REQUIRE(coordinates[j] == Approx(1.0).epsilon(0.001));
}

// The Rosenbrock function is a simple function to optimize.
TEST_CASE("RosenbrockTest","[SATest]")
{
  RosenbrockFunction f;
  ExponentialSchedule schedule;
  // The convergence is very sensitive to the choices of maxMove and initMove.
  SA<> sa(schedule, 1000000, 1000., 1000, 100, 1e-11, 3, 1.5, 0.3, 0.3);
  arma::mat coordinates = f.GetInitialPoint();

  const double result = sa.Optimize(f, coordinates);

  REQUIRE(result == Approx(0.0).margin(1e-5));
  REQUIRE(coordinates[0] == Approx(1.0).epsilon(1e-4));
  REQUIRE(coordinates[1] == Approx(1.0).epsilon(1e-4));
}

/**
 * The Rastrigrin function, a (not very) simple nonconvex function. It has very
 * many local minima, so finding the true global minimum is difficult.
 */
TEST_CASE("RastrigrinFunctionTest","[SATest]")
{
  // Simulated annealing isn't guaranteed to converge (except in very specific
  // situations).  If this works 1 of 4 times, I'm fine with that.  All I want
  // to know is that this implementation will escape from local minima.
  size_t successes = 0;

  for (size_t trial = 0; trial < 4; ++trial)
  {
    RastriginFunction f(2);
    ExponentialSchedule schedule;
    // The convergence is very sensitive to the choices of maxMove and initMove.
    // SA<> sa(schedule, 2000000, 100, 50, 1000, 1e-12, 2, 2.0, 0.5, 0.1);
    SA<> sa(schedule, 2000000, 100, 50, 1000, 1e-12, 2, 2.0, 0.5, 0.1);
    arma::mat coordinates = f.GetInitialPoint();

    const double result = sa.Optimize(f, coordinates);

    if ((std::abs(result) < 1e-3) &&
        (std::abs(coordinates[0]) < 1e-3) &&
        (std::abs(coordinates[1]) < 1e-3))
    {
      ++successes;
      break; // No need to continue.
    }
  }

  REQUIRE(successes >= 1);
}
