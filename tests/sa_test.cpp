/**
 * @file sa_test.cpp
 * @author Zhihao Lou
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

using namespace ens;
using namespace ens::test;

// The Generalized-Rosenbrock function is a simple function to optimize.
TEST_CASE("SAGeneralizedRosenbrockTest","[SATest]")
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
      REQUIRE(coordinates(j) == Approx(1.0).epsilon(0.001));
}

// The Rosenbrock function is a simple function to optimize.
TEST_CASE("SARosenbrockTest", "[SATest]")
{
  ExponentialSchedule schedule;
  // The convergence is very sensitive to the choices of maxMove and initMove.
  SA<> sa(schedule, 1000000, 1000., 1000, 100, 1e-11, 3, 1.5, 0.3, 0.3);
  FunctionTest<RosenbrockFunction>(sa, 0.01, 0.001);
}

// The Rosenbrock function is a simple function to optimize.  Use arma::fmat.
TEST_CASE("SARosenbrockFMatTest", "[SATest]")
{
  ExponentialSchedule schedule;
  // The convergence is very sensitive to the choices of maxMove and initMove.
  SA<> sa(schedule, 1000000, 1000., 1000, 100, 1e-11, 3, 1.5, 0.3, 0.3);
  FunctionTest<RosenbrockFunction, arma::fmat>(sa, 0.1, 0.01);
}

/**
 * The Rastrigrin function, a (not very) simple nonconvex function. It has very
 * many local minima, so finding the true global minimum is difficult.
 */
TEST_CASE("RastrigrinFunctionTest", "[SATest]")
{
  // Simulated annealing isn't guaranteed to converge (except in very specific
  // situations).  If this works 1 of 4 times, I'm fine with that.  All I want
  // to know is that this implementation will escape from local minima.
  ExponentialSchedule schedule;
  // The convergence is very sensitive to the choices of maxMove and initMove.
  // SA<> sa(schedule, 2000000, 100, 50, 1000, 1e-12, 2, 2.0, 0.5, 0.1);
  SA<> sa(schedule, 2000000, 100, 50, 1000, 1e-12, 2, 2.0, 0.5, 0.1);
  FunctionTest<RastriginFunction>(sa, 0.01, 0.001, 4);
}
