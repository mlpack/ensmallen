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
TEMPLATE_TEST_CASE("SA_GeneralizedRosenbrockFunction", "[SA]",
    ENS_ALL_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  size_t dim = 10;
  GeneralizedRosenbrockFunctionType<TestType> f(dim);

  size_t iteration = 0;
  ElemType result = DBL_MAX;
  TestType coordinates;
  while (result > 10 * Tolerances<TestType>::Obj)
  {
    ExponentialSchedule schedule;
    // The convergence is very sensitive to the choices of maxMove and initMove.
    SA<ExponentialSchedule> sa(schedule, 1000000, 1000., 1000, 100,
        Tolerances<TestType>::Obj / 1000, 3, 1.5, 0.5, 0.3);
    coordinates = f.template GetInitialPoint<TestType>();
    result = sa.Optimize(f, coordinates);

    // No more than three tries, or five for low-precision.
    const size_t limit = (sizeof(ElemType) < 4) ? 5 : 3;
    REQUIRE(iteration <= limit);

    ++iteration;
  }

  // Use type-specific tolerances.
  REQUIRE(result == Approx(0.0).margin(10 * Tolerances<TestType>::LargeObj));
  for (size_t j = 0; j < dim; ++j)
  {
    REQUIRE(coordinates(j) ==
        Approx(1.0).epsilon(Tolerances<TestType>::LargeCoord));
  }
}

// The Rosenbrock function is a simple function to optimize.
TEMPLATE_TEST_CASE("SA_RosenbrockFunction", "[SA]", ENS_ALL_TEST_TYPES)
{
  ExponentialSchedule schedule;
  // The convergence is very sensitive to the choices of maxMove and initMove.
  SA<> sa(schedule, 1000000, 1000., 1000, 100, Tolerances<TestType>::Obj, 3,
      1.5, 0.3, 0.3);
  // Large tolerances are necessary due to the sensitivity of convergence.
  FunctionTest<RosenbrockFunction, TestType>(sa,
      10 * Tolerances<TestType>::LargeObj,
      10 * Tolerances<TestType>::LargeCoord,
      5);
}

/**
 * The Rastrigrin function, a (not very) simple nonconvex function. It has very
 * many local minima, so finding the true global minimum is difficult.
 */
TEMPLATE_TEST_CASE("SA_RastrigrinFunction", "[SA]", ENS_ALL_TEST_TYPES)
{
  // Simulated annealing isn't guaranteed to converge (except in very specific
  // situations).  If this works 1 of 4 times, I'm fine with that.  All I want
  // to know is that this implementation will escape from local minima.
  ExponentialSchedule schedule;
  // The convergence is very sensitive to the choices of maxMove and initMove.
  // SA<> sa(schedule, 2000000, 100, 50, 1000, 1e-12, 2, 2.0, 0.5, 0.1);
  SA<> sa(schedule, 2000000, 100, 50, 1000, Tolerances<TestType>::Obj, 2, 2.0,
      0.5, 0.1);
  FunctionTest<RastriginFunction, TestType>(sa, Tolerances<TestType>::LargeObj,
      Tolerances<TestType>::LargeCoord, 4);
}

#ifdef USE_COOT

TEMPLATE_TEST_CASE("SA_RosenbrockFunction", "[SA]", coot::mat, coot::fmat)
{
  ExponentialSchedule schedule;
  // The convergence is very sensitive to the choices of maxMove and initMove.
  SA<> sa(schedule, 1000000, 1000., 1000, 100, 1e-11, 3, 1.5, 0.3, 0.3);
  FunctionTest<RosenbrockFunction, TestType>(sa, 0.01, 0.001);
}

TEMPLATE_TEST_CASE("SA_RastrigrinFunction", "[SA]", coot::mat, coot::fmat)
{
  // Simulated annealing isn't guaranteed to converge (except in very specific
  // situations).  If this works 1 of 4 times, I'm fine with that.  All I want
  // to know is that this implementation will escape from local minima.
  ExponentialSchedule schedule;
  // The convergence is very sensitive to the choices of maxMove and initMove.
  // SA<> sa(schedule, 2000000, 100, 50, 1000, 1e-12, 2, 2.0, 0.5, 0.1);
  SA<> sa(schedule, 2000000, 100, 50, 1000, 1e-12, 2, 2.0, 0.5, 0.1);
  FunctionTest<RastriginFunctionType<
      TestType, coot::Row<size_t> >, TestType>(sa, 0.01, 0.001, 4);
}

#endif
