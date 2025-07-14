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
#include "test_function_tools.hpp"
#include "test_types.hpp"

using namespace ens;
using namespace ens::test;

/**
 * Tests the Quasi Hyperbolic Momentum SGD update policy.
 */
TEMPLATE_TEST_CASE("QHSphereFunction", "[QHMomentumSGD]", ENS_TEST_TYPES,
    ENS_SPARSE_TEST_TYPES)
{
  QHUpdate update(0.4, 0.9);
  QHSGD s(0.0025, 1, 500000, Tolerances<TestType>::Obj / 10, true, update,
      NoDecay(), true, true);
  s.ExactObjective() = true;
  FunctionTest<SphereFunction>(s, 10 * Tolerances<TestType>::LargeObj,
      10 * Tolerances<TestType>::LargeCoord);
}

/**
 * Tests the Quasi hyperbolic SGD with Generalized Rosenbrock Test.
 */
TEMPLATE_TEST_CASE("QHSGDSGDGeneralizedRosenbrockTest", "[QHMomentumSGD]",
    ENS_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  // Loop over several variants.
  for (size_t i = 10; i < 50; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunctionType<TestType, arma::Row<size_t>> f(i);
    QHUpdate update(0.9, 0.99);
    // Tolerance set at -1 to force maximum iterations.
    QHSGD s(0.0005, 1, 2500000, -1.0, true, update, NoDecay(), true, true);

    TestType coordinates = f.GetInitialPoint();
    ElemType result = s.Optimize(f, coordinates);

    REQUIRE(result == Approx(0.0).margin(Tolerances<TestType>::Obj));
    for (size_t j = 0; j < i; ++j)
    {
      REQUIRE(coordinates(j) ==
          Approx(1.0).epsilon(Tolerances<TestType>::Coord));
    }
  }
}
