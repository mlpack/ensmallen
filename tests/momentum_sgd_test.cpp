/**
 * @file momentum_sgd_test.cpp
 * @author Ryan Curtin
 * @author Marcus Edel
 * @author Conrad Sanderson
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#if defined(ENS_USE_COOT)
  #include <armadillo>
  #include <bandicoot>
#endif
#include <ensmallen.hpp>
#include "catch.hpp"
#include "test_types.hpp"

using namespace ens;
using namespace ens::test;

// NOTE: we don't use low-precision for this test because it is very
// specifically tuned to compare momentum SGD and regular SGD.
TEMPLATE_TEST_CASE("MomentumSGD_SGDTestFunction", "[MomentumSGD]",
    ENS_FULLPREC_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  SGDTestFunction f;
  MomentumUpdate momentumUpdate(0.7);
  MomentumSGD s(0.0003, 1, 2500000, 1e-9, true, momentumUpdate, NoDecay(), true,
      true);

  TestType coordinates = f.GetInitialPoint<TestType>();
  ElemType result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(-1.0).epsilon(10 * Tolerances<TestType>::LargeObj));
  REQUIRE(coordinates(0) ==
      Approx(0.0).margin(Tolerances<TestType>::LargeCoord));
  REQUIRE(coordinates(1) ==
      Approx(0.0).margin(Tolerances<TestType>::LargeCoord));
  REQUIRE(coordinates(2) ==
      Approx(0.0).margin(Tolerances<TestType>::LargeCoord));

  // Compare with SGD with vanilla update.
  SGDTestFunction f1;
  VanillaUpdate vanillaUpdate;
  StandardSGD s1(0.0003, 1, 2500000, 1e-9, true, vanillaUpdate, NoDecay(), true,
      true);

  TestType coordinates1 = f1.GetInitialPoint<TestType>();
  ElemType result1 = s1.Optimize(f1, coordinates1);

  // Result doesn't converge in 2500000 iterations.
  REQUIRE((result1 + 1.0) > 0.05);
  REQUIRE(coordinates1(0) >= 0.015);
  REQUIRE(coordinates1(1) ==
      Approx(0.0).margin(Tolerances<TestType>::LargeCoord));
  REQUIRE(coordinates1(2) ==
      Approx(0.0).margin(Tolerances<TestType>::LargeCoord));

  REQUIRE(result < result1);
}

TEMPLATE_TEST_CASE("MomentumSGD_GeneralizedRosenbrockFunction", "[MomentumSGD]",
    ENS_FULLPREC_TEST_TYPES, ENS_SPARSE_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  // Loop over several variants.
  for (size_t i = 10; i < 50; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction f(i);
    MomentumUpdate momentumUpdate(0.4);
    // Set tolerance to -1 so that maximum iterations is always used for
    // termination.
    MomentumSGD s(0.0008, 1, 2500000, -1.0, true, momentumUpdate, NoDecay(),
        true, true);

    TestType coordinates = f.GetInitialPoint<TestType>();
    ElemType result = s.Optimize(f, coordinates);

    REQUIRE(result == Approx(0.0).margin(Tolerances<TestType>::LargeObj));
    for (size_t j = 0; j < i; ++j)
    {
      REQUIRE(coordinates(j) ==
          Approx(1.0).epsilon(Tolerances<TestType>::LargeCoord));
    }
  }
}

TEMPLATE_TEST_CASE("MomentumSGD_GeneralizedRosenbrockFunctionLoose",
    "[MomentumSGD]", ENS_ALL_CPU_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  // Create the generalized Rosenbrock function.
  GeneralizedRosenbrockFunction f(2);
  MomentumUpdate momentumUpdate(0.2);
  MomentumSGD s(0.0015);
  s.UpdatePolicy() = std::move(momentumUpdate);
  s.Tolerance() = 1e-9;

  TestType coordinates = f.GetInitialPoint<TestType>();
  ElemType result = s.Optimize(f, coordinates);

  // Allow wider tolerances for low-precision types.
  const ElemType factor = (sizeof(ElemType) < 4) ? 5 : 1;
  REQUIRE(result ==
      Approx(0.0).margin(factor * Tolerances<TestType>::LargeObj));

  REQUIRE(coordinates(0) ==
      Approx(1.0).epsilon(factor * Tolerances<TestType>::LargeCoord));
  REQUIRE(coordinates(1) ==
      Approx(1.0).epsilon(factor * Tolerances<TestType>::LargeCoord));
}
