/**
 * @file aug_lagrangian_test.cpp
 * @author Ryan Curtin
 * @author Marcus Edel
 * @author Conrad Sanderson
 *
 * Test of the AugmentedLagrangian class using the test functions defined in
 * aug_lagrangian_test_functions.hpp.
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

/**
 * Tests the Augmented Lagrangian optimizer using the
 * AugmentedLagrangianTestFunction class.
 */
TEMPLATE_TEST_CASE("AugLagrangian_AugLagrangianTestFunction", "[AugLagrangian]",
    ENS_ALL_CPU_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  // The choice of 10 memory slots is arbitrary.
  AugLagrangianTestFunction<TestType> f;
  AugLagrangian aug;

  arma::Col<ElemType> coords = f.GetInitialPoint();

  if (!aug.Optimize(f, coords))
    FAIL("Optimization reported failure.");

  ElemType finalValue = f.Evaluate(coords);

  double objTol = Tolerances<TestType>::Obj;
  double coordTol = Tolerances<TestType>::Coord;

  // Low-precision optimization requires wider tolerances than usual.
  if (sizeof(ElemType) < 4)
  {
    objTol = Tolerances<TestType>::LargeObj;
    coordTol = Tolerances<TestType>::LargeCoord;
  }

  REQUIRE(finalValue == Approx(70.0).epsilon(objTol));
  REQUIRE(coords(0) == Approx(1.0).epsilon(coordTol));
  REQUIRE(coords(1) == Approx(4.0).epsilon(coordTol));
}

/**
 * Tests the Augmented Lagrangian optimizer using the Gockenbach function.
 */
TEMPLATE_TEST_CASE("AugLagrangian_GockenbachFunction", "[AugLagrangian]",
    ENS_ALL_TEST_TYPES, ENS_SPARSE_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  GockenbachFunction f;
  AugLagrangian aug;

  TestType coords = f.GetInitialPoint<TestType>();

  if (!aug.Optimize(f, coords))
    FAIL("Optimization reported failure.");

  ElemType finalValue = f.Evaluate(coords);

  double objTol = 100 * Tolerances<TestType>::Obj;
  double coordTol = 10 * Tolerances<TestType>::Coord;

  // Low-precision optimization requires wider tolerances than usual.
  if (sizeof(ElemType) < 4)
  {
    objTol = Tolerances<TestType>::LargeObj;
    coordTol = 10 * Tolerances<TestType>::LargeCoord;
  }

  // Higher tolerance for smaller values.
  REQUIRE(finalValue == Approx(29.633926).epsilon(objTol));
  REQUIRE(coords(0) == Approx(0.12288178).epsilon(coordTol));
  REQUIRE(coords(1) == Approx(-1.10778185).epsilon(coordTol));
  REQUIRE(coords(2) == Approx(0.015099932).epsilon(coordTol));
}
