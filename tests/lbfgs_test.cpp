/**
 * @file lbfgs_test.cpp
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

using namespace ens;
using namespace ens::test;

// NOTE: L-BFGS depends on the squared gradient norm and the squared norm of
// different of gradients between iterations, and this can be much too large to
// be represented by FP16.  So, we have only one FP16 test case.

TEMPLATE_TEST_CASE("LBFGS_RosenbrockFunction", "[LBFGS]", ENS_TEST_TYPES,
    ENS_SPARSE_TEST_TYPES)
{
  L_BFGS lbfgs;
  lbfgs.MaxIterations() = 10000;

  FunctionTest<RosenbrockFunction, TestType>(lbfgs,
      Tolerances<TestType>::LargeObj,
      Tolerances<TestType>::LargeCoord);
}

/**
 * Test the L-BFGS optimizer using an arma::mat with the Rosenbrock function and
 * a sparse gradient.
 */
TEMPLATE_TEST_CASE("LBFGS_RosenbrockGradFunction", "[LBFGS]",
    ENS_SPARSE_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  RosenbrockFunction f;
  L_BFGS lbfgs;
  lbfgs.MaxIterations() = 10000;

  arma::Mat<ElemType> coords = f.GetInitialPoint<arma::Col<ElemType> >();
  lbfgs.Optimize<RosenbrockFunction, arma::Mat<ElemType>, TestType>(f, coords);

  ElemType finalValue = f.Evaluate(coords);

  REQUIRE(finalValue == Approx(0.0).margin(Tolerances<TestType>::Obj));
  REQUIRE(coords(0) == Approx(1.0).epsilon(Tolerances<TestType>::Coord));
  REQUIRE(coords(1) == Approx(1.0).epsilon(Tolerances<TestType>::Coord));
}

TEMPLATE_TEST_CASE("LBFGS_ColvilleFunction", "[LBFGS]", ENS_TEST_TYPES)
{
  L_BFGS lbfgs;
  lbfgs.MaxIterations() = 10000;
  FunctionTest<ColvilleFunction, TestType>(lbfgs,
      Tolerances<TestType>::LargeObj,
      Tolerances<TestType>::LargeCoord);
}

TEMPLATE_TEST_CASE("LBFGS_WoodFunction", "[LBFGS]", ENS_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  L_BFGS lbfgs;
  // Special tolerances: L-BFGS with floats will converge too early.
  const double tol = std::is_same<ElemType, float>::value ? 20.0 : 1e-8;
  FunctionTest<WoodFunction, TestType>(lbfgs, ElemType(tol),
      ElemType(tol / 10));
}

/**
 * Tests the L-BFGS optimizer using the generalized Rosenbrock function.  This
 * is actually multiple tests, increasing the dimension by powers of 2, from 4
 * dimensions to 1024 dimensions.
 */
TEMPLATE_TEST_CASE("LBFGS_GeneralizedRosenbrockFunction", "[LBFGS]",
    ENS_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  for (size_t i = 2; i < 10; i++)
  {
    // Dimension: powers of 2
    size_t dim = std::pow(2.0, i);

    GeneralizedRosenbrockFunctionType<TestType, arma::Row<size_t>> f(dim);
    L_BFGS lbfgs(20);
    lbfgs.MaxIterations() = 10000;

    TestType coords = f.GetInitialPoint();
    lbfgs.Optimize(f, coords);

    ElemType finalValue = f.Evaluate(coords);

    // Test the output to make sure it is correct.
    REQUIRE(finalValue == Approx(0.0).margin(Tolerances<TestType>::Obj));
    for (size_t j = 0; j < dim; j++)
      REQUIRE(coords(j) == Approx(1.0).epsilon(Tolerances<TestType>::Coord));
  }
}

// This test will work with all test types (including FP16), but we leave the
// tolerances quite loose.
TEMPLATE_TEST_CASE("LBFGS_GeneralizedRosenbrockFunctionLoose", "[LBFGS]",
    ENS_ALL_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  GeneralizedRosenbrockFunctionType<TestType, arma::Row<size_t>> f(2);
  L_BFGS lbfgs(20);
  lbfgs.MaxIterations() = 1000;
  // For FP16, to keep the gradient different norm small enough, we must limit
  // the step size.
  if (sizeof(ElemType) < 4)
    lbfgs.MaxStep() = 0.15;

  TestType coords = f.GetInitialPoint();
  lbfgs.Optimize(f, coords);

  ElemType finalValue = f.Evaluate(coords);

  // Test the output to make sure it is correct.
  REQUIRE(finalValue ==
      Approx(0.0).margin(50 * Tolerances<TestType>::LargeObj));
  REQUIRE(coords(0) ==
        Approx(1.0).margin(50 * Tolerances<TestType>::LargeCoord));
  REQUIRE(coords(1) ==
        Approx(1.0).margin(50 * Tolerances<TestType>::LargeCoord));
}

TEMPLATE_TEST_CASE("LBFGS_RosenbrockWoodFunction", "[LBFGS]", ENS_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  L_BFGS lbfgs;
  lbfgs.MaxIterations() = 10000;
  // Special tolerances: L-BFGS with floats will converge too early.
  const double tol = std::is_same<ElemType, float>::value ? 20.0 : 1e-8;
  FunctionTest<RosenbrockWoodFunctionType<TestType, arma::Row<size_t>>,
               TestType>(lbfgs, ElemType(tol), ElemType(tol / 10));
}

#ifdef USE_COOT

TEMPLATE_TEST_CASE("LBFGS_RosenbrockFunction", "[LBFGS]",
    coot::mat, coot::fmat)
{
  L_BFGS lbfgs;
  lbfgs.MaxIterations() = 10000;

  FunctionTest<RosenbrockFunction, TestType>(lbfgs, 0.01, 0.001);
}

 /**
  * Tests the L-BFGS optimizer using the generalized Rosenbrock function.  This
  * is actually multiple tests, increasing the dimension by powers of 2, from 4
  * dimensions to 1024 dimensions.
  */
TEMPLATE_TEST_CASE("LBFGS_GeneralizedRosenbrockFunction", "[LBFGS]",
    coot::mat, coot::fmat)
{
  typedef typename TestType::elem_type ElemType;

  for (int i = 2; i < 10; i++)
  {
    // Dimension: powers of 2
    int dim = std::pow(2.0, i);

    GeneralizedRosenbrockFunctionType<TestType, coot::Row<size_t>> f(dim);
    L_BFGS lbfgs(20);
    lbfgs.MaxIterations() = 10000;

    TestType coords = f. template GetInitialPoint<TestType>();
    lbfgs.Optimize(f, coords);

    double finalValue = f.Evaluate(coords);

    // Test the output to make sure it is correct.
    REQUIRE(finalValue == Approx(0.0).margin(1e-5));
    for (int j = 0; j < dim; j++)
      REQUIRE(ElemType(coords(j)) == Approx(1.0).epsilon(1e-3));
  }
}

TEMPLATE_TEST_CASE("LBFGS_WoodFunction", "[LBFGS]", coot::mat)
{
  L_BFGS lbfgs;
  lbfgs.MaxIterations() = 10000;
  FunctionTest<WoodFunction, TestType>(lbfgs, 0.01, 0.001);
}

TEMPLATE_TEST_CASE("LBFGS_RosenbrockWoodFunction", "[LBFGS]",
    coot::mat)
{
  L_BFGS lbfgs;
  lbfgs.MaxIterations() = 10000;
  FunctionTest<RosenbrockWoodFunctionType<
      TestType, coot::Row<size_t>>, TestType>(lbfgs, 0.01, 0.001);
}

#endif
