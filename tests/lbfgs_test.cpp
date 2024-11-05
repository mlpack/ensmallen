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

TEMPLATE_TEST_CASE("RosenbrockFunctionTest", "[LBFGS]", arma::mat)
{
  L_BFGS lbfgs;
  lbfgs.MaxIterations() = 10000;

  FunctionTest<RosenbrockFunction, TestType>(lbfgs, 0.01, 0.001);
}

/**
 * Test the L-BFGS optimizer using an arma::mat with the Rosenbrock function and
 * a sparse gradient.
 */
TEST_CASE("RosenbrockFunctionSpGradTest", "[LBFGS]")
{
  RosenbrockFunction f;
  L_BFGS lbfgs;
  lbfgs.MaxIterations() = 10000;

  arma::mat coords = f.GetInitialPoint<arma::vec>();
  lbfgs.Optimize<RosenbrockFunction, arma::mat, arma::sp_mat>(f, coords);

  double finalValue = f.Evaluate(coords);

  REQUIRE(finalValue == Approx(0.0).margin(1e-5));
  REQUIRE(coords(0) == Approx(1.0).epsilon(1e-7));
  REQUIRE(coords(1) == Approx(1.0).epsilon(1e-7));
}

/**
 * Test the L-BFGS optimizer using an arma::sp_mat with the Rosenbrock function.
 */
TEST_CASE("RosenbrockFunctionSpMatTest", "[LBFGS]")
{
  L_BFGS lbfgs;
  lbfgs.MaxIterations() = 10000;
  FunctionTest<RosenbrockFunction, arma::sp_mat>(lbfgs, 0.01, 0.001);
}

TEMPLATE_TEST_CASE("ColvilleFunctionTest", "[LBFGS]", arma::mat)
{
  L_BFGS lbfgs;
  lbfgs.MaxIterations() = 10000;
  FunctionTest<ColvilleFunction, TestType>(lbfgs, 0.01, 0.001);
}

TEMPLATE_TEST_CASE("WoodFunctionTest", "[LBFGS]", arma::mat)
{
  L_BFGS lbfgs;
  FunctionTest<WoodFunction, TestType>(lbfgs, 0.01, 0.001);
}

/**
 * Tests the L-BFGS optimizer using the generalized Rosenbrock function.  This
 * is actually multiple tests, increasing the dimension by powers of 2, from 4
 * dimensions to 1024 dimensions.
 */
TEMPLATE_TEST_CASE("GeneralizedRosenbrockFunctionTest", "[LBFGS]",
    arma::mat, arma::fmat)
{
  for (int i = 2; i < 10; i++)
  {
    // Dimension: powers of 2
    int dim = std::pow(2.0, i);

    GeneralizedRosenbrockFunction<TestType, arma::Row<size_t>> f(dim);
    L_BFGS lbfgs(20);
    lbfgs.MaxIterations() = 10000;

    TestType coords = f.GetInitialPoint();
    lbfgs.Optimize(f, coords);

    double finalValue = f.Evaluate(coords);

    // Test the output to make sure it is correct.
    REQUIRE(finalValue == Approx(0.0).margin(1e-5));
    for (int j = 0; j < dim; j++)
      REQUIRE(coords(j) == Approx(1.0).epsilon(1e-3));
  }
}

TEMPLATE_TEST_CASE("RosenbrockWoodFunctionTest", "[LBFGS]",
    arma::mat)
{
  L_BFGS lbfgs;
  lbfgs.MaxIterations() = 10000;
  FunctionTest<RosenbrockWoodFunction<TestType, arma::Row<size_t>>, TestType>(
      lbfgs, 0.01, 0.001);
}

#ifdef USE_COOT

TEMPLATE_TEST_CASE("RosenbrockFunctionTest", "[LBFGS]", coot::mat, coot::fmat)
{
  L_BFGS lbfgs;
  lbfgs.MaxIterations() = 10000;

  FunctionTest<RosenbrockFunction, TestType>(lbfgs, 0.01, 0.001);
}

// /**
//  * Tests the L-BFGS optimizer using the generalized Rosenbrock function.  This
//  * is actually multiple tests, increasing the dimension by powers of 2, from 4
//  * dimensions to 1024 dimensions.
//  */
// TEMPLATE_TEST_CASE("GeneralizedRosenbrockFunctionTest", "[LBFGS]",
//     coot::mat, coot::fmat)
// {
//   typedef typename TestType::elem_type ElemType;

//   for (int i = 2; i < 10; i++)
//   {
//     // Dimension: powers of 2
//     int dim = std::pow(2.0, i);

//     GeneralizedRosenbrockFunction<TestType, coot::Row<size_t>> f(dim);
//     L_BFGS lbfgs(20);
//     lbfgs.MaxIterations() = 10000;

//     TestType coords = f.GetInitialPoint();
//     lbfgs.Optimize(f, coords);

//     double finalValue = f.Evaluate(coords);

//     // Test the output to make sure it is correct.
//     REQUIRE(finalValue == Approx(0.0).margin(1e-5));
//     for (int j = 0; j < dim; j++)
//       REQUIRE(ElemType(coords(j)) == Approx(1.0).epsilon(1e-3));
//   }
// }

// TEMPLATE_TEST_CASE("WoodFunctionTest", "[LBFGS]", coot::mat)
// {
//   L_BFGS lbfgs;
//   lbfgs.MaxIterations() = 10000;
//   FunctionTest<WoodFunction, TestType>(lbfgs, 0.01, 0.001);
// }

// TEMPLATE_TEST_CASE("RosenbrockWoodFunctionTest", "[LBFGS]",
//     coot::mat)
// {
//   L_BFGS lbfgs;
//   lbfgs.MaxIterations() = 10000;
//   FunctionTest<RosenbrockWoodFunction<TestType, coot::Row<size_t>>, TestType>(
//       lbfgs, 0.01, 0.001);
// }

#endif
