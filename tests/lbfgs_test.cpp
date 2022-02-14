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

/**
 * Tests the L-BFGS optimizer using the Rosenbrock Function.
 */
TEST_CASE("RosenbrockFunctionTest", "[LBFGSTest]")
{
  L_BFGS lbfgs;
  lbfgs.MaxIterations() = 10000;
  FunctionTest<RosenbrockFunction>(lbfgs, 0.01, 0.001);
}

/**
 * Test the L-BFGS optimizer using an arma::fmat with the Rosenbrock function.
 */
TEST_CASE("RosenbrockFunctionFloatTest", "[LBFGSTest]")
{
  L_BFGS lbfgs;
  lbfgs.MaxIterations() = 10000;
  FunctionTest<RosenbrockFunction, arma::fmat>(lbfgs, 0.1, 0.01);
}

/**
 * Test the L-BFGS optimizer using an arma::mat with the Rosenbrock function and
 * a sparse gradient.
 */
TEST_CASE("RosenbrockFunctionSpGradTest", "[LBFGSTest]")
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
TEST_CASE("RosenbrockFunctionSpMatTest", "[LBFGSTest]")
{
  L_BFGS lbfgs;
  lbfgs.MaxIterations() = 10000;
  FunctionTest<RosenbrockFunction, arma::sp_mat>(lbfgs, 0.01, 0.001);
}

/**
 * Tests the L-BFGS optimizer using the Colville Function.
 */
TEST_CASE("ColvilleFunctionTest", "[LBFGSTest]")
{
  L_BFGS lbfgs;
  lbfgs.MaxIterations() = 10000;
  FunctionTest<ColvilleFunction>(lbfgs, 0.01, 0.001);
}

/**
 * Tests the L-BFGS optimizer using the Wood Function.
 */
TEST_CASE("WoodFunctionTest", "[LBFGSTest]")
{
  L_BFGS lbfgs;
  lbfgs.MaxIterations() = 10000;
  FunctionTest<WoodFunction>(lbfgs, 0.01, 0.001);
}

/**
 * Tests the L-BFGS optimizer using the generalized Rosenbrock function.  This
 * is actually multiple tests, increasing the dimension by powers of 2, from 4
 * dimensions to 1024 dimensions.
 */
TEST_CASE("GeneralizedRosenbrockFunctionTest", "[LBFGSTest]")
{
  for (int i = 2; i < 10; i++)
  {
    // Dimension: powers of 2
    int dim = std::pow(2.0, i);

    GeneralizedRosenbrockFunction f(dim);
    L_BFGS lbfgs(20);
    lbfgs.MaxIterations() = 10000;

    arma::vec coords = f.GetInitialPoint();
    lbfgs.Optimize(f, coords);

    double finalValue = f.Evaluate(coords);

    // Test the output to make sure it is correct.
    REQUIRE(finalValue == Approx(0.0).margin(1e-5));
    for (int j = 0; j < dim; j++)
      REQUIRE(coords(j) == Approx(1.0).epsilon(1e-7));
  }
}

/**
 * Tests the L-BFGS optimizer using the Rosenbrock-Wood combined function.  This
 * is a test on optimizing a matrix of coordinates.
 */
TEST_CASE("RosenbrockWoodFunctionTest", "[LBFGSTest]")
{
  L_BFGS lbfgs;
  lbfgs.MaxIterations() = 10000;
  FunctionTest<RosenbrockWoodFunction>(lbfgs, 0.01, 0.001);
}
