/**
 * @file active_cmaes_test.cpp
 * @author Suvarsha Chennareddy
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
 * Run Active CMA-ES with the full selection policy on Rosenbrock function and
 * make sure the results are acceptable.
 */
TEST_CASE("ActiveCMAESRosenbrockFunctionTest", "[ActiveCMAESTest]")
{
  BoundaryBoxConstraint<> b(0, 10);
  ActiveCMAES<FullSelection, BoundaryBoxConstraint<>> 
      activecmaes(0, b, 1, 0, 1e-8);
  activecmaes.StepSize() = 0.01;
  FunctionTest<RosenbrockFunction>(activecmaes, 0.1, 0.1, 10);
}

/**
 * Run Active CMA-ES with the random selection policy on Rosenbrock function and
 * make sure the results are acceptable.
 */
TEST_CASE("ApproxActiveCMAESRosenbrockFunctionTest", "[ActiveCMAESTest]")
{
  BoundaryBoxConstraint<> b(0, 10);
  ApproxActiveCMAES<BoundaryBoxConstraint<arma::mat>> 
      activecmaes(2048, b, 1, 0, 1e-13);
  activecmaes.StepSize() = 0.01;
  FunctionTest<RosenbrockFunction>(activecmaes, 0.1, 0.1, 10);
}

/**
 * Run Active CMA-ES with the full selection policy on  Rosenbrock function and
 * make sure the results are acceptable.  Use arma::fmat.
 */
TEST_CASE("ActiveCMAESRosenbrockFunctionFMatTest", "[ActiveCMAESTest]")
{
  BoundaryBoxConstraint<arma::fmat> b(0, 10);
  ActiveCMAES<FullSelection, BoundaryBoxConstraint<arma::fmat>> 
      activecmaes(0, b, 1, 0, 1e-8);
  activecmaes.StepSize() = 0.01;
  FunctionTest<RosenbrockFunction, arma::fmat>(activecmaes, 0.1, 0.1, 5);
}

/**
 * Run Active CMA-ES with the random selection policy on  Rosenbrock function and
 * make sure the results are acceptable.  Use arma::fmat.
 */
TEST_CASE("ApproxActiveCMAESRosenbrockFunctionFMatTest", "[ActiveCMAESTest]")
{
  BoundaryBoxConstraint<arma::fmat> b(0, 10);
  ApproxActiveCMAES<BoundaryBoxConstraint<arma::fmat>> 
      activecmaes(2048, b, 1, 0, 1e-5);
  activecmaes.StepSize() = 0.01;
  FunctionTest<RosenbrockFunction, arma::fmat>(activecmaes, 0.1, 0.1, 10);
}

/**
 * Run Active CMA-ES with the random selection and empty transformation policies
 * on  Rosenbrock function and make sure the results are acceptable.
 * Use arma::fmat.
 */
TEST_CASE("ApproxActiveCMAESEmptyTransformationLogisticRegressionFMatTest",
  "[ActiveCMAESTest]")
{
  ApproxActiveCMAES<EmptyTransformation<arma::fmat>>
      activecmaes(0, EmptyTransformation<arma::fmat>(), 16, 0, 1e-3);
  activecmaes.StepSize() = 0.55;
  LogisticRegressionFunctionTest<arma::fmat>(activecmaes, 0.01, 0.02, 5);
}
