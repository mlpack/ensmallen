/**
 * @file ipop_cmaes_test.cpp
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
 * Run IPOP CMA-ES with the full selection policy on Rastrigin function and
 * make sure the results are acceptable.
 */
TEST_CASE("IPOPCMAESRastriginFunctionTest", "[IPOPCMAESTest]")
{
  const size_t numFunctions = 2;

  BoundaryBoxConstraint<> b(-5.12, 5.12);
  CMAES<FullSelection, BoundaryBoxConstraint<>> cmaes(0, b, numFunctions, 0, 1e-12);

  IPOPCMAES<CMAES<FullSelection, BoundaryBoxConstraint<>>>
    ipopcmaes(cmaes, 3, 5);

  RastriginFunction f(numFunctions);
  arma::mat initialPoint = f.template GetInitialPoint<arma::mat>();
  arma::mat expectedResult = f.template GetFinalPoint<arma::mat>();
  
  MultipleTrialOptimizerTest(f, ipopcmaes, initialPoint, expectedResult,
    0.01, f.GetFinalObjective(), 0.01, 1);
}

/**
 * Run IPOP CMA-ES with the full selection policy on Rosenbrock function and
 * make sure the results are acceptable.
 */
TEST_CASE("IPOPCMAESRosenbrockFunctionTest", "[IPOPCMAESTest]")
{
  BoundaryBoxConstraint<> b(0, 2);

  IPOPCMAES<CMAES<FullSelection, BoundaryBoxConstraint<>>>
    ipopcmaes(0, b, 16, 0, 1e-3);

  FunctionTest<RosenbrockFunction>(ipopcmaes, 0.1, 0.1);
}

/**
 * Run IPOP-Active CMA-ES with the full selection policy on Rastrigin function and
 * make sure the results are acceptable.
 */
TEST_CASE("IPOPActiveCMAESchwefelFunctionTest", "[IPOPCMAESTest]")
{

  const size_t numFunctions = 2;

  BoundaryBoxConstraint<> b(-5.12, 5.12);
  ActiveCMAES<FullSelection, BoundaryBoxConstraint<>>
    activecmaes(0, b, numFunctions, 0, 1e-12);

  IPOPCMAES<ActiveCMAES<FullSelection, BoundaryBoxConstraint<>>>
    ipopcmaes(activecmaes, 3, 5);

  RastriginFunction f(numFunctions);
  arma::mat initialPoint = f.template GetInitialPoint<arma::mat>();
  arma::mat expectedResult = f.template GetFinalPoint<arma::mat>();

  MultipleTrialOptimizerTest(f, ipopcmaes, initialPoint, expectedResult,
    0.01, f.GetFinalObjective(), 0.01, 1);
}

/**
 * Run IPOP-Active CMA-ES with the full selection policy on Rosenbrock function and
 * make sure the results are acceptable.  Use arma::fmat.
 */
TEST_CASE("IPOPActiveCMAESRosenbrockFunctionFMatTest", "[IPOPCMAESTest]")
{
  BoundaryBoxConstraint<arma::fmat> b(0, 2);

  ActiveCMAES<FullSelection, BoundaryBoxConstraint<arma::fmat>>
    activecmaes(0, b, 16, 0, 1e-3);

  IPOPCMAES<ActiveCMAES<FullSelection, BoundaryBoxConstraint<arma::fmat>>>
    ipopcmaes(activecmaes);

  FunctionTest<RosenbrockFunction, arma::fmat>(ipopcmaes, 0.1, 0.1);
}
