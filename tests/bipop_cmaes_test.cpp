/**
 * @file bipop_cmaes_test.cpp
 * @author Benjami Parellada
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
 * Run BIPOP CMA-ES with the full selection policy on Rastrigin function and
 * make sure the results are acceptable.
 */
TEST_CASE("BIPOPCMAESRastriginFunctionTest", "[BIPOPCMAESTest]")
{
  const size_t numFunctions = 2;
  BoundaryBoxConstraint<> b(-5.12, 5.12);
  CMAES<FullSelection, BoundaryBoxConstraint<>> cmaes(10, b, numFunctions, 0, 1e-5);
  cmaes.StepSize() = 3.72;
  BIPOPCMAES<CMAES<FullSelection, BoundaryBoxConstraint<>>> bipopcmaes(cmaes, 10);
  
  RastriginFunction f(numFunctions);
  arma::mat initialPoint = f.template GetInitialPoint<arma::mat>();
  arma::mat expectedResult = f.template GetFinalPoint<arma::mat>();
  
  MultipleTrialOptimizerTest(f, bipopcmaes, initialPoint, expectedResult,
                             0.01, f.GetFinalObjective(), 0.1, 5);
}

/**
 * Run BIPOP CMA-ES with the full selection policy on Rosenbrock function and
 * make sure the results are acceptable.
 */
TEST_CASE("BIPOPCMAESRosenbrockFunctionTest", "[BIPOPCMAESTest]")
{
  BoundaryBoxConstraint<> b(0, 2);
  BIPOPCMAES<CMAES<FullSelection, BoundaryBoxConstraint<>>> bipopcmaes(0, b, 16, 0, 1e-3);
  bipopcmaes.CMAES().StepSize() = 0.25;
  
  FunctionTest<RosenbrockFunction>(bipopcmaes, 0.1, 0.1);
}
