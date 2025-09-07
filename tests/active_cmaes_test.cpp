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
#include "test_types.hpp"

using namespace ens;
using namespace ens::test;

/**
 * Run Active CMA-ES with the full selection policy on Rosenbrock function and
 * make sure the results are acceptable.
 */
TEMPLATE_TEST_CASE("ActiveCMAESRosenbrockFunctionTest", "[ActiveCMAES]",
    ENS_TEST_TYPES)
{
  BoundaryBoxConstraint<TestType> b(0, 10);
  ActiveCMAES<FullSelection, BoundaryBoxConstraint<TestType>>
      activecmaes(0, b, 1, 0, Tolerances<TestType>::Obj);
  activecmaes.StepSize() = 0.01;
  FunctionTest<RosenbrockFunction, TestType>(activecmaes,
      100 * Tolerances<TestType>::LargeObj,
      10 * Tolerances<TestType>::LargeCoord,
      10);
}

/**
 * Run Active CMA-ES with the random selection policy on Rosenbrock function and
 * make sure the results are acceptable.
 */
TEMPLATE_TEST_CASE("ApproxActiveCMAESRosenbrockFunctionTest", "[ActiveCMAES]",
    ENS_TEST_TYPES)
{
  BoundaryBoxConstraint<TestType> b(0, 10);
  ApproxActiveCMAES<BoundaryBoxConstraint<TestType>>
      activecmaes(2048, b, 1, 0, Tolerances<TestType>::Obj / 100);
  activecmaes.StepSize() = 0.01;
  FunctionTest<RosenbrockFunction, TestType>(activecmaes,
      100 * Tolerances<TestType>::LargeObj,
      10 * Tolerances<TestType>::LargeCoord,
      10);
}

/**
 * Run Active CMA-ES with the random selection and empty transformation policies
 * on  Rosenbrock function and make sure the results are acceptable.
 * Use arma::fmat.
 */
TEMPLATE_TEST_CASE("ApproxActiveCMAESEmptyTransformationLogisticRegressionTest",
    "[ActiveCMAES]", ENS_TEST_TYPES)
{
  ApproxActiveCMAES<EmptyTransformation<TestType>>
      activecmaes(0, EmptyTransformation<TestType>(), 16, 0,
                  10 * Tolerances<TestType>::Obj);

  activecmaes.StepSize() = 0.55;
  LogisticRegressionFunctionTest<TestType>(activecmaes,
      Tolerances<TestType>::LRTrainAcc, Tolerances<TestType>::LRTestAcc, 5);
}
