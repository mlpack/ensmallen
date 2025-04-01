/**
 * @file pop_cmaes_test.cpp
 * @author Benjami Parellada
 *
 * Tests for the POP_CMAES class, including IPOP CMA-ES and BIPOP CMA-ES.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <ensmallen.hpp>
#include "catch.hpp"
#include "test_function_tools.hpp"
#include <ensmallen_bits/cmaes/pop_cmaes.hpp>

using namespace ens;
using namespace ens::test;

/**
 * Run IPOP-CMA-ES on the Rastrigin function and check whether the optimizer
 * converges to the expected solution within tolerance limits.
 */
TEST_CASE("IPOPCMAESRastriginFunctionTest", "[POPCMAESTest]")
{
  RastriginFunction f(2);
  BoundaryBoxConstraint<> b(-10, 10);

  IPOP_CMAES<FullSelection, BoundaryBoxConstraint<>> ipopcmaes(
    15, // lambda
    b, // transformationPolicy
    32, // batchSize
    10000, // maxIterations
    1e-8, // tolerance
    FullSelection(), // selectionPolicy
    3.72, // stepSize
    2.0, // populationFactor
    7, // maxRestarts
    1e6 // maxFunctionEvaluations
  );

  arma::mat initialPoint = f.GetInitialPoint();
  arma::mat expectedResult = f.GetFinalPoint();

  MultipleTrialOptimizerTest(f, ipopcmaes, initialPoint, expectedResult, 0.5, f.GetFinalObjective(), 0.5, 5);
}

/**
 * Run BIPOP-CMA-ES on the Rastrigin function and check whether the optimizer
 * converges to the expected solution within tolerance limits.
 */
TEST_CASE("BIPOPCMAESRastriginFunctionTest", "[POPCMAESTest]")
{
  RastriginFunction f(2);
  BoundaryBoxConstraint<> b(-10, 10);

  IPOP_CMAES<FullSelection, BoundaryBoxConstraint<>> ipopcmaes(
    15, // lambda
    b, // transformationPolicy
    32, // batchSize
    10000, // maxIterations
    1e-8, // tolerance
    FullSelection(), // selectionPolicy
    3.72, // stepSize
    2.0, // populationFactor
    7, // maxRestarts
    1e6 // maxFunctionEvaluations
  );

  arma::mat initialPoint = f.GetInitialPoint();
  arma::mat expectedResult = f.GetFinalPoint();

  MultipleTrialOptimizerTest(f, ipopcmaes, initialPoint, expectedResult, 0.5, f.GetFinalObjective(), 0.5, 5);
}

/**
 * Run IPOP-CMA-ES on the Rosenbrock function and check whether the optimizer
 * converges to the expected solution within tolerance limits.
 */
TEST_CASE("IPOPCMAESRosenbrockFunctionTest", "[POPCMAESTest]")
{
  BoundaryBoxConstraint<> b(0, 2);

  BIPOP_CMAES<FullSelection, BoundaryBoxConstraint<>> bipopcmaes(
    15, // lambda
    b, // transformationPolicy
    32, // batchSize
    10000, // maxIterations
    1e-8, // tolerance
    FullSelection(), // selectionPolicy
    0.25, // stepSize
    1.5, // populationFactor
    7, // maxRestarts
    1e6 // maxFunctionEvaluations
  );

  FunctionTest<RosenbrockFunction>(bipopcmaes, 0.5, 0.5);
}

/**
 * Run BIPOP-CMA-ES on the Rosenbrock function and check whether the optimizer
 * converges to the expected solution within tolerance limits.
 */
TEST_CASE("BIPOPCMAESRosenbrockFunctionTest", "[POPCMAESTest]")
{
  BoundaryBoxConstraint<> b(0, 2);

  BIPOP_CMAES<FullSelection, BoundaryBoxConstraint<>> bipopcmaes(
    15, // lambda
    b, // transformationPolicy
    32, // batchSize
    10000, // maxIterations
    1e-8, // tolerance
    FullSelection(), // selectionPolicy
    0.25, // stepSize
    1.5, // populationFactor
    7, // maxRestarts
    1e6 // maxFunctionEvaluations
  );

  FunctionTest<RosenbrockFunction>(bipopcmaes, 0.5, 0.5);
}

/**
 * Run IPOP-CMA-ES with the full selection policy on logistic regression and
 * make sure the results are acceptable.
 */
TEST_CASE("IPOPCMAESLogisticRegressionTest", "[POPCMAESTest]")
{
  BoundaryBoxConstraint<> b(-10, 10);
  IPOP_CMAES<FullSelection, BoundaryBoxConstraint<>> cmaes(0, b, 32, 1000, 1e-3, FullSelection(), 0.6, 2.0, 7, 1e7);
  LogisticRegressionFunctionTest(cmaes, 0.003, 0.006, 5);
}

/**
 * Run BIPOP-CMA-ES with the random selection policy on logistic regression and
 * make sure the results are acceptable.
 */
TEST_CASE("BIPOPCMAESLogisticRegressionTest", "[POPCMAESTest]")
{
  BoundaryBoxConstraint<> b(-10, 10);
  BIPOP_CMAES<FullSelection, BoundaryBoxConstraint<>> cmaes(0, b, 32, 1000, 1e-3, FullSelection(), 0.6, 2.0, 7, 1e7);
  LogisticRegressionFunctionTest(cmaes, 0.003, 0.006, 5);
}
