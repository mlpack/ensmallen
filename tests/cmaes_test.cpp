/**
 * @file cmaes_test.cpp
 * @author Marcus Edel
 * @author Kartik Nighania
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
 * Run CMA-ES with the full selection policy on logistic regression and
 * make sure the results are acceptable.
 */
TEMPLATE_TEST_CASE("CMAESLogisticRegressionTest", "[CMAES]", arma::mat)
{
  BoundaryBoxConstraint<> b(-10, 10);
  CMAES<FullSelection, BoundaryBoxConstraint<>> cmaes(0, b, 32, 500, 1e-3);
  cmaes.StepSize() = 0.6;
  LogisticRegressionFunctionTest(cmaes, 0.003, 0.006, 5);
}

/**
 * Run CMA-ES with the random selection policy on logistic regression and
 * make sure the results are acceptable.
 */
TEST_CASE("ApproxCMAESLogisticRegressionTest", "[CMAESTest]")
{
  BoundaryBoxConstraint<> b(-10, 10);
  ApproxCMAES<BoundaryBoxConstraint<arma::mat>> cmaes(256, b, 16, 500, 1e-3);
  cmaes.StepSize() = 0.6;
  LogisticRegressionFunctionTest(cmaes, 0.003, 0.006, 5);
}

/**
 * Run CMA-ES with the full selection policy on logistic regression and
 * make sure the results are acceptable.  Use arma::fmat.
 */
TEST_CASE("CMAESLogisticRegressionFMatTest", "[CMAESTest]")
{
  BoundaryBoxConstraint<arma::fmat> b(-10, 10);
  CMAES<FullSelection, BoundaryBoxConstraint<arma::fmat>> cmaes(120, b, 32, 500, 1e-3);
  LogisticRegressionFunctionTest<arma::fmat>(cmaes, 0.01, 0.02, 5);
}

/**
 * Run CMA-ES with the random selection policy on logistic regression and
 * make sure the results are acceptable.  Use arma::fmat.
 */
/* TEST_CASE("ApproxCMAESLogisticRegressionFMatTest", "[CMAESTest]") */
/* { */
/*   ApproxCMAES<> cmaes(0, -1, 1, 32, 200, 1e-3); */
/*   LogisticRegressionFunctionTest<arma::fmat>(cmaes, 0.01, 0.02, 5); */
/* } */

#ifdef USE_COOT

TEMPLATE_TEST_CASE("CMAESLogisticRegressionTest", "[CMAES]", coot::fmat)
{
  BoundaryBoxConstraint<coot::fmat> b(-10, 10);
  ApproxCMAES<BoundaryBoxConstraint<coot::fmat>> cmaes(0, b, 16, 500, 1e-3);
  LogisticRegressionFunctionTest<coot::fmat>(cmaes, 0.01, 0.02, 5);
}

/**
 * Run CMA-ES with the random selection and empty transformation policies
 * on logistic regression and make sure the results are acceptable.
 * Use arma::fmat.
 */
TEST_CASE("ApproxCMAESEmptyTransformationLogisticRegressionFMatTest",
  "[CMAESTest]")
{
  ApproxCMAES<EmptyTransformation<coot::fmat>>
    cmaes(0, EmptyTransformation<coot::fmat>(), 16, 500, 1e-3);
  LogisticRegressionFunctionTest<coot::fmat>(cmaes, 0.01, 0.02, 5);
}

#endif
