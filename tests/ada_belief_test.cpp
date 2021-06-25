/**
 * @file ada_belief_test.cpp
 * @author Marcus Edel
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
 * Test the AdaBelief optimizer on the Sphere function.
 */
TEST_CASE("AdaBeliefSphereFunctionTest", "[AdaBeliefTest]")
{
  AdaBelief optimizer(0.5, 2, 0.7, 0.999, 1e-8, 500000, 1e-3, false);
  FunctionTest<SphereFunction>(optimizer, 0.5, 0.1);
}

/**
 * Test the AdaBelief optimizer on the Sphere function with arma::fmat.
 */
TEST_CASE("AdaBeliefSphereFunctionTestFMat", "[AdaBeliefTest]")
{
  AdaBelief optimizer(0.5, 2, 0.7, 0.999, 1e-8, 500000, 1e-3, false);
  FunctionTest<SphereFunction, arma::fmat>(optimizer, 0.5, 0.1);
}

/**
 * Test the AdaBelief optimizer on the McCormick function.
 */
TEST_CASE("AdaBeliefMcCormickFunctionTest", "[AdaBeliefTest]")
{
  AdaBelief optimizer(0.5, 1, 0.7, 0.999, 1e-8, 500000, 1e-5, false);
  FunctionTest<McCormickFunction>(optimizer, 0.5, 0.1);
}

/**
 * Run AdaBelief on logistic regression and make sure the results are
 * acceptable.
 */
TEST_CASE("AdaBeliefLogisticRegressionTest", "[AdaBeliefTest]")
{
  AdaBelief optimizer;
  LogisticRegressionFunctionTest(optimizer, 0.003, 0.006);
}

/**
 * Run AdaBelief on logistic regression and make sure the results are
 * acceptable, using arma::fmat.
 */
TEST_CASE("AdaBeliefLogisticRegressionFMatTest", "[AdaBeliefTest]")
{
  arma::fmat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<arma::fmat> lr(shuffledData, shuffledResponses, 0.5);

  AdaBelief optimizer;
  LogisticRegressionFunctionTest<arma::fmat>(optimizer, 0.003, 0.006);
}

