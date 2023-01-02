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
TEST_CASE("CMAESLogisticRegressionTest", "[CMAESTest]")
{
  arma::mat lb(1, 1, arma::fill::value(-1));
  arma::mat ub(1, 1, arma::fill::value(1));
  BoundaryBoxConstraint b(lb, ub);

  CMAES cmaes(0, b, 32, 200, 1e-3);
  LogisticRegressionFunctionTest(cmaes, 0.003, 0.006, 5);
}

/**
 * Run CMA-ES with the random selection policy on logistic regression and
 * make sure the results are acceptable.
 */
TEST_CASE("ApproxCMAESLogisticRegressionTest", "[CMAESTest]")
{
  arma::mat lb(1, 1, arma::fill::value(-1));
  arma::mat ub(1, 1, arma::fill::value(1));
  BoundaryBoxConstraint b(lb, ub);

  ApproxCMAES<BoundaryBoxConstraint<arma::mat>> cmaes(0, b, 32, 200, 1e-3);
  LogisticRegressionFunctionTest(cmaes, 0.003, 0.006, 5);
}

/**
 * Run CMA-ES with the full selection policy on logistic regression and
 * make sure the results are acceptable.  Use arma::fmat.
 */
TEST_CASE("CMAESLogisticRegressionFMatTest", "[CMAESTest]")
{
  arma::fmat lb(1, 1, arma::fill::value(-1));
  arma::fmat ub(1, 1, arma::fill::value(1));
  BoundaryBoxConstraint b(lb, ub);

  CMAES cmaes(0, b, 32, 200, 1e-3);
  LogisticRegressionFunctionTest<arma::fmat>(cmaes, 0.01, 0.02, 5);
}

/**
 * Run CMA-ES with the random selection policy on logistic regression and
 * make sure the results are acceptable.  Use arma::fmat.
 */
TEST_CASE("ApproxCMAESLogisticRegressionFMatTest", "[CMAESTest]")
{
  arma::fmat lb(1, 1, arma::fill::value(-1));
  arma::fmat ub(1, 1, arma::fill::value(1));
  BoundaryBoxConstraint b(lb, ub);

  ApproxCMAES<BoundaryBoxConstraint<arma::fmat>> cmaes(0, b, 32, 200, 1e-3);
  LogisticRegressionFunctionTest<arma::fmat>(cmaes, 0.01, 0.02, 5);
}
