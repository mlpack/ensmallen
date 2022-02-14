/**
 * @file de_test.cpp
 * @author Rahul Ganesh Prabhu
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include "../include/ensmallen.hpp"
#include "catch.hpp"
#include "test_function_tools.hpp"

using namespace ens;
using namespace ens::test;

/**
 * Train and test a logistic regression function using DE optimizer.
 */
TEST_CASE("DELogisticRegressionTest", "[DETest]")
{
  DE opt(200, 1000, 0.6, 0.8, 1e-5);
  LogisticRegressionFunctionTest(opt, 0.01, 0.02, 3);
}

/**
 * Train and test a logistic regression function using DE optimizer.  Use
 * arma::fmat.
 */
TEST_CASE("DELogisticRegressionFMatTest", "[DETest]")
{
  DE opt(200, 1000, 0.6, 0.8, 1e-5);
  LogisticRegressionFunctionTest<arma::fmat>(opt, 0.03, 0.06, 3);
}
