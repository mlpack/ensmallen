/**
 * @file snorms3_test.cpp
 * @author Vivek Pal
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
 * Run SMORMS3 on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("SMORMS3LogisticRegressionTest","[SMORMS3Test]")
{
  SMORMS3 smorms3;
  LogisticRegressionFunctionTest(smorms3, 0.003, 0.006);
}

/**
 * Run SMORMS3 on logistic regression and make sure the results are acceptable.
 * Use arma::fmat.
 */
TEST_CASE("SMORMS3LogisticRegressionFMatTest","[SMORMS3Test]")
{
  SMORMS3 smorms3;
  LogisticRegressionFunctionTest<arma::fmat>(smorms3, 0.003, 0.006);
}
