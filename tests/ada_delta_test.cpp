/**
 * @file ada_delta_test.cpp
 * @author Marcus Edel
 * @author Vasanth Kalingeri
 * @author Abhinav Moudgil
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
 * Run AdaDelta on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("AdaDeltaLogisticRegressionTest", "[AdaDeltaTest]")
{
  AdaDelta adaDelta;
  LogisticRegressionFunctionTest(adaDelta, 0.003, 0.006, 1);
}

/**
 * Run AdaDelta on logistic regression and make sure the results are acceptable
 * with arma::fmat as the type.
 */
TEST_CASE("AdaDeltaLogisticRegressionTestFMat", "[AdaDeltaTest]")
{
  AdaDelta adaDelta;
  LogisticRegressionFunctionTest(adaDelta, 0.003, 0.006, 1);
}
