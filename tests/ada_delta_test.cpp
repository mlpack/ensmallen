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

TEMPLATE_TEST_CASE("AdaDeltaLogisticRegressionTest", "[AdaDelta]",
    arma::mat, arma::fmat)
{
  AdaDelta adaDelta;
  LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(
      adaDelta, 0.003, 0.006, 1);
}

#ifdef USE_COOT

TEMPLATE_TEST_CASE("AdaDeltaLogisticRegressionTest", "[AdaDelta]",
    coot::mat, coot::fmat)
{
  AdaDelta adaDelta;
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      adaDelta, 0.003, 0.006, 1);
}

#endif
