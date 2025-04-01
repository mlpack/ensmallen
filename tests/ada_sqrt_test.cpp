/**
 * @file ada_sqrt_test.cpp
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

TEMPLATE_TEST_CASE("AdaSqrtLogisticRegressionTest", "[AdaSqrt]",
    arma::mat, arma::fmat)
{
  AdaSqrt optimizer(0.01, 32, 1e-8, 5000000, 1e-9, true);
  LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(
      optimizer, 0.003, 0.006, 1);
}

#ifdef ENS_HAS_COOT

TEMPLATE_TEST_CASE("AdaSqrtLogisticRegressionTest", "[AdaSqrt]",
    coot::mat, coot::fmat)
{
  AdaSqrt optimizer(0.01, 32, 1e-8, 5000000, 1e-9, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      optimizer, 0.003, 0.006, 1);
}

#endif
