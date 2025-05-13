/**
 * @file demon_sgd_test.cpp
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

TEMPLATE_TEST_CASE("DemonSGD_LogisticRegressionFunction", "[DemonSGD]",
    arma::mat, arma::fmat)
{
  DemonSGD optimizer(0.1, 32, 0.9, 1000000, 1e-9, true, true, true);
  LogisticRegressionFunctionTest<TestType>(optimizer, 0.003, 0.006, 6);
}

#ifdef USE_COOT

TEMPLATE_TEST_CASE("DemonSGD_LogisticRegressionFunction", "[DemonSGD]",
    coot::mat, coot::fmat)
{
  DemonSGD optimizer(0.1, 32, 0.9, 1000000, 1e-9, true, true, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      optimizer, 0.003, 0.006, 6);
}

#endif

