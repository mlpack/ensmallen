/**
 * @file de_test.cpp
 * @author Rahul Ganesh Prabhu
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

TEMPLATE_TEST_CASE("DELogisticRegressionTest", "[DE]", arma::mat)
{
  DE opt(200, 1000, 0.6, 0.8, 1e-5);
  LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(
      opt, 0.01, 0.02, 3);
}

#ifdef ENS_HAS_COOT

TEMPLATE_TEST_CASE("DELogisticRegressionTest", "[DE]", coot::mat)
{
  DE opt(200, 1000, 0.6, 0.8, 1e-5);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      opt, 0.01, 0.02, 3);
}

#endif
