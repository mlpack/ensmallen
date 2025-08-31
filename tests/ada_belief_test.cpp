/**
 * @file ada_belief_test.cpp
 * @author Marcus Edel
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#if defined(ENS_USE_COOT)
  #include <armadillo>
  #include <bandicoot>
#endif
#include <ensmallen.hpp>
#include "catch.hpp"
#include "test_function_tools.hpp"
#include "test_types.hpp"

using namespace ens;
using namespace ens::test;

TEMPLATE_TEST_CASE("AdaBelief_LogisticRegressionFunction", "[AdaBelief]",
    ENS_ALL_TEST_TYPES)
{
  AdaBelief adaBelief;
  LogisticRegressionFunctionTest<TestType, arma::Row<size_t>>(adaBelief);
}

#ifdef ENS_HAVE_COOT

TEMPLATE_TEST_CASE("AdaBelief_LogisticRegressionFunction", "[AdaBelief]",
    coot::mat, coot::fmat)
{
  AdaBelief adaBelief;
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(adaBelief);
}

#endif
