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
#include "test_types.hpp"

using namespace ens;
using namespace ens::test;

TEMPLATE_TEST_CASE("SMORMS3_LogisticRegressionFunction", "[SMORMS3]",
    ENS_TEST_TYPES)
{
  SMORMS3 smorms3;
  LogisticRegressionFunctionTest<TestType>(smorms3);
}

#ifdef USE_COOT

TEMPLATE_TEST_CASE("SMORMS3_LogisticRegressionFunction", "[SMORMS3]",
    coot::mat, coot::fmat)
{
  SMORMS3 smorms3;
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      smorms3, 0.003, 0.006);
}

#endif
