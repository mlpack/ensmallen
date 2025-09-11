/**
 * @file line_search_test.cpp
 * @author Chenzhe Diao
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
#include "test_types.hpp"

using namespace arma;
using namespace ens;
using namespace ens::test;

/**
 * Simple test of Line Search with TestFuncFW function.
 */
TEMPLATE_TEST_CASE("FuncFWTest", "[LineSearch]", ENS_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  TestType x1 = zeros<TestType>(3, 1);
  TestType x2 = { 0.2, 0.4, 0.6 };
  x2 = x2.t();

  TestFuncFW<TestType> f;
  LineSearch s;

  ElemType result = s.Optimize(f, x1, x2);

  REQUIRE(result == Approx(0.0).margin(Tolerances<TestType>::Obj));
  REQUIRE((x2(0) - 0.1) == Approx(0.0).margin(Tolerances<TestType>::Coord));
  REQUIRE((x2(1) - 0.2) == Approx(0.0).margin(Tolerances<TestType>::Coord));
  REQUIRE((x2(2) - 0.3) == Approx(0.0).margin(Tolerances<TestType>::Coord));
}
