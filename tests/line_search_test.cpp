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

using namespace arma;
using namespace ens;
using namespace ens::test;

/**
 * Simple test of Line Search with TestFuncFW function.
 */
TEST_CASE("FuncFWTest", "[LineSearchTest]")
{
  mat x1 = zeros<mat>(3, 1);
  mat x2 = { 0.2, 0.4, 0.6 };
  x2 = x2.t();

  TestFuncFW<> f;
  LineSearch s;

  double result = s.Optimize(f, x1, x2);

  REQUIRE(result == Approx(0.0).margin(1e-10));
  REQUIRE((x2(0) - 0.1) == Approx(0.0).margin(1e-10));
  REQUIRE((x2(1) - 0.2) == Approx(0.0).margin(1e-10));
  REQUIRE((x2(2) - 0.3) == Approx(0.0).margin(1e-10));
}

/**
 * Simple test of Line Search with TestFuncFW function.  Use arma::fmat.
 */
TEST_CASE("FuncFWFMatTest", "[LineSearchTest]")
{
  fmat x1 = zeros<fmat>(3, 1);
  fmat x2 = { 0.2, 0.4, 0.6 };
  x2 = x2.t();

  TestFuncFW<arma::fmat> f;
  LineSearch s;

  float result = s.Optimize(f, x1, x2);

  REQUIRE(result == Approx(0.0).margin(1e-5));
  REQUIRE((x2(0) - 0.1) == Approx(0.0).margin(1e-5));
  REQUIRE((x2(1) - 0.2) == Approx(0.0).margin(1e-5));
  REQUIRE((x2(2) - 0.3) == Approx(0.0).margin(1e-5));
}
