// Copyright (c) 2018 ensmallen developers.
// 
// Licensed under the 3-clause BSD license (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.opensource.org/licenses/BSD-3-Clause

#include <ensmallen.hpp>
#include "catch.hpp"

using namespace std;
using namespace arma;
using namespace ens;
using namespace ens::test;

// #include <mlpack/core.hpp>
// #include <mlpack/core/optimizers/line_search/line_search.hpp>
// #include <mlpack/core/optimizers/fw/test_func_fw.hpp>
// 
// using namespace mlpack;
// using namespace mlpack::optimization;

/**
 * Simple test of Line Search with TestFuncFW function.
 */
TEST_CASE("FuncFWTest", "[LineSearchTest]")
{
  vec x1 = zeros<vec>(3);
  vec x2;
  x2 << 0.2 << 0.4 << 0.6;

  TestFuncFW f;
  LineSearch s;

  double result = s.Optimize(f, x1, x2);

  REQUIRE(result == Approx(0.0).margin(1e-10));
  REQUIRE((x2[0] - 0.1) == Approx(0.0).margin(1e-10));
  REQUIRE((x2[1] - 0.2) == Approx(0.0).margin(1e-10));
  REQUIRE((x2[2] - 0.3) == Approx(0.0).margin(1e-10));
}
