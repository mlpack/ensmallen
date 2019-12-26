/**
 * @file nsga2_test.cpp
 * @author Sayan Goswami
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
using namespace std;


TEST_CASE("NSGA2SCHTest", "[NSGA2Test]") {
  NSGA2TestFuncSCH<arma::mat> SCH;
  NSGA2 opt(100, 2000, 0.6, 0.3, 1e-3, 1e-3);

  arma::mat coords = SCH.GetInitialPoint();
  std::cout << "NSGA2::Optimzation\n";
  arma::mat bestFront = opt.Optimize(SCH, coords);
  bestFront.print();
}