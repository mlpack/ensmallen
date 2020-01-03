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


/**
 * Optimize for the Schaffer N.1 function using NSGA-II optimizer.
 */
TEST_CASE("NSGA2SchafferN1Test", "[NSGA2Test]")
{
  SchafferFunctionN1<arma::mat> SCH;
  NSGA2 opt(20, 1000, 0.6, 0.3, 0.01, 1e-6);

  arma::mat coords = SCH.GetInitialPoint();
  Info << "NSGA2:: Begin Optimzation\n";
  std::vector<arma::mat> bestFront = opt.Optimize(SCH, coords);

  for (arma::mat solution: bestFront)
  {
    solution.print();
  }
}