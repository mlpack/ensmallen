/**
 * @file nsgaiii_test.cpp
 * @author Rahul Ganesh Prabhu
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
 * Test NSGAIII on DTLZ.
 */
TEST_CASE("NSGAIIITest", "[NSGAIIITest]")
{
  DTLZ1 dtlz1(7, 3);
  NSGAIII opt(92, 400, 1, 30, 12);
  arma::mat coordinates = dtlz1.GetInitialPoint();
  cout << "Starting optimization." << endl;
  arma::cube best = opt.Optimize(dtlz1, coordinates);
  best.print();
}