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
  NSGA2 opt(20, 5000, 0.5, 0.5, 1e-3, 1e-6);

  arma::mat coords = SCH.GetInitialPoint();
  std::vector<arma::mat> bestFront = opt.Optimize(SCH, coords);

  bool all_in_range = true;

  for (arma::mat solution: bestFront)
  {
    double val = arma::as_scalar(solution);

    if (val < 0.0 || val > 2.0)
    {
      all_in_range = false;
      break;
    }
  }
  REQUIRE(all_in_range);
}

/**
 * Optimize for the Fonseca Flemming function using NSGA-II optimizer.
 */
TEST_CASE("NSGA2FonsecaFlemmingTest", "[NSGA2Test]")
{
  FonsecaFlemmingFunction<arma::mat> FON;
  NSGA2 opt(20, 5000, 0.5, 0.5, 1e-3, 1e-6);

  arma::mat coords = FON.GetInitialPoint();
  std::vector<arma::mat> bestFront = opt.Optimize(FON, coords);

  bool all_in_range = true;

  for(arma::mat solution: bestFront) {
    double valX = arma::as_scalar(solution(0));
    double valY = arma::as_scalar(solution(1));
    double valZ = arma::as_scalar(solution(2));

    if (valX < -1.0f/sqrt(3) || valX > 1.0f/sqrt(3) || valY < -1.0f/sqrt(3) ||
        valY > 1.0f/sqrt(3) || valZ < -1.0f/sqrt(3) || valZ > 1.0f/sqrt(3)) {
      all_in_range = false;
      break;
    }
  }
  REQUIRE(all_in_range);
}