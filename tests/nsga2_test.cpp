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
 * Checks if low <= value <= high. Used by NSGA2FonsecaFlemmingTest.
 *
 * @param value The value being checked.
 * @param low The lower bound.
 * @param high The upper bound.
 * @return true if value lies in the range [low, high].
 * @return false if value does not lie in the range [low, high].
 */
bool IsInBounds(const double& value, const double& low, const double& high)
{
  return !(value < low) && !(high < value);
}

/**
 * Optimize for the Schaffer N.1 function using NSGA-II optimizer.
 */
TEST_CASE("NSGA2SchafferN1Test", "[NSGA2Test]")
{
  SchafferFunctionN1<arma::mat> SCH;
  arma::vec lowerBound("-1000 -1000");
  arma::vec upperBound("1000 1000");
  NSGA2 opt(20, 5000, 0.5, 0.5, 1e-3, 1e-6, lowerBound, upperBound);

  arma::mat coords = SCH.GetInitialPoint();
  auto objectives = SCH.GetObjectives();

  std::vector<arma::mat> bestFront = opt.Optimize(objectives, coords);

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
  const arma::vec lowerBound("-4 -4");
  const arma::vec upperBound("4 4");
  const double tolerance = 1e-5;
  const double strength = 1e-4;
  const double expectedLowerBound = -1.0f/sqrt(3);
  const double expectedUpperBound = 1.0f/sqrt(3);

  NSGA2 opt(40, 5000, 0.5, 0.5, strength, tolerance, lowerBound, upperBound);

  arma::mat coords = FON.GetInitialPoint();
  auto objectives = FON.GetObjectives();

  std::vector<arma::mat> bestFront = opt.Optimize(objectives, coords);

  bool all_in_range = true;

  for(arma::mat solution: bestFront)
  {
    double valX = arma::as_scalar(solution(0));
    double valY = arma::as_scalar(solution(1));
    double valZ = arma::as_scalar(solution(2));

    if (!IsInBounds(valX, expectedLowerBound, expectedUpperBound) ||
        !IsInBounds(valY, expectedLowerBound, expectedUpperBound) ||
        !IsInBounds(valZ, expectedLowerBound, expectedUpperBound))
    {
      all_in_range = false;
      break;
    }
  }
  REQUIRE(all_in_range);
}
