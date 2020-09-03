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
 * Checks if low <= value <= high. Used by NSGA2FonsecaFlemingTest.
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
  const double lowerBound = -1000;
  const double upperBound = 1000;

  NSGA2 opt(20, 5000, 0.5, 0.5, 1e-3, 1e-6, lowerBound, upperBound);

  typedef decltype(SCH.objectiveA) ObjectiveTypeA;
  typedef decltype(SCH.objectiveB) ObjectiveTypeB;

  arma::mat coords = SCH.GetInitialPoint();
  std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = SCH.GetObjectives();

  opt.Optimize(objectives, coords);
  std::vector<arma::mat> bestFront = opt.Front();

  bool allInRange = true;
  double minimumPositive = 1000;

  for (arma::mat solution: bestFront)
  {
    double val = arma::as_scalar(solution);
    if(val >= 0.0)
      minimumPositive = std::min(minimumPositive, val);

    if ((val < 0.0 && std::abs(val) >= minimumPositive) || val > 2.0)
    {
      allInRange = false;
      break;
    }
  }
  REQUIRE(allInRange);
}

/**
 * Optimize for the Schaffer N.1 function using NSGA-II optimizer.
 */
TEST_CASE("NSGA2SchafferN1TestVectorBounds", "[NSGA2Test]")
{
  SchafferFunctionN1<arma::mat> SCH;
  const arma::vec lowerBound = {-1000};
  const arma::vec upperBound = {1000};

  NSGA2 opt(20, 5000, 0.5, 0.5, 1e-3, 1e-6, lowerBound, upperBound);

  typedef decltype(SCH.objectiveA) ObjectiveTypeA;
  typedef decltype(SCH.objectiveB) ObjectiveTypeB;

  arma::mat coords = SCH.GetInitialPoint();
  std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = SCH.GetObjectives();

  opt.Optimize(objectives, coords);
  std::vector<arma::mat> bestFront = opt.Front();

  bool allInRange = true;

  for (arma::mat solution: bestFront)
  {
    double val = arma::as_scalar(solution);

    if (val < 0.0 || val > 2.0)
    {
      allInRange = false;
      break;
    }
  }
  REQUIRE(allInRange);
}

/**
 * Optimize for the Fonseca Fleming function using NSGA-II optimizer.
 */
TEST_CASE("NSGA2FonsecaFlemingTest", "[NSGA2Test]")
{
  FonsecaFlemingFunction<arma::mat> FON;
  const double lowerBound = -4;
  const double upperBound = 4;
  const double tolerance = 1e-6;
  const double strength = 1e-4;
  const double expectedLowerBound = -1.0 / sqrt(3);
  const double expectedUpperBound = 1.0 / sqrt(3);

  NSGA2 opt(20, 4000, 0.6, 0.3, strength, tolerance, lowerBound, upperBound);

  typedef decltype(FON.objectiveA) ObjectiveTypeA;
  typedef decltype(FON.objectiveB) ObjectiveTypeB;

  arma::mat coords = FON.GetInitialPoint();
  std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = FON.GetObjectives();

  opt.Optimize(objectives, coords);
  std::vector<arma::mat> bestFront = opt.Front();

  bool allInRange = true;

  for (size_t i = 0; i < bestFront.size(); i++)
  {
    const arma::mat solution = bestFront[i];
    double valX = arma::as_scalar(solution(0));
    double valY = arma::as_scalar(solution(1));
    double valZ = arma::as_scalar(solution(2));

    if (!IsInBounds(valX, expectedLowerBound, expectedUpperBound) ||
        !IsInBounds(valY, expectedLowerBound, expectedUpperBound) ||
        !IsInBounds(valZ, expectedLowerBound, expectedUpperBound))
    {
      allInRange = false;
      break;
    }
  }
  REQUIRE(allInRange);
}

/**
 * Optimize for the Fonseca Fleming function using NSGA-II optimizer.
 */
TEST_CASE("NSGA2FonsecaFlemingTestVectorBounds", "[NSGA2Test]")
{
  FonsecaFlemingFunction<arma::mat> FON;
  const arma::vec lowerBound = {-4, -4, -4};
  const arma::vec upperBound = {4, 4, 4};
  const double tolerance = 1e-6;
  const double strength = 1e-4;
  const double expectedLowerBound = -1.0 / sqrt(3);
  const double expectedUpperBound = 1.0 / sqrt(3);

  NSGA2 opt(20, 4000, 0.6, 0.3, strength, tolerance, lowerBound, upperBound);

  typedef decltype(FON.objectiveA) ObjectiveTypeA;
  typedef decltype(FON.objectiveB) ObjectiveTypeB;

  arma::mat coords = FON.GetInitialPoint();
  std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = FON.GetObjectives();

  opt.Optimize(objectives, coords);
  std::vector<arma::mat> bestFront = opt.Front();

  bool allInRange = true;

  for (size_t i = 0; i < bestFront.size(); i++)
  {
    const arma::mat solution = bestFront[i];
    double valX = arma::as_scalar(solution(0));
    double valY = arma::as_scalar(solution(1));
    double valZ = arma::as_scalar(solution(2));

    if (!IsInBounds(valX, expectedLowerBound, expectedUpperBound) ||
        !IsInBounds(valY, expectedLowerBound, expectedUpperBound) ||
        !IsInBounds(valZ, expectedLowerBound, expectedUpperBound))
    {
      allInRange = false;
      break;
    }
  }
  REQUIRE(allInRange);
}
