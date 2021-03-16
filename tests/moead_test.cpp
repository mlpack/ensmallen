
/**
 * @file moead_test.cpp
 * @authors Sayan Goswami, Utkarsh Rai, Nanubala Gnana Sai
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
 * Checks if low <= value <= high. Used by MOEADFonsecaFlemingTest.
 *
 * @param value The value being checked.
 * @param low The lower bound.
 * @param high The upper bound.
 * @return true if value lies in the range [low, high].
 * @return false if value does not lie in the range [low, high].
 */
bool InBounds(const double& value, const double& low, const double& high)
{
  return !(value < low) && !(high < value);
}

/**
 * Optimize for the Schaffer N.1 function using MOEA/D-DE optimizer.
 */
TEST_CASE("MOEADSchafferN1Test", "[MOEADTest]")
{
  SchafferFunctionN1<arma::mat> SCH;
  const double lowerBound = -10;
  const double upperBound = 10;

  MOEAD opt(
          150, // population size
          1000,  // max generations
          1.0,  // crossover prob
          20, //neighbor size
          20, //distribution index
          0.9, //neighbor prob
          0.5, //differential weight
          2, //maxreplace,
          lowerBound, //lower bound
          upperBound //upper bound
        );

  typedef decltype(SCH.objectiveA) ObjectiveTypeA;
  typedef decltype(SCH.objectiveB) ObjectiveTypeB;

  // We allow a few trials in case of poor convergence.
  bool success = false;
  for (size_t trial = 0; trial < 3; ++trial)
  {
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

    if (allInRange)
    {
      success = true;
      break;
    }
  }

  REQUIRE(success == true);
}

/**
 * Optimize for the Schaffer N.1 function using MOEA/D-DE optimizer.
 */
TEST_CASE("MOEADSchafferN1VectorBoundsTest", "[MOEADTest]")
{
  SchafferFunctionN1<arma::mat> SCH;
  arma::vec lowerBound = {-10};
  arma::vec upperBound = {10};

  MOEAD opt(
          150, // population size
          1000,  // max generations
          1.0,  // crossover prob
          20, //neighbor size
          20, //distribution index
          0.9, //neighbor prob
          0.5, //differential weight
          2, //maxreplace,
          lowerBound, //lower bound
          upperBound //upper bound
        );

  typedef decltype(SCH.objectiveA) ObjectiveTypeA;
  typedef decltype(SCH.objectiveB) ObjectiveTypeB;

  bool success = false;
  for (size_t trial = 0; trial < 3; ++trial)
  {
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

    if (allInRange)
    {
      success = true;
      break;
    }
  }

  REQUIRE(success == true);
}

/**
 * Optimize for the Fonseca Fleming function using MOEA/D-DE optimizer.
 */
TEST_CASE("MOEADFonsecaFlemingTest", "[MOEADTest]")
{
  FonsecaFlemingFunction<arma::mat> FON;
  const double lowerBound = -4;
  const double upperBound = 4;
  const double expectedLowerBound = -1.0 / sqrt(3);
  const double expectedUpperBound = 1.0 / sqrt(3);

  MOEAD opt(
          150, // population size
          1000,  // max generations
          1.0,  // crossover prob
          20, //neighbor size
          20, //distribution index
          0.9, //neighbor prob
          0.5, //differential weight
          2, //maxreplace,
          lowerBound, //lower bound
          upperBound //upper bound
        );

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

    if (!InBounds(valX, expectedLowerBound, expectedUpperBound) ||
        !InBounds(valY, expectedLowerBound, expectedUpperBound) ||
        !InBounds(valZ, expectedLowerBound, expectedUpperBound))
    {
      allInRange = false;
      break;
    }
  }
  REQUIRE(allInRange);
}

/**
 * Optimize for the Fonseca Fleming function using MOEA/D-DE optimizer.
 */
TEST_CASE("MOEADFonsecaFlemingVectorBoundsTest", "[MOEADTest]")
{
  FonsecaFlemingFunction<arma::mat> FON;
  const arma::vec lowerBound = {-4, -4, -4};
  const arma::vec upperBound = {4, 4, 4};
  const double expectedLowerBound = -1.0 / sqrt(3);
  const double expectedUpperBound = 1.0 / sqrt(3);

  MOEAD opt(
          150, // population size
          1000,  // max generations
          1.0,  // crossover prob
          20, //neighbor size
          20, //distribution index
          0.9, //neighbor prob
          0.5, //differential weight
          2, //maxreplace,
          lowerBound, //lower bound
          upperBound //upper bound
        );

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

    if (!InBounds(valX, expectedLowerBound, expectedUpperBound) ||
        !InBounds(valY, expectedLowerBound, expectedUpperBound) ||
        !InBounds(valZ, expectedLowerBound, expectedUpperBound))
    {
      allInRange = false;
      break;
    }
  }
  REQUIRE(allInRange);
}