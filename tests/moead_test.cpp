
/**
 * @file moead_test.cpp
 * @author Sayan Goswami
 * @author Utkarsh Rai
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
 * Optimize for the Fonseca Fleming function using MOEA/D optimizer.
 */
TEST_CASE("MOEADFonsecaFlemingTest", "[MOEADTest]")
{
  FonsecaFlemingFunction<arma::mat> FON;
  const arma::vec lowerBound("-4 -4 -4");
  const arma::vec upperBound("4 4 4");
  const double strength = 1e-3;
  const double expectedLowerBound = -1.0 / sqrt(3);
  const double expectedUpperBound = 1.0 / sqrt(3);
  MOEAD opt(150, 10, 0.6, 0.7, strength, 10, 0.5, lowerBound, upperBound);

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
    std::cout<<valX<<" "<<valY<<" "<<valZ<<"\n";

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
 * Optimize for the function using MOEA/D optimizer.
 */
TEST_CASE("MOEADSchafferN1Test", "[MOEADTest]")
{
    SchafferFunctionN1<arma::mat> SCH;
    arma::vec lowerBound = {-1000};
    arma::vec upperBound = {1000};

    MOEAD opt(150, 10, 0.6, 0.7, 1e-3, 10, 0.5, lowerBound, upperBound);

    typedef decltype(SCH.objectiveA) ObjectiveTypeA;
    typedef decltype(SCH.objectiveB) ObjectiveTypeB;

    arma::mat coords = SCH.GetInitialPoint();
    std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives =
        SCH.GetObjectives();

    opt.Optimize(objectives, coords);
    std::vector<arma::mat> bestFronts = opt.Front();

    bool allInRange = true;

    for (arma::mat solution: bestFronts)
    {
      double val = arma::as_scalar(solution);
      std::cout<<val<<"\n";
      if (val < 0.0 || val > 2.0)
      {
        allInRange = false;
        break;
      }
    }
  REQUIRE(allInRange);
}
