/**
 * @file nsga2_test.cpp
 * @author Sayan Goswami
 * @author Nanubala Gnana Sai
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
 * @tparam The type of elements in the population set.
 * @return true if value lies in the range [low, high].
 * @return false if value does not lie in the range [low, high].
 */
template<typename ElemType>
bool IsInBounds(const ElemType& value, const ElemType& low, const ElemType& high)
{
  ElemType roundoff = 0.1;
  return !(value < (low - roundoff)) && !((high + roundoff) < value);
}

/**
 * Optimize for the Schaffer N.1 function using NSGA-II optimizer.
 * Tests for data of type double.
 */
TEST_CASE("NSGA2SchafferN1DoubleTest", "[NSGA2Test]")
{
  SchafferFunctionN1<arma::mat> SCH;
  const double lowerBound = -1000;
  const double upperBound = 1000;
  const double expectedLowerBound = 0.0;
  const double expectedUpperBound = 2.0;

  NSGA2 opt(20, 300, 0.5, 0.5, 1e-3, 1e-6, lowerBound, upperBound);

  typedef decltype(SCH.objectiveA) ObjectiveTypeA;
  typedef decltype(SCH.objectiveB) ObjectiveTypeB;

  // We allow a few trials in case of poor convergence.
  bool success = false;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    arma::mat coords = SCH.GetInitialPoint();
    std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = SCH.GetObjectives();

    opt.Optimize(objectives, coords);
    arma::cube paretoSet= opt.ParetoSet();

    bool allInRange = true;

    for (size_t solutionIdx = 0; solutionIdx < paretoSet.n_slices; ++solutionIdx)
    {
      double val = arma::as_scalar(paretoSet.slice(solutionIdx));
      if (!IsInBounds<double>(val, expectedLowerBound, expectedUpperBound))
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
 * Optimize for the Schaffer N.1 function using NSGA-II optimizer.
 * Tests for data of type double.
 */
TEST_CASE("NSGA2SchafferN1TestVectorDoubleBounds", "[NSGA2Test]")
{
  // This test can be a little flaky, so we try it a few times.
  SchafferFunctionN1<arma::mat> SCH;
  const arma::vec lowerBound = {-1000};
  const arma::vec upperBound = {1000};
  const double expectedLowerBound = 0.0;
  const double expectedUpperBound = 2.0;

  NSGA2 opt(20, 300, 0.5, 0.5, 1e-3, 1e-6, lowerBound, upperBound);

  typedef decltype(SCH.objectiveA) ObjectiveTypeA;
  typedef decltype(SCH.objectiveB) ObjectiveTypeB;

  bool success = false;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    arma::mat coords = SCH.GetInitialPoint();
    std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = SCH.GetObjectives();

    opt.Optimize(objectives, coords);
    arma::cube paretoSet= opt.ParetoSet();

    bool allInRange = true;

    for (size_t solutionIdx = 0; solutionIdx < paretoSet.n_slices; ++solutionIdx)
    {
      double val = arma::as_scalar(paretoSet.slice(solutionIdx));
      if (!IsInBounds<double>(val, expectedLowerBound, expectedUpperBound))
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
 * Optimize for the Fonseca Fleming function using NSGA-II optimizer.
 * Tests for data of type double.
 */
TEST_CASE("NSGA2FonsecaFlemingDoubleTest", "[NSGA2Test]")
{
  FonsecaFlemingFunction<arma::mat> FON;
  const double lowerBound = -4;
  const double upperBound = 4;
  const double tolerance = 1e-6;
  const double strength = 1e-4;
  const double expectedLowerBound = -1.0 / sqrt(3);
  const double expectedUpperBound = 1.0 / sqrt(3);

  NSGA2 opt(20, 300, 0.6, 0.3, strength, tolerance, lowerBound, upperBound);

  typedef decltype(FON.objectiveA) ObjectiveTypeA;
  typedef decltype(FON.objectiveB) ObjectiveTypeB;

  arma::mat coords = FON.GetInitialPoint();
  std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = FON.GetObjectives();

  opt.Optimize(objectives, coords);
  arma::cube paretoSet = opt.ParetoSet();

  bool allInRange = true;

  for (size_t solutionIdx = 0; solutionIdx < paretoSet.n_slices; ++solutionIdx)
  {
    const arma::mat solution = paretoSet.slice(solutionIdx);
    double valX = arma::as_scalar(solution(0));
    double valY = arma::as_scalar(solution(1));
    double valZ = arma::as_scalar(solution(2));

    if (!IsInBounds<double>(valX, expectedLowerBound, expectedUpperBound) ||
        !IsInBounds<double>(valY, expectedLowerBound, expectedUpperBound) ||
        !IsInBounds<double>(valZ, expectedLowerBound, expectedUpperBound))
    {
      allInRange = false;
      break;
    }
  }

  REQUIRE(allInRange);
}

/**
 * Optimize for the Fonseca Fleming function using NSGA-II optimizer.
 * Tests for data of type double.
 */
TEST_CASE("NSGA2FonsecaFlemingTestVectorDoubleBounds", "[NSGA2Test]")
{
  FonsecaFlemingFunction<arma::mat> FON;
  const arma::vec lowerBound = {-4, -4, -4};
  const arma::vec upperBound = {4, 4, 4};
  const double tolerance = 1e-6;
  const double strength = 1e-4;
  const double expectedLowerBound = -1.0 / sqrt(3);
  const double expectedUpperBound = 1.0 / sqrt(3);

  NSGA2 opt(20, 300, 0.6, 0.3, strength, tolerance, lowerBound, upperBound);

  typedef decltype(FON.objectiveA) ObjectiveTypeA;
  typedef decltype(FON.objectiveB) ObjectiveTypeB;

  arma::mat coords = FON.GetInitialPoint();
  std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = FON.GetObjectives();

  opt.Optimize(objectives, coords);
  arma::cube paretoSet = opt.ParetoSet();

  bool allInRange = true;

  for (size_t solutionIdx = 0; solutionIdx < paretoSet.n_slices; ++solutionIdx)
  {
    const arma::mat solution = paretoSet.slice(solutionIdx);
    double valX = arma::as_scalar(solution(0));
    double valY = arma::as_scalar(solution(1));
    double valZ = arma::as_scalar(solution(2));

    if (!IsInBounds<double>(valX, expectedLowerBound, expectedUpperBound) ||
        !IsInBounds<double>(valY, expectedLowerBound, expectedUpperBound) ||
        !IsInBounds<double>(valZ, expectedLowerBound, expectedUpperBound))
    {
      allInRange = false;
      break;
    }
  }

  REQUIRE(allInRange);
}

/**
 * Optimize for the Schaffer N.1 function using NSGA-II optimizer.
 * Tests for data of type float.
 */
TEST_CASE("NSGA2SchafferN1FloatTest", "[NSGA2Test]")
{
  SchafferFunctionN1<arma::fmat> SCH;
  const double lowerBound = -1000;
  const double upperBound = 1000;
  const double expectedLowerBound = 0.0;
  const double expectedUpperBound = 2.0;

  NSGA2 opt(20, 300, 0.5, 0.5, 1e-3, 1e-6, lowerBound, upperBound);

  typedef decltype(SCH.objectiveA) ObjectiveTypeA;
  typedef decltype(SCH.objectiveB) ObjectiveTypeB;

  // We allow a few trials in case of poor convergence.
  bool success = false;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    arma::fmat coords = SCH.GetInitialPoint();
    std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = SCH.GetObjectives();

    opt.Optimize(objectives, coords);
    arma::fcube paretoSet = arma::conv_to<arma::fcube>::from(opt.ParetoSet());

    bool allInRange = true;

    for (size_t solutionIdx = 0; solutionIdx < paretoSet.n_slices; ++solutionIdx)
    {
      float val = arma::as_scalar(paretoSet.slice(solutionIdx));
      if (!IsInBounds<float>(val, expectedLowerBound, expectedUpperBound))
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
 * Optimize for the Schaffer N.1 function using NSGA-II optimizer.
 * Tests for data of type float.
 */
TEST_CASE("NSGA2SchafferN1TestVectorFloatBounds", "[NSGA2Test]")
{
  // This test can be a little flaky, so we try it a few times.
  SchafferFunctionN1<arma::fmat> SCH;
  const arma::vec lowerBound = {-1000};
  const arma::vec upperBound = {1000};
  const double expectedLowerBound = 0.0;
  const double expectedUpperBound = 2.0;

  NSGA2 opt(20, 300, 0.5, 0.5, 1e-3, 1e-6, lowerBound, upperBound);

  typedef decltype(SCH.objectiveA) ObjectiveTypeA;
  typedef decltype(SCH.objectiveB) ObjectiveTypeB;

  bool success = false;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    arma::fmat coords = SCH.GetInitialPoint();
    std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = SCH.GetObjectives();

    opt.Optimize(objectives, coords);
    arma::fcube paretoSet = arma::conv_to<arma::fcube>::from(opt.ParetoSet());

    bool allInRange = true;

    for (size_t solutionIdx = 0; solutionIdx < paretoSet.n_slices; ++solutionIdx)
    {
      float val = arma::as_scalar(paretoSet.slice(solutionIdx));
      if (!IsInBounds<float>(val, expectedLowerBound, expectedUpperBound))
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
 * Optimize for the Fonseca Fleming function using NSGA-II optimizer.
 * Tests for data of type float.
 */
TEST_CASE("NSGA2FonsecaFlemingFloatTest", "[NSGA2Test]")
{
  FonsecaFlemingFunction<arma::fmat> FON;
  const double lowerBound = -4;
  const double upperBound = 4;
  const double tolerance = 1e-6;
  const double strength = 1e-4;
  const float expectedLowerBound = -1.0 / sqrt(3);
  const float expectedUpperBound = 1.0 / sqrt(3);

  NSGA2 opt(20, 300, 0.6, 0.3, strength, tolerance, lowerBound, upperBound);

  typedef decltype(FON.objectiveA) ObjectiveTypeA;
  typedef decltype(FON.objectiveB) ObjectiveTypeB;

  arma::fmat coords = FON.GetInitialPoint();
  std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = FON.GetObjectives();

  opt.Optimize(objectives, coords);
  arma::fcube paretoSet = arma::conv_to<arma::fcube>::from(opt.ParetoSet());

  bool allInRange = true;

  for (size_t solutionIdx = 0; solutionIdx < paretoSet.n_slices; ++solutionIdx)
  {
    const arma::fmat solution = paretoSet.slice(solutionIdx);
    float valX = arma::as_scalar(solution(0));
    float valY = arma::as_scalar(solution(1));
    float valZ = arma::as_scalar(solution(2));

    if (!IsInBounds<float>(valX, expectedLowerBound, expectedUpperBound) ||
        !IsInBounds<float>(valY, expectedLowerBound, expectedUpperBound) ||
        !IsInBounds<float>(valZ, expectedLowerBound, expectedUpperBound))
    {
      allInRange = false;
      break;
    }
  }

  REQUIRE(allInRange);
}

/**
 * Optimize for the Fonseca Fleming function using NSGA-II optimizer.
 * Tests for data of type float.
 */
TEST_CASE("NSGA2FonsecaFlemingTestVectorFloatBounds", "[NSGA2Test]")
{
  FonsecaFlemingFunction<arma::fmat> FON;
  const arma::vec lowerBound = {-4, -4, -4};
  const arma::vec upperBound = {4, 4, 4};
  const double tolerance = 1e-6;
  const double strength = 1e-4;
  const float expectedLowerBound = -1.0 / sqrt(3);
  const float expectedUpperBound = 1.0 / sqrt(3);

  NSGA2 opt(20, 300, 0.6, 0.3, strength, tolerance, lowerBound, upperBound);

  typedef decltype(FON.objectiveA) ObjectiveTypeA;
  typedef decltype(FON.objectiveB) ObjectiveTypeB;

  arma::fmat coords = FON.GetInitialPoint();
  std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = FON.GetObjectives();

  opt.Optimize(objectives, coords);
  arma::fcube paretoSet = arma::conv_to<arma::fcube>::from(opt.ParetoSet());

  bool allInRange = true;

  for (size_t solutionIdx = 0; solutionIdx < paretoSet.n_slices; ++solutionIdx)
  {
    const arma::fmat solution = paretoSet.slice(solutionIdx);
    float valX = arma::as_scalar(solution(0));
    float valY = arma::as_scalar(solution(1));
    float valZ = arma::as_scalar(solution(2));

    if (!IsInBounds<float>(valX, expectedLowerBound, expectedUpperBound) ||
        !IsInBounds<float>(valY, expectedLowerBound, expectedUpperBound) ||
        !IsInBounds<float>(valZ, expectedLowerBound, expectedUpperBound))
    {
      allInRange = false;
      break;
    }
  }

  REQUIRE(allInRange);
}