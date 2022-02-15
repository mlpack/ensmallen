/**
 * @file moead_test.cpp
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
 * Checks if low <= value <= high. Used by MOEADFonsecaFlemingTest.
 *
 * @param value The value being checked.
 * @param low The lower bound.
 * @param high The upper bound.
 * @param roundoff To round off precision.
 * @tparam The type of elements in the population set.
 * @return true if value lies in the range [low, high].
 * @return false if value does not lie in the range [low, high].
 */
template<typename ElemType>
bool IsInBounds(const ElemType& value,
                const ElemType& low,
                const ElemType& high,
                const ElemType& roundoff)
{
  return !(value < (low - roundoff)) && !((high + roundoff) < value);
}

/**
 * Optimize for the Schaffer N.1 function using NSGA-II optimizer.
 * Tests for data of type double.
 */
TEST_CASE("MOEADSchafferN1DoubleTest", "[MOEADTest]")
{
  SchafferFunctionN1<arma::mat> SCH;
  const double lowerBound = -1000;
  const double upperBound = 1000;
  const double expectedLowerBound = 0.0;
  const double expectedUpperBound = 2.0;

  DefaultMOEAD opt(
        300, // Population size.
        300,  // Max generations.
        1.0,  // Crossover probability.
        0.9, // Probability of sampling from neighbor.
        20, // Neighborhood size.
        20, // Perturbation index.
        0.5, // Differential weight.
        2, // Max childrens to replace parents.
        1E-10, // epsilon.
        lowerBound, // Lower bound.
        upperBound // Upper bound.
      );

  typedef decltype(SCH.objectiveA) ObjectiveTypeA;
  typedef decltype(SCH.objectiveB) ObjectiveTypeB;

  // We allow a few trials in case of poor convergence.
  bool success = false;
  for (size_t trial = 0; trial < 5; ++trial)
  {
    arma::mat coords = SCH.GetInitialPoint();
    std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = SCH.GetObjectives();

    opt.Optimize(objectives, coords);
    arma::cube paretoSet= opt.ParetoSet();

    bool allInRange = true;

    for (size_t solutionIdx = 0; solutionIdx < paretoSet.n_slices; ++solutionIdx)
    {
      double val = arma::as_scalar(paretoSet.slice(solutionIdx));
      if (!IsInBounds<double>(val, expectedLowerBound, expectedUpperBound, 0.1))
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
TEST_CASE("MOEADSchafferN1TestVectorDoubleBounds", "[MOEADTest]")
{
  // This test can be a little flaky, so we try it a few times.
  SchafferFunctionN1<arma::mat> SCH;
  const arma::vec lowerBound = {-1000};
  const arma::vec upperBound = {1000};
  const double expectedLowerBound = 0.0;
  const double expectedUpperBound = 2.0;

  DefaultMOEAD opt(
        300, // Population size.
        300,  // Max generations.
        1.0,  // Crossover probability.
        0.9, // Probability of sampling from neighbor.
        20, // Neighborhood size.
        20, // Perturbation index.
        0.5, // Differential weight.
        2, // Max childrens to replace parents.
        1E-10, // epsilon.
        lowerBound, // Lower bound.
        upperBound // Upper bound.
      );

  typedef decltype(SCH.objectiveA) ObjectiveTypeA;
  typedef decltype(SCH.objectiveB) ObjectiveTypeB;

  bool success = false;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    arma::mat coords = SCH.GetInitialPoint();
    std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = SCH.GetObjectives();

    opt.Optimize(objectives, coords);
    arma::cube paretoSet = opt.ParetoSet();

    bool allInRange = true;

    for (size_t solutionIdx = 0; solutionIdx < paretoSet.n_slices; ++solutionIdx)
    {
      double val = arma::as_scalar(paretoSet.slice(solutionIdx));
      if (!IsInBounds<double>(val, expectedLowerBound, expectedUpperBound, 0.1))
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
TEST_CASE("MOEADFonsecaFlemingDoubleTest", "[MOEADTest]")
{
  FonsecaFlemingFunction<arma::mat> FON;
  const double lowerBound = -4;
  const double upperBound = 4;
  const double expectedLowerBound = -1.0 / sqrt(3);
  const double expectedUpperBound = 1.0 / sqrt(3);

  DefaultMOEAD opt(
        300,  // Max generations.
        300,  // Max generations.
        1.0,  // Crossover probability.
        0.9, // Probability of sampling from neighbor.
        20, // Neighborhood size.
        20, // Perturbation index.
        0.5, // Differential weight.
        2, // Max childrens to replace parents.
        1E-10, // epsilon.
        lowerBound, // Lower bound.
        upperBound // Upper bound.
      );
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

    if (!IsInBounds<double>(valX, expectedLowerBound, expectedUpperBound, 0.1) ||
        !IsInBounds<double>(valY, expectedLowerBound, expectedUpperBound, 0.1) ||
        !IsInBounds<double>(valZ, expectedLowerBound, expectedUpperBound, 0.1))
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
TEST_CASE("MOEADFonsecaFlemingTestVectorDoubleBounds", "[MOEADTest]")
{
  FonsecaFlemingFunction<arma::mat> FON;
  const arma::vec lowerBound = {-4, -4, -4};
  const arma::vec upperBound = {4, 4, 4};
  const double expectedLowerBound = -1.0 / sqrt(3);
  const double expectedUpperBound = 1.0 / sqrt(3);

  DefaultMOEAD opt(
        300,  // Max generations.
        300,  // Max generations.
        1.0,  // Crossover probability.
        0.9, // Probability of sampling from neighbor.
        20, // Neighborhood size.
        20, // Perturbation index.
        0.5, // Differential weight.
        2, // Max childrens to replace parents.
        1E-10, // epsilon.
        lowerBound, // Lower bound.
        upperBound // Upper bound.
      );
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

    if (!IsInBounds<double>(valX, expectedLowerBound, expectedUpperBound, 0.1) ||
        !IsInBounds<double>(valY, expectedLowerBound, expectedUpperBound, 0.1) ||
        !IsInBounds<double>(valZ, expectedLowerBound, expectedUpperBound, 0.1))
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
TEST_CASE("MOEADSchafferN1FloatTest", "[MOEADTest]")
{
  SchafferFunctionN1<arma::fmat> SCH;
  const double lowerBound = -1000;
  const double upperBound = 1000;
  const double expectedLowerBound = 0.0;
  const double expectedUpperBound = 2.0;

  DefaultMOEAD opt(
        300, // Population size.
        300,  // Max generations.
        1.0,  // Crossover probability.
        0.9, // Probability of sampling from neighbor.
        20, // Neighborhood size.
        20, // Perturbation index.
        0.5, // Differential weight.
        2, // Max childrens to replace parents.
        1E-10, // epsilon.
        lowerBound, // Lower bound.
        upperBound // Upper bound.
      );

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
      if (!IsInBounds<float>(val, expectedLowerBound, expectedUpperBound, 0.1))
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
TEST_CASE("MOEADSchafferN1TestVectorFloatBounds", "[MOEADTest]")
{
  // This test can be a little flaky, so we try it a few times.
  SchafferFunctionN1<arma::fmat> SCH;
  const arma::vec lowerBound = {-1000};
  const arma::vec upperBound = {1000};
  const double expectedLowerBound = 0.0;
  const double expectedUpperBound = 2.0;

  DefaultMOEAD opt(
        300, // Population size.
        300,  // Max generations.
        1.0,  // Crossover probability.
        0.9, // Probability of sampling from neighbor.
        20, // Neighborhood size.
        20, // Perturbation index.
        0.5, // Differential weight.
        2, // Max childrens to replace parents.
        1E-10, // epsilon.
        lowerBound, // Lower bound.
        upperBound // Upper bound.
      );

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
      if (!IsInBounds<float>(val, expectedLowerBound, expectedUpperBound, 0.1))
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
TEST_CASE("MOEADFonsecaFlemingFloatTest", "[MOEADTest]")
{
  FonsecaFlemingFunction<arma::fmat> FON;
  const double lowerBound = -4;
  const double upperBound = 4;
  const float expectedLowerBound = -1.0 / sqrt(3);
  const float expectedUpperBound = 1.0 / sqrt(3);

  DefaultMOEAD opt(
        300,  // Max generations.
        300,  // Max generations.
        1.0,  // Crossover probability.
        0.9, // Probability of sampling from neighbor.
        20, // Neighborhood size.
        20, // Perturbation index.
        0.5, // Differential weight.
        2, // Max childrens to replace parents.
        1E-10, // epsilon.
        lowerBound, // Lower bound.
        upperBound // Upper bound.
      );
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

    if (!IsInBounds<float>(valX, expectedLowerBound, expectedUpperBound, 0.1) ||
        !IsInBounds<float>(valY, expectedLowerBound, expectedUpperBound, 0.1) ||
        !IsInBounds<float>(valZ, expectedLowerBound, expectedUpperBound, 0.1))
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
TEST_CASE("MOEADFonsecaFlemingTestVectorFloatBounds", "[MOEADTest]")
{
  FonsecaFlemingFunction<arma::fmat> FON;
  const arma::vec lowerBound = {-4, -4, -4};
  const arma::vec upperBound = {4, 4, 4};
  const float expectedLowerBound = -1.0 / sqrt(3);
  const float expectedUpperBound = 1.0 / sqrt(3);

  DefaultMOEAD opt(
        300,  // Max generations.
        300,  // Max generations.
        1.0,  // Crossover probability.
        0.9, // Probability of sampling from neighbor.
        20, // Neighborhood size.
        20, // Perturbation index.
        0.5, // Differential weight.
        2, // Max childrens to replace parents.
        1E-10, // epsilon.
        lowerBound, // Lower bound.
        upperBound // Upper bound.
      );
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

    if (!IsInBounds<float>(valX, expectedLowerBound, expectedUpperBound, 0.1) ||
        !IsInBounds<float>(valY, expectedLowerBound, expectedUpperBound, 0.1) ||
        !IsInBounds<float>(valZ, expectedLowerBound, expectedUpperBound, 0.1))
    {
      allInRange = false;
      break;
    }
  }

  REQUIRE(allInRange);
}

/**
 * Test against the first problem of ZDT Test Suite.  ZDT-1 is a 30 
 * variable-2 objective problem with a convex Pareto Front. 
 *
 * NOTE: For the sake of runtime, only ZDT-1 is tested against the
 * algorithm. Others have been tested separately.
 *
 * We run the test multiple times, since it sometimes fails, in order to get the
 * probability of failure down.
 */
TEST_CASE("MOEADZDTONETest", "[MOEADTest]")
{
  //! Parameters taken from original ZDT Paper.
  ZDT1<> ZDT_ONE(100);
  const double lowerBound = 0;
  const double upperBound = 1;

  DefaultMOEAD opt(
      300, // Population size.
      150,  // Max generations.
      1.0,  // Crossover probability.
      0.9, // Probability of sampling from neighbor.
      20, // Neighborhood size.
      20, // Perturbation index.
      0.5, // Differential weight.
      2, // Max childrens to replace parents.
      1E-10, // epsilon.
      lowerBound, // Lower bound.
      upperBound // Upper bound.
    );

  typedef decltype(ZDT_ONE.objectiveF1) ObjectiveTypeA;
  typedef decltype(ZDT_ONE.objectiveF2) ObjectiveTypeB;

  const size_t trials = 8;
  for (size_t trial = 0; trial < trials; ++trial)
  {
    arma::mat coords = ZDT_ONE.GetInitialPoint();
    std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives =
        ZDT_ONE.GetObjectives();

    opt.Optimize(objectives, coords);

    //! Refer the ZDT_ONE implementation for g objective implementation.
    //! The optimal g value is taken from the docs of ZDT_ONE.
    size_t numVariables = coords.size();
    double sum = arma::accu(coords(arma::span(1, numVariables - 1), 0));
    const double g = 1.0 + 9.0 * sum / (static_cast<double>(numVariables - 1));
    if (trial < trials - 1 && g != Approx(1.0).margin(0.99))
      continue;

    REQUIRE(g == Approx(1.0).margin(0.99));
    break;
  }
}

/**
 * Check if the final population lies in the optimal region in variable space.
 *
 * @param paretoSet The final population in variable space.
 */
bool VariableBoundsCheck(const arma::cube& paretoSet)
{
  bool inBounds = true;
  const arma::mat regions{
    {0.0, 0.182228780, 0.4093136748,
      0.6183967944, 0.8233317983},
    {0.0830015349, 0.2577623634, 0.4538821041,
      0.6525117038, 0.8518328654}
  };

  for (size_t pointIdx = 0; pointIdx < paretoSet.n_slices; ++pointIdx)
  {
    const arma::mat& point = paretoSet.slice(pointIdx);
    const double firstVariable = point(0, 0);

    const bool notInRegion0 = !IsInBounds<double>(firstVariable, regions(0, 0), regions(1, 0), 1e-2);
    const bool notInRegion1 = !IsInBounds<double>(firstVariable, regions(0, 1), regions(1, 1), 1e-2);
    const bool notInRegion2 = !IsInBounds<double>(firstVariable, regions(0, 2), regions(1, 2), 1e-2);
    const bool notInRegion3 = !IsInBounds<double>(firstVariable, regions(0, 3), regions(1, 3), 1e-2);
    const bool notInRegion4 = !IsInBounds<double>(firstVariable, regions(0, 4), regions(1, 4), 1e-2);

    if (notInRegion0 && notInRegion1 && notInRegion2 && notInRegion3 && notInRegion4)
    {
      inBounds = false;
      break;
    }
  }

  return inBounds;
}

/**
 * Test DirichletMOEAD against the third problem of ZDT Test Suite. ZDT-3 is a 30 
 * variable-2 objective problem with disconnected Pareto Fronts. 
 */
TEST_CASE("MOEADDIRICHLETZDT3Test", "[MOEADTest]")
{
  //! Parameters taken from original ZDT Paper.
  ZDT3<> ZDT_THREE(300);
  const double lowerBound = 0;
  const double upperBound = 1;

  DirichletMOEAD opt(
      300, // Population size.
      300,  // Max generations.
      1.0,  // Crossover probability.
      0.9, // Probability of sampling from neighbor.
      20, // Neighborhood size.
      20, // Perturbation index.
      0.5, // Differential weight.
      2, // Max childrens to replace parents.
      1E-10, // epsilon.
      lowerBound, // Lower bound.
      upperBound // Upper bound.
    );

  typedef decltype(ZDT_THREE.objectiveF1) ObjectiveTypeA;
  typedef decltype(ZDT_THREE.objectiveF2) ObjectiveTypeB;

  arma::mat coords = ZDT_THREE.GetInitialPoint();
  std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = ZDT_THREE.GetObjectives();

  opt.Optimize(objectives, coords);

  const arma::cube& finalPopulation = opt.ParetoSet();
  REQUIRE(VariableBoundsCheck(finalPopulation));
}
