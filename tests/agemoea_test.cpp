/**
 * @file agemoea_test.cpp
 * @author Satyam Shukla
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

// NOTE: we can't use ENS_TEST_TYPES for AGEMOEA, because AGEMOEA uses
// solve(), which is not implemented for FP16.

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
 * Optimize for the Schaffer N.1 function using AGE-MOEA optimizer.
 */
TEMPLATE_TEST_CASE("AGEMOEASchafferN1Test", "[AGEMOEA]", ENS_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  SchafferFunctionN1<TestType> sch;
  const double lowerBound = -1000;
  const double upperBound = 1000;
  const ElemType expectedLowerBound = 0;
  const ElemType expectedUpperBound = 2;

  AGEMOEA opt(20, 500, 0.6, 20, 1e-6, 20, lowerBound, upperBound);

  typedef decltype(sch.objectiveA) ObjectiveTypeA;
  typedef decltype(sch.objectiveB) ObjectiveTypeB;

  // We allow a few trials in case of poor convergence.
  bool success = false;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    TestType coords = sch.GetInitialPoint();
    std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = sch.GetObjectives();

    opt.Optimize(objectives, coords);
    arma::Cube<ElemType> paretoSet = arma::conv_to<arma::Cube<ElemType>>::from(
        opt.ParetoSet());

    bool allInRange = true;

    for (size_t solutionIdx = 0; solutionIdx < paretoSet.n_slices;
         ++solutionIdx)
    {
      ElemType val = arma::as_scalar(paretoSet.slice(solutionIdx));
      if (!IsInBounds<ElemType>(val, expectedLowerBound, expectedUpperBound,
          ElemType(0.1)))
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
 * Optimize for the Schaffer N.1 function using AGE-MOEA optimizer.
 */
TEMPLATE_TEST_CASE("AGEMOEASchafferN1TestVectorBounds", "[AGEMOEA]",
    ENS_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  // This test can be a little flaky, so we try it a few times.
  SchafferFunctionN1<TestType> sch;
  const arma::vec lowerBound = {-1000};
  const arma::vec upperBound = {1000};
  const ElemType expectedLowerBound = 0;
  const ElemType expectedUpperBound = 2;

  AGEMOEA opt(20, 500, 0.6, 20, 1e-6, 20, lowerBound, upperBound);

  typedef decltype(sch.objectiveA) ObjectiveTypeA;
  typedef decltype(sch.objectiveB) ObjectiveTypeB;

  bool success = false;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    TestType coords = sch.GetInitialPoint();
    std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = sch.GetObjectives();

    opt.Optimize(objectives, coords);
    arma::Cube<ElemType> paretoSet = arma::conv_to<arma::Cube<ElemType>>::from(
        opt.ParetoSet());

    bool allInRange = true;

    for (size_t solutionIdx = 0; solutionIdx < paretoSet.n_slices;
         ++solutionIdx)
    {
      ElemType val = arma::as_scalar(paretoSet.slice(solutionIdx));
      if (!IsInBounds<ElemType>(val, expectedLowerBound, expectedUpperBound,
          ElemType(0.1)))
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
 * Optimize for the Fonseca Fleming function using AGE-MOEA optimizer.
 */
TEMPLATE_TEST_CASE("AGEMOEAFonsecaFlemingTest", "[AGEMOEA]", ENS_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  FonsecaFlemingFunction<TestType> fon;
  const double lowerBound = -4;
  const double upperBound = 4;
  const ElemType lbExpected = -1 / sqrt(ElemType(3));
  const ElemType ubExpected = 1 / sqrt(ElemType(3));

  AGEMOEA opt(20, 500, 0.6, 20, 1e-6, 20, lowerBound, upperBound);
  typedef decltype(fon.objectiveA) ObjectiveTypeA;
  typedef decltype(fon.objectiveB) ObjectiveTypeB;

  bool success = false;
  for (size_t trial = 0; trial < 6; ++trial)
  {
    TestType coords = fon.GetInitialPoint();
    std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = fon.GetObjectives();

    opt.Optimize(objectives, coords);
    arma::Cube<ElemType> paretoSet = arma::conv_to<arma::Cube<ElemType>>::from(
        opt.ParetoSet());

    bool allInRange = true;

    for (size_t solutionIdx = 0; solutionIdx < paretoSet.n_slices;
         ++solutionIdx)
    {
      const TestType& solution = paretoSet.slice(solutionIdx);
      const ElemType valX = arma::as_scalar(solution(0));
      const ElemType valY = arma::as_scalar(solution(1));
      const ElemType valZ = arma::as_scalar(solution(2));

      if (!IsInBounds<ElemType>(valX, lbExpected, ubExpected, ElemType(0.3)) ||
          !IsInBounds<ElemType>(valY, lbExpected, ubExpected, ElemType(0.3)) ||
          !IsInBounds<ElemType>(valZ, lbExpected, ubExpected, ElemType(0.3)))
      {
        allInRange = false;
        break;
      }
    }

    if (allInRange == true)
    {
      success = true;
      break;
    }
  }

  REQUIRE(success == true);
}

/**
 * Optimize for the Fonseca Fleming function using AGE-MOEA optimizer.
 */
TEMPLATE_TEST_CASE("AGEMOEAFonsecaFlemingTestVectorBounds", "[AGEMOEA]",
    ENS_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  FonsecaFlemingFunction<TestType> fon;
  const arma::vec lowerBound = {-4, -4, -4};
  const arma::vec upperBound = {4, 4, 4};
  const ElemType lbExpected = -1 / sqrt(ElemType(3));
  const ElemType ubExpected = 1 / sqrt(ElemType(3));

  AGEMOEA opt(20, 300, 0.6, 20, 1e-6, 20, lowerBound, upperBound);
  typedef decltype(fon.objectiveA) ObjectiveTypeA;
  typedef decltype(fon.objectiveB) ObjectiveTypeB;

  bool success = false;
  for (size_t trial = 0; trial < 6; ++trial)
  {
    TestType coords = fon.GetInitialPoint();
    std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = fon.GetObjectives();

    opt.Optimize(objectives, coords);
    arma::Cube<ElemType> paretoSet = arma::conv_to<arma::Cube<ElemType>>::from(
        opt.ParetoSet());

    bool allInRange = true;
    for (size_t solutionIdx = 0; solutionIdx < paretoSet.n_slices;
         ++solutionIdx)
    {
      const TestType& solution = paretoSet.slice(solutionIdx);
      const ElemType valX = arma::as_scalar(solution(0));
      const ElemType valY = arma::as_scalar(solution(1));
      const ElemType valZ = arma::as_scalar(solution(2));

      if (!IsInBounds<ElemType>(valX, lbExpected, ubExpected, ElemType(0.3)) ||
          !IsInBounds<ElemType>(valY, lbExpected, ubExpected, ElemType(0.3)) ||
          !IsInBounds<ElemType>(valZ, lbExpected, ubExpected, ElemType(0.3)))
      {
        allInRange = false;
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

TEMPLATE_TEST_CASE("AGEMOEAZDTONETest", "[AGEMOEA]", ENS_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  //! Parameters taken from original ZDT Paper.
  ZDT1<TestType> zdt1(100);
  const double lowerBound = 0;
  const double upperBound = 1;

  AGEMOEA opt(20, 300, 0.6, 20, 1e-6, 20, lowerBound, upperBound);

  typedef decltype(zdt1.objectiveF1) ObjectiveTypeA;
  typedef decltype(zdt1.objectiveF2) ObjectiveTypeB;

  const size_t trials = 8;
  for (size_t trial = 0; trial < trials; ++trial)
  {
    TestType coords = zdt1.GetInitialPoint();
    std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives =
        zdt1.GetObjectives();

    opt.Optimize(objectives, coords);

    //! Refer the ZDT_ONE implementation for g objective implementation.
    //! The optimal g value is taken from the docs of ZDT_ONE.
    size_t numVariables = coords.size();
    ElemType sum = arma::accu(coords(arma::span(1, numVariables - 1), 0));
    const ElemType g = 1 + 9 * sum / (static_cast<ElemType>(numVariables - 1));
    if (trial < trials - 1 && g != Approx(1).margin(ElemType(0.99)))
      continue;

    REQUIRE(g == Approx(1).margin(ElemType(0.99)));
    break;
  }
}

/**
 * Check if the final population lies in the optimal region in variable space.
 *
 * @param paretoSet The final population in variable space.
 */
template<typename CubeType>
bool AVariableBoundsCheck(const CubeType& paretoSet)
{
  typedef typename CubeType::elem_type ElemType;

  const arma::Mat<ElemType> regions{
    {0.0, 0.182228780, 0.4093136748, 0.6183967944, 0.8233317983},
    {0.0830015349, 0.2577623634, 0.4538821041, 0.6525117038, 0.8518328654}};

  ElemType notInBounds = 0;
  for (size_t pointIdx = 0; pointIdx < paretoSet.n_slices; ++pointIdx)
  {
    const arma::Mat<ElemType>& point = paretoSet.slice(pointIdx);
    const ElemType firstVariable = point(0, 0);

    const bool notInRegion0 = !IsInBounds<ElemType>(firstVariable,
        regions(0, 0), regions(1, 0), 3e-2);
    const bool notInRegion1 = !IsInBounds<ElemType>(firstVariable,
        regions(0, 1), regions(1, 1), 3e-2);
    const bool notInRegion2 = !IsInBounds<ElemType>(firstVariable,
        regions(0, 2), regions(1, 2), 3e-2);
    const bool notInRegion3 = !IsInBounds<ElemType>(firstVariable,
        regions(0, 3), regions(1, 3), 3e-2);
    const bool notInRegion4 = !IsInBounds<ElemType>(firstVariable,
        regions(0, 4), regions(1, 4), 3e-2);

    if (notInRegion0 && notInRegion1 && notInRegion2 && notInRegion3 &&
        notInRegion4)
    {
      notInBounds++;
    }
  }

  notInBounds = notInBounds / paretoSet.n_slices;
  return notInBounds < ElemType(0.80);
}

/**
 * Test AGEMOEA against the third problem of ZDT Test Suite. ZDT-3 is a 30
 * variable-2 objective problem with disconnected Pareto Fronts.
 */
TEMPLATE_TEST_CASE("AGEMOEADIRICHLETZDT3Test", "[AGEMOEAD]", ENS_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  //! Parameters taken from original ZDT Paper.
  ZDT3<TestType> zdt3(300);
  const double lowerBound = 0;
  const double upperBound = 1;

  AGEMOEA opt(50, 500, 0.8, 20, 1e-6, 20, lowerBound, upperBound);

  typedef decltype(zdt3.objectiveF1) ObjectiveTypeA;
  typedef decltype(zdt3.objectiveF2) ObjectiveTypeB;
  bool success = true;
  for (size_t tries = 0; tries < 2; tries++)
  {
    TestType coords = zdt3.GetInitialPoint();
    std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives =
        zdt3.GetObjectives();

    opt.Optimize(objectives, coords);

    const arma::Cube<ElemType> finalPopulation =
        arma::conv_to<arma::Cube<ElemType>>::from(opt.ParetoSet());
    success = AVariableBoundsCheck(finalPopulation);
    if (success)
      break;
  }

  REQUIRE(success);
}
