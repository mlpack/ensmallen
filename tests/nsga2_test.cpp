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
#include "test_types.hpp"

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
bool IsInBounds(
    const ElemType& value, const ElemType& low, const ElemType& high)
{
  ElemType roundoff = 0.1;
  return !(value < (low - roundoff)) && !((high + roundoff) < value);
}

TEMPLATE_TEST_CASE("NSGA2_SchafferFunctionN1ElemTypeBounds", "[NSGA2]",
    ENS_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  SchafferFunctionN1<TestType> sch;
  const double lowerBound = -1000;
  const double upperBound = 1000;
  const ElemType expectedLowerBound = 0.0;
  const ElemType expectedUpperBound = 2.0;

  NSGA2Type<TestType> opt(
      20, 300, 0.5, 0.5, 1e-3, 1e-6, lowerBound, upperBound);

  typedef decltype(sch.objectiveA) ObjectiveTypeA;
  typedef decltype(sch.objectiveB) ObjectiveTypeB;

  // We allow a few trials in case of poor convergence.
  bool success = false;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    TestType coords = sch.GetInitialPoint();
    std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives =
      sch.GetObjectives();

    opt.Optimize(objectives, coords);
    arma::Cube<ElemType> paretoSet = opt.ParetoSet();

    bool allInRange = true;

    for (size_t solutionIdx = 0; solutionIdx < paretoSet.n_slices;
         ++solutionIdx)
    {
      ElemType val = arma::as_scalar(paretoSet.slice(solutionIdx));
      if (!IsInBounds<ElemType>(val, expectedLowerBound, expectedUpperBound))
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

TEMPLATE_TEST_CASE("NSGA2_SchafferFunctionN1VectorBounds", "[NSGA2]",
    ENS_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  // This test can be a little flaky, so we try it a few times.
  SchafferFunctionN1<TestType> sch;
  const arma::Col<ElemType> lowerBound = {-1000};
  const arma::Col<ElemType> upperBound = {1000};
  const ElemType expectedLowerBound = 0.0;
  const ElemType expectedUpperBound = 2.0;

  NSGA2Type<TestType> opt(
      20, 300, 0.5, 0.5, 1e-3, 1e-6, lowerBound, upperBound);

  typedef decltype(sch.objectiveA) ObjectiveTypeA;
  typedef decltype(sch.objectiveB) ObjectiveTypeB;

  bool success = false;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    TestType coords = sch.GetInitialPoint();
    std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = sch.GetObjectives();

    opt.Optimize(objectives, coords);
    arma::Cube<ElemType> paretoSet = opt.ParetoSet();

    bool allInRange = true;

    for (size_t solutionIdx = 0; solutionIdx < paretoSet.n_slices;
         ++solutionIdx)
    {
      ElemType val = arma::as_scalar(paretoSet.slice(solutionIdx));
      if (!IsInBounds<ElemType>(val, expectedLowerBound, expectedUpperBound))
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
 */
TEMPLATE_TEST_CASE("NSGA2_FonsecaFlemingFunctionElemTypeBounds", "[NSGA2]",
    ENS_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  FonsecaFlemingFunction<TestType> fon;

  const double lowerBound = -4;
  const double upperBound = 4;
  const double tolerance = 1e-6;
  const double strength = 1e-4;
  const ElemType expectedLowerBound = -1.0 / sqrt(3);
  const ElemType expectedUpperBound = 1.0 / sqrt(3);

  NSGA2Type<TestType> opt(20, 300, 0.6, 0.3, strength, tolerance, lowerBound,
      upperBound);

  typedef decltype(fon.objectiveA) ObjectiveTypeA;
  typedef decltype(fon.objectiveB) ObjectiveTypeB;

  TestType coords = fon.GetInitialPoint();
  std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = fon.GetObjectives();

  opt.Optimize(objectives, coords);
  arma::Cube<ElemType> paretoSet = opt.ParetoSet();

  bool allInRange = true;

  for (size_t solutionIdx = 0; solutionIdx < paretoSet.n_slices; ++solutionIdx)
  {
    const TestType solution = paretoSet.slice(solutionIdx);
    ElemType valX = arma::as_scalar(solution(0));
    ElemType valY = arma::as_scalar(solution(1));
    ElemType valZ = arma::as_scalar(solution(2));

    if (!IsInBounds<ElemType>(valX, expectedLowerBound, expectedUpperBound) ||
        !IsInBounds<ElemType>(valY, expectedLowerBound, expectedUpperBound) ||
        !IsInBounds<ElemType>(valZ, expectedLowerBound, expectedUpperBound))
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
TEMPLATE_TEST_CASE("NSGA2_FonsecaFlemingFunctionVectorBounds", "[NSGA2]",
    ENS_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  FonsecaFlemingFunction<TestType> fon;

  const arma::Col<ElemType> lowerBound = {-4, -4, -4};
  const arma::Col<ElemType> upperBound = {4, 4, 4};
  const double tolerance = 1e-6;
  const double strength = 1e-4;
  const ElemType expectedLowerBound = -1.0 / sqrt(3);
  const ElemType expectedUpperBound = 1.0 / sqrt(3);

  NSGA2Type<TestType> opt(20, 300, 0.6, 0.3, strength, tolerance, lowerBound,
      upperBound);

  typedef decltype(fon.objectiveA) ObjectiveTypeA;
  typedef decltype(fon.objectiveB) ObjectiveTypeB;

  TestType coords = fon.GetInitialPoint();
  std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = fon.GetObjectives();

  opt.Optimize(objectives, coords);
  arma::Cube<ElemType> paretoSet = opt.ParetoSet();

  bool allInRange = true;

  for (size_t solutionIdx = 0; solutionIdx < paretoSet.n_slices; ++solutionIdx)
  {
    const TestType solution = paretoSet.slice(solutionIdx);
    ElemType valX = arma::as_scalar(solution(0));
    ElemType valY = arma::as_scalar(solution(1));
    ElemType valZ = arma::as_scalar(solution(2));

    if (!IsInBounds<ElemType>(valX, expectedLowerBound, expectedUpperBound) ||
        !IsInBounds<ElemType>(valY, expectedLowerBound, expectedUpperBound) ||
        !IsInBounds<ElemType>(valZ, expectedLowerBound, expectedUpperBound))
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
 */
TEMPLATE_TEST_CASE("NSGA2_ZDTONEFunction", "[NSGA2]", ENS_TEST_TYPES)
{
  //! Parameters taken from original ZDT Paper.
  ZDT1<TestType> zdt1(100);
  const double lowerBound = 0;
  const double upperBound = 1;
  const double tolerance = 1e-6;
  const double mutationRate = 1e-2;
  const double crossoverRate = 0.8;
  const double strength = 1e-4;

  NSGA2Type<TestType> opt(100, 250, crossoverRate, mutationRate, strength,
      tolerance, lowerBound, upperBound);

  typedef decltype(zdt1.objectiveF1) ObjectiveTypeA;
  typedef decltype(zdt1.objectiveF2) ObjectiveTypeB;

  TestType coords = zdt1.GetInitialPoint();
  std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives =
      zdt1.GetObjectives();

  opt.Optimize(objectives, coords);

  // Refer to the zdt1 implementation for g objective implementation.
  // The optimal g value is taken from the docs of zdt1.
  size_t numVariables = coords.size();
  double sum = arma::accu(coords(arma::span(1, numVariables - 1), 0));
  double g = 1. + 9. * sum / (static_cast<double>(numVariables - 1));

  REQUIRE(g == Approx(1.0).margin(0.99));
}

/**
 * Ensure that the reverse-compatible Front() function works.
 *
 * This test can be removed when Front() is removed, in ensmallen 3.x.
 */
TEMPLATE_TEST_CASE("NSGA2_FrontTest", "[NSGA2]", ENS_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  SchafferFunctionN1<TestType> sch;
  const double lowerBound = -1000;
  const double upperBound = 1000;

  NSGA2Type<TestType> opt(
      20, 300, 0.5, 0.5, 1e-3, 1e-6, lowerBound, upperBound);

  typedef decltype(sch.objectiveA) ObjectiveTypeA;
  typedef decltype(sch.objectiveB) ObjectiveTypeB;

  TestType coords = sch.GetInitialPoint();
  std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = sch.GetObjectives();

  opt.Optimize(objectives, coords);
  arma::Cube<ElemType> paretoFront = opt.ParetoFront();

  std::vector<TestType> rcFront = opt.Front();

  REQUIRE(paretoFront.n_slices == rcFront.size());
  for (size_t i = 0; i < paretoFront.n_slices; ++i)
  {
    TestType paretoM = paretoFront.slice(i);
    CheckMatrices(paretoM, rcFront[i]);
  }
}

#ifdef USE_COOT

TEMPLATE_TEST_CASE("NSGA2_SchafferFunctionN1", "[NSGA2]",
    coot::mat, coot::fmat)
{
  typedef typename TestType::elem_type ElemType;

  SchafferFunctionN1<TestType> sch;
  const double lowerBound = -1000;
  const double upperBound = 1000;
  const ElemType expectedLowerBound = 0.0;
  const ElemType expectedUpperBound = 2.0;

  NSGA2Type<TestType> opt(
      20, 300, 0.5, 0.5, 1e-3, 1e-6, lowerBound, upperBound);

  typedef decltype(sch.objectiveA) ObjectiveTypeA;
  typedef decltype(sch.objectiveB) ObjectiveTypeB;

  // We allow a few trials in case of poor convergence.
  bool success = false;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    TestType coords = sch.GetInitialPoint();
    std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives =
      sch.GetObjectives();

    opt.Optimize(objectives, coords);
    coot::Cube<ElemType> paretoSet = opt.ParetoSet();

    bool allInRange = true;

    for (size_t solutionIdx = 0; solutionIdx < paretoSet.n_slices;
         ++solutionIdx)
    {
      ElemType val = coot::as_scalar(paretoSet.slice(solutionIdx));

      if (!IsInBounds<ElemType>(val, expectedLowerBound, expectedUpperBound))
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

TEMPLATE_TEST_CASE("NSGA2_SchafferFunctionN1VectorBounds", "[NSGA2Test]",
    coot::mat, coot::fmat)
{
  typedef typename TestType::elem_type ElemType;

  // This test can be a little flaky, so we try it a few times.
  SchafferFunctionN1<TestType> sch;
  coot::Col<ElemType> lowerBound(1);
  lowerBound(0) = -1000.0;
  coot::Col<ElemType> upperBound(1);
  upperBound(0) = 1000.0;
  const ElemType expectedLowerBound = 0.0;
  const ElemType expectedUpperBound = 2.0;

  NSGA2Type<TestType> opt(
      20, 300, 0.5, 0.5, 1e-3, 1e-6, lowerBound, upperBound);

  typedef decltype(sch.objectiveA) ObjectiveTypeA;
  typedef decltype(sch.objectiveB) ObjectiveTypeB;

  bool success = false;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    TestType coords = sch.GetInitialPoint();
    std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = sch.GetObjectives();

    opt.Optimize(objectives, coords);
    coot::Cube<ElemType> paretoSet = opt.ParetoSet();

    bool allInRange = true;

    for (size_t solutionIdx = 0; solutionIdx < paretoSet.n_slices;
         ++solutionIdx)
    {
      ElemType val = coot::as_scalar(paretoSet.slice(solutionIdx));
      if (!IsInBounds<ElemType>(val, expectedLowerBound, expectedUpperBound))
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

#endif
