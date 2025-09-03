/**
 * @file moead_test.cpp
 * @author Nanubala Gnana Sai
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#if defined(ENS_USE_COOT)
  #include <armadillo>
  #include <bandicoot>
#endif
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
 * Check if the final population lies in the optimal region in variable space.
 *
 * @param paretoSet The final population in variable space.
 */
template<typename CubeType>
bool VariableBoundsCheck(const CubeType& paretoSet)
{
  typedef typename ForwardType<CubeType>::bmat BaseMatType;

  bool inBounds = true;
  const BaseMatType regions(
      "0.0 0.182228780 0.4093136748 0.6183967944 0.8233317983; 0.0830015349 \
       0.2577623634 0.4538821041 0.6525117038 0.8518328654");

  for (size_t pointIdx = 0; pointIdx < paretoSet.n_slices; ++pointIdx)
  {
    const BaseMatType& point = paretoSet.slice(pointIdx);
    const double firstVariable = point(0, 0);

    const bool notInRegion0 = !IsInBounds<double>(firstVariable, regions(0, 0),
        regions(1, 0), 1e-2);
    const bool notInRegion1 = !IsInBounds<double>(firstVariable, regions(0, 1),
        regions(1, 1), 1e-2);
    const bool notInRegion2 = !IsInBounds<double>(firstVariable, regions(0, 2),
        regions(1, 2), 1e-2);
    const bool notInRegion3 = !IsInBounds<double>(firstVariable, regions(0, 3),
        regions(1, 3), 1e-2);
    const bool notInRegion4 = !IsInBounds<double>(firstVariable, regions(0, 4),
        regions(1, 4), 1e-2);

    if (notInRegion0 && notInRegion1 && notInRegion2 &&
        notInRegion3 && notInRegion4)
    {
      inBounds = false;
      break;
    }
  }

  return inBounds;
}

TEMPLATE_TEST_CASE("DefaultMOEAD_SchafferFunctionN1", "[MOEAD]",
    arma::mat, arma::fmat)
{
  typedef typename TestType::elem_type ElemType;

  SchafferFunctionN1<TestType> SCH;
  const double lowerBound = -1000;
  const double upperBound = 1000;
  const double expectedLowerBound = 0.0;
  const double expectedUpperBound = 2.0;

  MOEAD<Uniform, Tchebycheff> opt(
      300, // Population size.
      300,  // Max generations.
      1.0,  // Crossover probability.
      0.9, // Probability of sampling from neighbor.
      20, // Neighborhood size.
      20, // Perturbation index.
      0.5, // Differential weight.
      2, // Max childrens to replace parents.
      1e-10, // epsilon.
      lowerBound, // Lower bound.
      upperBound // Upper bound.
  );

  typedef decltype(SCH.objectiveA) ObjectiveTypeA;
  typedef decltype(SCH.objectiveB) ObjectiveTypeB;

  // We allow a few trials in case of poor convergence.
  bool success = false;
  for (size_t trial = 0; trial < 5; ++trial)
  {
    TestType coords = SCH.GetInitialPoint();
    std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = SCH.GetObjectives();

    opt.Optimize(objectives, coords);
    arma::Cube<ElemType> paretoSet= opt.ParetoSet<arma::Cube<ElemType>>();

    bool allInRange = true;

    for (size_t solutionIdx = 0; solutionIdx < paretoSet.n_slices; ++solutionIdx)
    {
      ElemType val = arma::as_scalar(paretoSet.slice(solutionIdx));
      if (!IsInBounds<ElemType>(
          val, expectedLowerBound, expectedUpperBound, 0.1))
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

TEMPLATE_TEST_CASE("DefaultMOEAD_SchafferFunctionN1Vec", "[MOEAD]",
    arma::mat, arma::fmat)
{
  typedef typename TestType::elem_type ElemType;

  // This test can be a little flaky, so we try it a few times.
  SchafferFunctionN1<TestType> SCH;
  const arma::vec lowerBound = {-1000};
  const arma::vec upperBound = {1000};
  const double expectedLowerBound = 0.0;
  const double expectedUpperBound = 2.0;

  MOEAD<Uniform, Tchebycheff> opt(
      300, // Population size.
      300,  // Max generations.
      1.0,  // Crossover probability.
      0.9, // Probability of sampling from neighbor.
      20, // Neighborhood size.
      20, // Perturbation index.
      0.5, // Differential weight.
      2, // Max childrens to replace parents.
      1e-10, // epsilon.
      lowerBound, // Lower bound.
      upperBound // Upper bound.
  );

  typedef decltype(SCH.objectiveA) ObjectiveTypeA;
  typedef decltype(SCH.objectiveB) ObjectiveTypeB;

  bool success = false;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    TestType coords = SCH.GetInitialPoint();
    std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = SCH.GetObjectives();

    opt.Optimize(objectives, coords);
    arma::Cube<ElemType> paretoSet = opt.ParetoSet<arma::Cube<ElemType>>();

    bool allInRange = true;

    for (size_t solutionIdx = 0; solutionIdx < paretoSet.n_slices; ++solutionIdx)
    {
      ElemType val = arma::as_scalar(paretoSet.slice(solutionIdx));
      if (!IsInBounds<ElemType>(
          val, expectedLowerBound, expectedUpperBound, 0.1))
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

TEMPLATE_TEST_CASE("DefaultMOEAD_FonsecaFlemingFunction", "[MOEAD]",
    arma::mat, arma::fmat)
{
  typedef typename TestType::elem_type ElemType;

  FonsecaFlemingFunction<TestType> FON;
  const double lowerBound = -4;
  const double upperBound = 4;
  const double expectedLowerBound = -1.0 / sqrt(3);
  const double expectedUpperBound = 1.0 / sqrt(3);

  MOEAD<Uniform, Tchebycheff> opt(
      300,  // Max generations.
      300,  // Max generations.
      1.0,  // Crossover probability.
      0.9, // Probability of sampling from neighbor.
      20, // Neighborhood size.
      20, // Perturbation index.
      0.5, // Differential weight.
      2, // Max childrens to replace parents.
      1e-10, // epsilon.
      lowerBound, // Lower bound.
      upperBound // Upper bound.
  );
  typedef decltype(FON.objectiveA) ObjectiveTypeA;
  typedef decltype(FON.objectiveB) ObjectiveTypeB;

  TestType coords = FON.GetInitialPoint();
  std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = FON.GetObjectives();

  opt.Optimize(objectives, coords);
  arma::Cube<ElemType> paretoSet = opt.ParetoSet<arma::Cube<ElemType>>();

  bool allInRange = true;

  for (size_t solutionIdx = 0; solutionIdx < paretoSet.n_slices; ++solutionIdx)
  {
    const TestType solution = paretoSet.slice(solutionIdx);
    ElemType valX = arma::as_scalar(solution(0));
    ElemType valY = arma::as_scalar(solution(1));
    ElemType valZ = arma::as_scalar(solution(2));

    if (!IsInBounds<ElemType>(
        valX, expectedLowerBound, expectedUpperBound, 0.1) ||
        !IsInBounds<ElemType>(
        valY, expectedLowerBound, expectedUpperBound, 0.1) ||
        !IsInBounds<ElemType>(
        valZ, expectedLowerBound, expectedUpperBound, 0.1))
    {
      allInRange = false;
      break;
    }
  }

  REQUIRE(allInRange);
}

TEMPLATE_TEST_CASE("DefaultMOEAD_FonsecaFlemingFunctionVec", "[MOEAD]",
    arma::mat, arma::fmat)
{
  typedef typename TestType::elem_type ElemType;

  FonsecaFlemingFunction<TestType> FON;
  const arma::vec lowerBound = {-4, -4, -4};
  const arma::vec upperBound = {4, 4, 4};
  const ElemType expectedLowerBound = -1.0 / sqrt(3);
  const ElemType expectedUpperBound = 1.0 / sqrt(3);

  MOEAD<Uniform, Tchebycheff> opt(
      300,  // Max generations.
      300,  // Max generations.
      1.0,  // Crossover probability.
      0.9, // Probability of sampling from neighbor.
      20, // Neighborhood size.
      20, // Perturbation index.
      0.5, // Differential weight.
      2, // Max childrens to replace parents.
      1e-10, // epsilon.
      lowerBound, // Lower bound.
      upperBound // Upper bound.
  );

  typedef decltype(FON.objectiveA) ObjectiveTypeA;
  typedef decltype(FON.objectiveB) ObjectiveTypeB;

  TestType coords = FON.GetInitialPoint();
  std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives = FON.GetObjectives();

  opt.Optimize(objectives, coords);
  arma::Cube<ElemType> paretoSet = opt.ParetoSet<arma::Cube<ElemType>>();

  bool allInRange = true;

  for (size_t solutionIdx = 0; solutionIdx < paretoSet.n_slices; ++solutionIdx)
  {
    const TestType solution = paretoSet.slice(solutionIdx);
    ElemType valX = arma::as_scalar(solution(0));
    ElemType valY = arma::as_scalar(solution(1));
    ElemType valZ = arma::as_scalar(solution(2));

    if (!IsInBounds<ElemType>(
        valX, expectedLowerBound, expectedUpperBound, 0.1) ||
        !IsInBounds<ElemType>(
        valY, expectedLowerBound, expectedUpperBound, 0.1) ||
        !IsInBounds<ElemType>(
        valZ, expectedLowerBound, expectedUpperBound, 0.1))
    {
      allInRange = false;
      break;
    }
  }

  REQUIRE(allInRange);
}

/**
 * Test DirichletMOEAD against the third problem of ZDT Test Suite. MAF-3 is a 12
 * variable-3 objective problem with disconnected Pareto Fronts.
 */
TEST_CASE("MOEADDIRICHLETMAF3Test", "[MOEAD]")
{
  //! Parameters taken from original ZDT Paper.
  MAF3<arma::mat> MAF_THREE;
  const double lowerBound = 0;
  const double upperBound = 1;
  const double expectedLowerBound = 0.5;
  const double expectedUpperBound = 0.5;

  DirichletMOEAD opt(
      105, // Population size.
      1000,  // Max generations.
      1.0,  // Crossover probability.
      0.9, // Probability of sampling from neighbor.
      20, // Neighborhood size.
      20, // Perturbation index.
      0.5, // Differential weight.
      2, // Max childrens to replace parents.
      1e-10, // epsilon.
      lowerBound, // Lower bound.
      upperBound // Upper bound.
    );

  typedef decltype(MAF_THREE.objectiveF1) ObjectiveTypeA;
  typedef decltype(MAF_THREE.objectiveF2) ObjectiveTypeB;
  typedef decltype(MAF_THREE.objectiveF3) ObjectiveTypeC;

  arma::mat coords = MAF_THREE.GetInitialPoint();
  std::tuple<ObjectiveTypeA, ObjectiveTypeB, ObjectiveTypeC> objectives =
      MAF_THREE.GetObjectives();
  opt.Optimize(objectives, coords);

  bool success = true;
  arma::cube paretoSet = opt.ParetoSet();
  for (size_t i = 0; i < paretoSet.n_slices; i++)
  {
    arma::mat solution = paretoSet.slice(i);
    bool allInRange = true;
    for (size_t j = 2; j < MAF_THREE.GetNumVariables(); j++)
    {
      double val = arma::as_scalar(solution(j));
      if (!IsInBounds<double>(val, expectedLowerBound, expectedUpperBound, 0.1))
      {
        allInRange = false;
        break;
      }
    }
    if(!allInRange)
    {
      success = false;
      break;
    }
  }
  REQUIRE(success == true);
}

/**
 * Test DirichletMOEAD against the third problem of ZDT Test Suite. MAF-1 is a 12
 * variable-3 objective problem with disconnected Pareto Fronts.
 */
TEST_CASE("MOEADDIRICHLETMAF1Test", "[MOEAD]")
{
  //! Parameters taken from original ZDT Paper.
  MAF1<arma::mat> MAF_ONE;
  const double lowerBound = 0;
  const double upperBound = 1;
  const double expectedLowerBound = 0.5;
  const double expectedUpperBound = 0.5;

  DirichletMOEAD opt(
      105, // Population size.
      1000,  // Max generations.
      1.0,  // Crossover probability.
      0.9, // Probability of sampling from neighbor.
      20, // Neighborhood size.
      20, // Perturbation index.
      0.5, // Differential weight.
      2, // Max childrens to replace parents.
      1e-10, // epsilon.
      lowerBound, // Lower bound.
      upperBound // Upper bound.
    );

  typedef decltype(MAF_ONE.objectiveF1) ObjectiveTypeA;
  typedef decltype(MAF_ONE.objectiveF2) ObjectiveTypeB;
  typedef decltype(MAF_ONE.objectiveF3) ObjectiveTypeC;

  arma::mat coords = MAF_ONE.GetInitialPoint();
  std::tuple<ObjectiveTypeA, ObjectiveTypeB, ObjectiveTypeC> objectives =
      MAF_ONE.GetObjectives();
  opt.Optimize(objectives, coords);

  bool success = true;
  arma::cube paretoSet = opt.ParetoSet();
  for (size_t i = 0; i < paretoSet.n_slices; i++)
  {
    arma::mat solution = paretoSet.slice(i);
    bool allInRange = true;
    for (size_t j = 2; j < MAF_ONE.GetNumVariables(); j++)
    {
      double val = arma::as_scalar(solution(j));
      if (!IsInBounds<double>(val, expectedLowerBound, expectedUpperBound, 0.1))
      {
        allInRange = false;
        break;
      }
    }
    if (!allInRange)
    {
      success = false;
      break;
    }
  }
  REQUIRE(success == true);
}

/**
 * Test DirichletMOEAD against the third problem of ZDT Test Suite. MAF-4 is a 12
 * variable-3 objective problem with disconnected Pareto Fronts.
 */
TEST_CASE("MOEADDIRICHLETMAF4Test", "[MOEAD]")
{
  //! Parameters taken from original ZDT Paper.
  MAF4<arma::mat> MAF_FOUR;
  const double lowerBound = 0;
  const double upperBound = 1;
  const double expectedLowerBound = 0.5;
  const double expectedUpperBound = 0.5;

  DirichletMOEAD opt(
      105, // Population size.
      1000,  // Max generations.
      1.0,  // Crossover probability.
      0.9, // Probability of sampling from neighbor.
      20, // Neighborhood size.
      20, // Perturbation index.
      0.5, // Differential weight.
      2, // Max childrens to replace parents.
      1e-10, // epsilon.
      lowerBound, // Lower bound.
      upperBound // Upper bound.
    );

  typedef decltype(MAF_FOUR.objectiveF1) ObjectiveTypeA;
  typedef decltype(MAF_FOUR.objectiveF2) ObjectiveTypeB;
  typedef decltype(MAF_FOUR.objectiveF3) ObjectiveTypeC;

  std::tuple<ObjectiveTypeA, ObjectiveTypeB, ObjectiveTypeC> objectives =
      MAF_FOUR.GetObjectives();

  bool success = false;
  arma::mat coords = MAF_FOUR.GetInitialPoint();
  opt.Optimize(objectives, coords);
  arma::cube paretoSet = opt.ParetoSet();
  for (size_t i = 0; i < paretoSet.n_slices; i++)
  {
    arma::mat solution = paretoSet.slice(i);
    bool allInRange = true;
    for (size_t j = 2; j < MAF_FOUR.GetNumVariables(); j++)
    {
      double val = arma::as_scalar(solution(j));
      if (!IsInBounds<double>(val, expectedLowerBound, expectedUpperBound, 0.2))
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
 * Test against the first problem of ZDT Test Suite.  ZDT-1 is a 30
 * variable-2 objective problem with a convex Pareto Front.
 *
 * NOTE: For the sake of runtime, only ZDT-1 is tested against the
 * algorithm. Others have been tested separately.
 *
 * We run the test multiple times, since it sometimes fails, in order to get the
 * probability of failure down.
 */
TEMPLATE_TEST_CASE("DefaultMOEAD_ZDT1Function", "[MOEAD]", arma::mat)
{
  typedef typename TestType::elem_type ElemType;

  //! Parameters taken from original ZDT Paper.
  ZDT1<TestType> ZDT_ONE(100);
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
      1e-10, // epsilon.
      lowerBound, // Lower bound.
      upperBound // Upper bound.
  );

  typedef decltype(ZDT_ONE.objectiveF1) ObjectiveTypeA;
  typedef decltype(ZDT_ONE.objectiveF2) ObjectiveTypeB;

  const size_t trials = 8;
  for (size_t trial = 0; trial < trials; ++trial)
  {
    TestType coords = ZDT_ONE.GetInitialPoint();
    std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives =
        ZDT_ONE.GetObjectives();

    opt.Optimize(objectives, coords);

    //! Refer the ZDT_ONE implementation for g objective implementation.
    //! The optimal g value is taken from the docs of ZDT_ONE.
    size_t numVariables = coords.size();
    ElemType sum = arma::accu(coords(arma::span(1, numVariables - 1), 0));
    const ElemType g = 1.0 + 9.0 * sum /
        (static_cast<ElemType>(numVariables - 1));
    if (trial < trials - 1 && g != Approx(1.0).margin(0.99))
      continue;

    REQUIRE(g == Approx(1.0).margin(0.99));
    break;
  }
}

/**
 * Test DirichletMOEAD against the third problem of ZDT Test Suite. ZDT-3 is a 30
 * variable-2 objective problem with disconnected Pareto Fronts.
 */
TEMPLATE_TEST_CASE("DirichletMOEAD_ZDT3Function", "[MOEAD]", arma::mat)
{
  typedef typename TestType::elem_type ElemType;

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
      1e-10, // epsilon.
      lowerBound, // Lower bound.
      upperBound // Upper bound.
    );

  typedef decltype(ZDT_THREE.objectiveF1) ObjectiveTypeA;
  typedef decltype(ZDT_THREE.objectiveF2) ObjectiveTypeB;

  TestType coords = ZDT_THREE.GetInitialPoint();
  std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives =
      ZDT_THREE.GetObjectives();

  opt.Optimize(objectives, coords);

  const arma::Cube<ElemType>& finalPopulation = opt.ParetoSet<
      arma::Cube<ElemType>>();
  REQUIRE(VariableBoundsCheck(finalPopulation));
}

#ifdef ENS_HAVE_COOT

TEMPLATE_TEST_CASE("MOEADDIRICHLETZDT3Test", "[MOEAD]", coot::mat, coot::fmat)
{
  typedef typename TestType::elem_type ElemType;

  //! Parameters taken from original ZDT Paper.
  ZDT3<TestType> ZDT_THREE(300);
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
      1e-10, // epsilon.
      lowerBound, // Lower bound.
      upperBound // Upper bound.
    );

  typedef decltype(ZDT_THREE.objectiveF1) ObjectiveTypeA;
  typedef decltype(ZDT_THREE.objectiveF2) ObjectiveTypeB;

  TestType coords = ZDT_THREE.GetInitialPoint();
  std::tuple<ObjectiveTypeA, ObjectiveTypeB> objectives =
      ZDT_THREE.GetObjectives();

  opt.Optimize(objectives, coords);

  const coot::Cube<ElemType>& finalPopulation = opt.ParetoSet<
      coot::Cube<ElemType>>();
  REQUIRE(VariableBoundsCheck(finalPopulation));
}

#endif
