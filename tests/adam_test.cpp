/**
 * @file adam_test.cpp
 * @author Vasanth Kalingeri
 * @author Vivek Pal
 * @author Sourabh Varshney
 * @author Haritha Nair
 * @author Marcus Edel
 * @author Conrad Sanderson
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

TEMPLATE_TEST_CASE("Adam_SphereFunction", "[Adam]", ENS_ALL_TEST_TYPES)
{
  Adam optimizer(0.5, 2, 0.7, 0.999, Tolerances<TestType>::Obj, 50000, 1e-3,
      false);
  FunctionTest<SphereFunctionType<TestType, arma::Row<size_t>>, TestType>(
      optimizer,
      10 * Tolerances<TestType>::LargeObj,
      10 * Tolerances<TestType>::LargeCoord);
}

TEMPLATE_TEST_CASE("Adam_StyblinskiTangFunction", "[Adam]", ENS_ALL_TEST_TYPES)
{
  Adam optimizer(0.5, 2, 0.7, 0.999, Tolerances<TestType>::Obj, 50000, 1e-3,
      false);
  FunctionTest<StyblinskiTangFunction<TestType, arma::Row<size_t>>, TestType>(
      optimizer,
      30 * Tolerances<TestType>::LargeObj,
      10 * Tolerances<TestType>::LargeCoord);
}

TEMPLATE_TEST_CASE("Adam_McCormickFunction", "[Adam]", ENS_ALL_TEST_TYPES)
{
  Adam optimizer(0.4, 1, 0.7, 0.999, Tolerances<TestType>::Obj, 50000, 1e-5,
      false);
  FunctionTest<McCormickFunction, TestType>(
      optimizer,
      10 * Tolerances<TestType>::LargeObj,
      10 * Tolerances<TestType>::LargeCoord);
}

TEMPLATE_TEST_CASE("Adam_MatyasFunctionFunction", "[Adam]", ENS_ALL_TEST_TYPES)
{
  Adam optimizer(0.5, 1, 0.7, 0.999, Tolerances<TestType>::Obj, 50000, 1e-5,
      false);
  FunctionTest<MatyasFunction, TestType>(
      optimizer,
      Tolerances<TestType>::LargeObj,
      Tolerances<TestType>::LargeCoord);
}

TEMPLATE_TEST_CASE("Adam_EasomFunction", "[Adam]", ENS_ALL_TEST_TYPES)
{
  Adam optimizer(0.2, 1, 0.7, 0.999, Tolerances<TestType>::Obj, 50000, 1e-5,
      false);
  FunctionTest<EasomFunction, TestType>(
      optimizer,
      3 * Tolerances<TestType>::LargeObj,
      Tolerances<TestType>::LargeCoord);
}

TEMPLATE_TEST_CASE("Adam_BoothFunction", "[Adam]", ENS_ALL_TEST_TYPES)
{
  Adam optimizer(0.75, 1, 0.7, 0.999, Tolerances<TestType>::Obj, 50000, 1e-9,
      true);
  FunctionTest<BoothFunction, TestType>(
      optimizer,
      Tolerances<TestType>::LargeObj,
      Tolerances<TestType>::LargeCoord);
}

TEMPLATE_TEST_CASE("AMSGrad_SphereFunction", "[Adam]", ENS_ALL_TEST_TYPES)
{
  AMSGrad optimizer(0.01, 1, 0.9, 0.999, Tolerances<TestType>::Obj, 50000,
      Tolerances<TestType>::Obj / 100, true);
  FunctionTest<SphereFunctionType<TestType, arma::Row<size_t>>, TestType>(
      optimizer,
      10 * Tolerances<TestType>::LargeObj,
      10 * Tolerances<TestType>::LargeCoord);
}

TEMPLATE_TEST_CASE("Adam_LogisticRegressionFunction", "[Adam]",
    ENS_ALL_TEST_TYPES)
{
  Adam adam;
  LogisticRegressionFunctionTest<TestType>(adam);
}

TEMPLATE_TEST_CASE("AdaMax_LogisticRegressionFunction", "[Adam]",
    ENS_ALL_TEST_TYPES)
{
  AdaMax adamax(1e-3, 1, 0.9, 0.999, Tolerances<TestType>::Obj, 50000,
      Tolerances<TestType>::Obj / 10, true);
  LogisticRegressionFunctionTest<TestType>(adamax);
}

TEMPLATE_TEST_CASE("AMSGrad_LogisticRegressionFunction", "[Adam]",
    ENS_ALL_TEST_TYPES)
{
  AMSGrad amsgrad(1e-3, 1, 0.9, 0.999, Tolerances<TestType>::Obj, 50000,
      Tolerances<TestType>::Obj / 100, true);
  LogisticRegressionFunctionTest<TestType>(amsgrad);
}

TEMPLATE_TEST_CASE("Nadam_LogisticRegressionFunction", "[Adam]",
    ENS_ALL_TEST_TYPES)
{
  Nadam nadam;
  LogisticRegressionFunctionTest<TestType>(nadam);
}

TEMPLATE_TEST_CASE("NadaMax_LogisticRegressionFunction", "[Adam]",
    ENS_ALL_TEST_TYPES)
{
  NadaMax nadamax(1e-3, 1, 0.9, 0.999, Tolerances<TestType>::Obj, 50000,
      Tolerances<TestType>::Obj / 10, true);
  LogisticRegressionFunctionTest<TestType>(nadamax);
}

TEMPLATE_TEST_CASE("OptimisticAdam_LogisticRegressionFunction", "[Adam]",
    ENS_ALL_TEST_TYPES)
{
  OptimisticAdam optimisticAdam;
  LogisticRegressionFunctionTest<TestType>(optimisticAdam);
}

TEMPLATE_TEST_CASE("Padam_LogisticRegressionFunction", "[Adam]",
    ENS_ALL_TEST_TYPES)
{
  Padam optimizer;
  LogisticRegressionFunctionTest<TestType>(optimizer);
}

TEMPLATE_TEST_CASE("QHAdam_LogisticRegressionFunction", "[Adam]",
    ENS_ALL_TEST_TYPES)
{
  QHAdam optimizer;
  LogisticRegressionFunctionTest<TestType>(optimizer);
}

TEMPLATE_TEST_CASE("Adam_AckleyFunction", "[Adam]", ENS_ALL_TEST_TYPES)
{
  Adam optimizer(0.01, 2, 0.7, 0.999, 100 * Tolerances<TestType>::Obj, 50000,
      Tolerances<TestType>::Obj, false);
  FunctionTest<AckleyFunction, TestType>(
      optimizer,
      Tolerances<TestType>::LargeObj,
      Tolerances<TestType>::LargeCoord);
}

TEMPLATE_TEST_CASE("Adam_BealeFunction", "[Adam]", ENS_ALL_TEST_TYPES)
{
  Adam optimizer(0.005, 2, 0.7, 0.999, 10 * Tolerances<TestType>::Obj, 50000,
      1e-8, false);
  FunctionTest<BealeFunction, TestType>(
      optimizer,
      3 * Tolerances<TestType>::LargeObj,
      3 * Tolerances<TestType>::LargeCoord);
}

// FP16 cannot be used for this test because the initial gradient is too large.
TEMPLATE_TEST_CASE("Adam_GoldsteinPriceFunction", "[Adam]", ENS_TEST_TYPES)
{
  Adam optimizer(0.1, 2, 0.7, 0.999, 100 * Tolerances<TestType>::Obj, 50000,
      1e-8, false);
  FunctionTest<GoldsteinPriceFunction, TestType>(
      optimizer,
      Tolerances<TestType>::LargeObj,
      Tolerances<TestType>::LargeCoord);
}

TEMPLATE_TEST_CASE("Adam_LevyFunction", "[Adam]", ENS_ALL_TEST_TYPES)
{
  Adam optimizer(0.0025, 2, 0.7, 0.999, Tolerances<TestType>::Obj, 50000, 1e-8,
      false);
  FunctionTest<LevyFunctionN13, TestType>(
      optimizer,
      Tolerances<TestType>::LargeObj,
      Tolerances<TestType>::LargeCoord);
}

TEMPLATE_TEST_CASE("Adam_HimmelblauFunction", "[Adam]", ENS_ALL_TEST_TYPES)
{
  HimmelblauFunction f;
  Adam optimizer(0.0025, 2, 0.7, 0.999, Tolerances<TestType>::Obj, 50000, 1e-8,
      false);

  TestType coordinates = TestType("2.9; 1.9");
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(3.0).margin(0.05));
  REQUIRE(coordinates(1) == Approx(2.0).margin(0.05));
}

TEMPLATE_TEST_CASE("Adam_ThreeHumpCamelFunction", "[Adam]", ENS_ALL_TEST_TYPES)
{
  Adam optimizer(0.0025, 2, 0.7, 0.999, Tolerances<TestType>::Obj, 50000, 1e-8,
      false);
  FunctionTest<ThreeHumpCamelFunction, TestType>(
      optimizer,
      Tolerances<TestType>::LargeObj,
      Tolerances<TestType>::LargeCoord);
}

#ifdef USE_COOT

TEMPLATE_TEST_CASE("Adam_SphereFunction", "[Adam]", coot::mat, coot::fmat)
{
  Adam optimizer(0.5, 2, 0.7, 0.999, 1e-8, 50000, 1e-3, false);
  FunctionTest<SphereFunctionType<TestType, coot::Row<size_t>>, TestType>(
      optimizer, 0.5, 0.2);
}

TEMPLATE_TEST_CASE("Adam_StyblinskiTangFunction", "[Adam]", coot::mat)
{
  Adam optimizer(0.5, 2, 0.7, 0.999, 1e-8, 50000, 1e-3, false);
  FunctionTest<StyblinskiTangFunction<TestType, coot::Row<size_t>>, TestType>(
      optimizer, 0.5, 0.1);
}

TEMPLATE_TEST_CASE("Adam_McCormickFunction", "[Adam]", coot::mat)
{
  Adam optimizer(0.5, 1, 0.7, 0.999, 1e-8, 50000, 1e-5, false);
  FunctionTest<McCormickFunction, TestType>(optimizer, 0.5, 0.1);
}

TEMPLATE_TEST_CASE("Adam_MatyasFunction", "[Adam]", coot::mat)
{
  Adam optimizer(0.5, 1, 0.7, 0.999, 1e-8, 50000, 1e-5, false);
  FunctionTest<MatyasFunction, TestType>(optimizer, 0.1, 0.01);
}

TEMPLATE_TEST_CASE("Adam_EasomFunction", "[Adam]", coot::mat)
{
  Adam optimizer(0.2, 1, 0.7, 0.999, 1e-8, 50000, 1e-5, false);
  FunctionTest<EasomFunction, TestType>(optimizer, 1.5, 0.01);
}

TEMPLATE_TEST_CASE("Adam_BoothFunction", "[Adam]", coot::mat)
{
  Adam optimizer(1e-1, 1, 0.7, 0.999, 1e-8, 50000, 1e-9, true);
  FunctionTest<BoothFunction, TestType>(
      optimizer,
      50 * Tolerances<TestType>::Obj,
      10 * Tolerances<TestType>::Coord);
}

TEMPLATE_TEST_CASE("Adam_LogisticRegressionFunction", "[Adam]", coot::mat)
{
  Adam adam;
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(adam);
}

TEMPLATE_TEST_CASE("AdaMax_LogisticRegressionFunction", "[Adam]", coot::mat)
{
  AdaMax adamax(0.01, 8, 0.9, 0.999, 1e-8, 50000, 1e-9, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(adamax);
}

TEMPLATE_TEST_CASE("AMSGrad_LogisticRegressionFunction", "[Adam]", coot::mat)
{
  AMSGrad amsgrad(0.01, 8, 0.9, 0.999, 1e-8, 50000, 1e-11, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(amsgrad);
}

TEMPLATE_TEST_CASE("Nadam_LogisticRegressionFunction", "[Adam]", coot::mat)
{
  Nadam nadam;
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(nadam);
}

TEMPLATE_TEST_CASE("NadaMax_LogisticRegressionFunction", "[Adam]", coot::mat)
{
  NadaMax nadamax(0.01, 8, 0.9, 0.999, 1e-8, 50000, 1e-9, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(nadamax);
}

TEMPLATE_TEST_CASE("OptimisticAdam_LogisticRegressionFunction", "[Adam]",
    coot::mat)
{
  OptimisticAdam optimisticAdam;
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(optimisticAdam);
}

TEMPLATE_TEST_CASE("Padam_LogisticRegressionFunction", "[Adam]", coot::mat)
{
  Padam optimizer;
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(optimizer);
}

TEMPLATE_TEST_CASE("QHAdam_LogisticRegressionFunction", "[Adam]",
    coot::mat, coot::fmat)
{
  QHAdam optimizer;
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(optimizer);
}

 #endif
