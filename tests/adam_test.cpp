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

TEMPLATE_TEST_CASE("Adam_SphereFunction", "[Adam]", arma::mat, arma::fmat)
{
  Adam optimizer(0.5, 2, 0.7, 0.999, 1e-8, 50000, 1e-3, false);
  FunctionTest<SphereFunctionType<TestType, arma::Row<size_t>>, TestType>(
      optimizer, 0.5, 0.2);
}

TEMPLATE_TEST_CASE("Adam_StyblinskiTangFunction", "[Adam]", arma::mat)
{
  Adam optimizer(0.5, 2, 0.7, 0.999, 1e-8, 50000, 1e-3, false);
  FunctionTest<StyblinskiTangFunction<TestType, arma::Row<size_t>>, TestType>(
      optimizer, 0.5, 0.1);
}

TEMPLATE_TEST_CASE("Adam_McCormickFunctionFunction", "[Adam]", arma::mat)
{
  Adam optimizer(0.5, 1, 0.7, 0.999, 1e-8, 50000, 1e-5, false);
  FunctionTest<McCormickFunction, TestType>(optimizer, 0.5, 0.1);
}

TEMPLATE_TEST_CASE("Adam_MatyasFunctionFunction", "[Adam]", arma::mat)
{
  Adam optimizer(0.5, 1, 0.7, 0.999, 1e-8, 50000, 1e-5, false);
  FunctionTest<MatyasFunction, TestType>(optimizer, 0.1, 0.01);
}

TEMPLATE_TEST_CASE("Adam_EasomFunction", "[Adam]", arma::mat)
{
  Adam optimizer(0.2, 1, 0.7, 0.999, 1e-8, 50000, 1e-5, false);
  FunctionTest<EasomFunction, TestType>(optimizer, 1.5, 0.01);
}

TEMPLATE_TEST_CASE("Adam_BoothFunction", "[Adam]", arma::mat)
{
  Adam optimizer(1e-1, 1, 0.7, 0.999, 1e-8, 50000, 1e-9, true);
  FunctionTest<BoothFunction, TestType>(optimizer);
}

TEMPLATE_TEST_CASE("AMSGrad_SphereFunction", "[Adam]", arma::fmat)
{
  AMSGrad optimizer(1e-3, 1, 0.9, 0.999, 1e-8, 50000, 1e-11, true);
  FunctionTest<SphereFunctionType<TestType, arma::Row<size_t>>, TestType>(
      optimizer, 0.5, 0.1);
}

TEMPLATE_TEST_CASE("Adam_LogisticRegressionFunction", "[Adam]", arma::mat)
{
  Adam adam;
  LogisticRegressionFunctionTest<TestType>(adam, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("AdaMax_LogisticRegressionFunction", "[Adam]", arma::mat)
{
  AdaMax adamax(1e-3, 1, 0.9, 0.999, 1e-8, 50000, 1e-9, true);
  LogisticRegressionFunctionTest<TestType>(adamax, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("AMSGrad_LogisticRegressionFunction", "[Adam]", arma::mat)
{
  AMSGrad amsgrad(1e-3, 1, 0.9, 0.999, 1e-8, 50000, 1e-11, true);
  LogisticRegressionFunctionTest<TestType>(amsgrad, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("Nadam_LogisticRegressionFunction", "[Adam]", arma::mat)
{
  Nadam nadam;
  LogisticRegressionFunctionTest<TestType>(nadam, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("NadaMax_LogisticRegressionFunction", "[Adam]", arma::mat)
{
  NadaMax nadamax(1e-3, 1, 0.9, 0.999, 1e-8, 50000, 1e-9, true);
  LogisticRegressionFunctionTest<TestType>(nadamax, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("OptimisticAdam_LogisticRegressionFunction", "[Adam]",
    arma::mat)
{
  OptimisticAdam optimisticAdam;
  LogisticRegressionFunctionTest<TestType>(optimisticAdam, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("Padam_LogisticRegressionFunction", "[Adam]", arma::mat)
{
  Padam optimizer;
  LogisticRegressionFunctionTest<TestType>(optimizer, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("QHAdam_LogisticRegressionFunction", "[Adam]",
    arma::mat, arma::fmat)
{
  QHAdam optimizer;
  LogisticRegressionFunctionTest<TestType>(optimizer, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("Adam_AckleyFunction", "[Adam]", arma::mat)
{
  Adam optimizer(0.001, 2, 0.7, 0.999, 1e-8, 50000, 1e-7, false);
  FunctionTest<AckleyFunction, TestType>(optimizer);
}

TEMPLATE_TEST_CASE("Adam_BealeFunction", "[Adam]", arma::mat)
{
  Adam optimizer(0.001, 2, 0.7, 0.999, 1e-8, 50000, 1e-7, false);
  FunctionTest<BealeFunction, TestType>(optimizer, 0.1, 0.01);
}

TEMPLATE_TEST_CASE("Adam_GoldsteinPriceFunction", "[Adam]", arma::mat)
{
  Adam optimizer(0.0001, 2, 0.7, 0.999, 1e-8, 50000, 1e-9, false);
  FunctionTest<GoldsteinPriceFunction, TestType>(optimizer, 0.1, 0.01);
}

TEMPLATE_TEST_CASE("Adam_LevyFunction", "[Adam]", arma::mat)
{
  Adam optimizer(0.001, 2, 0.7, 0.999, 1e-8, 50000, 1e-9, false);
  FunctionTest<LevyFunctionN13, TestType>(optimizer, 0.1, 0.01);
}

TEMPLATE_TEST_CASE("Adam_HimmelblauFunction", "[Adam]", arma::mat)
{
  HimmelblauFunction f;
  Adam optimizer(0.001, 2, 0.7, 0.999, 1e-8, 50000, 1e-9, false);

  TestType coordinates = TestType("2.9; 1.9");
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(3.0).margin(0.05));
  REQUIRE(coordinates(1) == Approx(2.0).margin(0.05));
}

TEMPLATE_TEST_CASE("Adam_ThreeHumpCamelFunction", "[Adam]", arma::mat)
{
  Adam optimizer(0.001, 2, 0.7, 0.999, 1e-8, 50000, 1e-9, false);
  FunctionTest<ThreeHumpCamelFunction, TestType>(optimizer, 0.1, 0.01);
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
  FunctionTest<BoothFunction, TestType>(optimizer);
}

TEMPLATE_TEST_CASE("Adam_LogisticRegressionFunction", "[Adam]", coot::mat)
{
  Adam adam;
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      adam, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("AdaMax_LogisticRegressionFunction", "[Adam]", coot::mat)
{
  AdaMax adamax(0.01, 8, 0.9, 0.999, 1e-8, 50000, 1e-9, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      adamax, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("AMSGrad_LogisticRegressionFunction", "[Adam]", coot::mat)
{
  AMSGrad amsgrad(0.01, 8, 0.9, 0.999, 1e-8, 50000, 1e-11, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      amsgrad, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("Nadam_LogisticRegressionFunction", "[Adam]", coot::mat)
{
  Nadam nadam;
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      nadam, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("NadaMax_LogisticRegressionFunction", "[Adam]", coot::mat)
{
  NadaMax nadamax(0.01, 8, 0.9, 0.999, 1e-8, 50000, 1e-9, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      nadamax, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("OptimisticAdam_LogisticRegressionFunction", "[Adam]",
    coot::mat)
{
  OptimisticAdam optimisticAdam;
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      optimisticAdam, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("Padam_LogisticRegressionFunction", "[Adam]", coot::mat)
{
  Padam optimizer;
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      optimizer, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("QHAdam_LogisticRegressionFunction", "[Adam]",
    coot::mat, coot::fmat)
{
  QHAdam optimizer;
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      optimizer, 0.003, 0.006);
}

 #endif