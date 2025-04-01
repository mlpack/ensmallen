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

TEMPLATE_TEST_CASE("AdamSphereFunctionTest", "[Adam]", arma::mat, arma::fmat)
{
  Adam optimizer(0.5, 2, 0.7, 0.999, 1e-8, 500000, 1e-3, false);
  FunctionTest<SphereFunction<TestType, arma::Row<size_t>>, TestType>(
      optimizer, 0.5, 0.2);
}

TEMPLATE_TEST_CASE("AdamStyblinskiTangFunctionTest", "[Adam]", arma::mat)
{
  Adam optimizer(0.5, 2, 0.7, 0.999, 1e-8, 500000, 1e-3, false);
  FunctionTest<StyblinskiTangFunction<TestType, arma::Row<size_t>>, TestType>(
      optimizer, 0.5, 0.1);
}

TEMPLATE_TEST_CASE("AdamMcCormickFunctionTest", "[Adam]", arma::mat)
{
  Adam optimizer(0.5, 1, 0.7, 0.999, 1e-8, 500000, 1e-5, false);
  FunctionTest<McCormickFunction, TestType>(optimizer, 0.5, 0.1);
}

TEMPLATE_TEST_CASE("AdamMatyasFunctionTest", "[Adam]", arma::mat)
{
  Adam optimizer(0.5, 1, 0.7, 0.999, 1e-8, 500000, 1e-5, false);
  FunctionTest<MatyasFunction, TestType>(optimizer, 0.1, 0.01);
}

TEMPLATE_TEST_CASE("AdamEasomFunctionTest", "[Adam]", arma::mat)
{
  Adam optimizer(0.2, 1, 0.7, 0.999, 1e-8, 500000, 1e-5, false);
  FunctionTest<EasomFunction, TestType>(optimizer, 1.5, 0.01);
}

TEMPLATE_TEST_CASE("AdamBoothFunctionTest", "[Adam]", arma::mat)
{
  Adam optimizer(1e-1, 1, 0.7, 0.999, 1e-8, 500000, 1e-9, true);
  FunctionTest<BoothFunction, TestType>(optimizer);
}

TEMPLATE_TEST_CASE("AMSGradSphereFunctionTestFMat", "[Adam]", arma::fmat)
{
  AMSGrad optimizer(1e-3, 1, 0.9, 0.999, 1e-8, 500000, 1e-11, true);
  FunctionTest<SphereFunction<TestType, arma::Row<size_t>>, TestType>(
      optimizer, 0.5, 0.1);
}

TEMPLATE_TEST_CASE("AdamLogisticRegressionTest", "[Adam]", arma::mat)
{
  Adam adam;
  LogisticRegressionFunctionTest<TestType>(adam, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("AdaMaxLogisticRegressionTest", "[Adam]", arma::mat)
{
  AdaMax adamax(1e-3, 1, 0.9, 0.999, 1e-8, 5000000, 1e-9, true);
  LogisticRegressionFunctionTest<TestType>(adamax, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("AMSGradLogisticRegressionTest", "[Adam]", arma::mat)
{
  AMSGrad amsgrad(1e-3, 1, 0.9, 0.999, 1e-8, 500000, 1e-11, true);
  LogisticRegressionFunctionTest<TestType>(amsgrad, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("NadamLogisticRegressionTest", "[Adam]", arma::mat)
{
  Nadam nadam;
  LogisticRegressionFunctionTest<TestType>(nadam, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("NadaMaxLogisticRegressionTest", "[Adam]", arma::mat)
{
  NadaMax nadamax(1e-3, 1, 0.9, 0.999, 1e-8, 5000000, 1e-9, true);
  LogisticRegressionFunctionTest<TestType>(nadamax, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("OptimisticAdamLogisticRegressionTest", "[Adam]", arma::mat)
{
  OptimisticAdam optimisticAdam;
  LogisticRegressionFunctionTest<TestType>(optimisticAdam, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("PadamLogisticRegressionTest", "[Adam]", arma::mat)
{
  Padam optimizer;
  LogisticRegressionFunctionTest<TestType>(optimizer, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("QHAdamLogisticRegressionTest", "[Adam]",
    arma::mat, arma::fmat)
{
  QHAdam optimizer;
  LogisticRegressionFunctionTest<TestType>(optimizer, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("AdamAckleyFunctionTest", "[Adam]", arma::mat)
{
  Adam optimizer(0.001, 2, 0.7, 0.999, 1e-8, 500000, 1e-7, false);
  FunctionTest<AckleyFunction, TestType>(optimizer);
}

TEMPLATE_TEST_CASE("AdamBealeFunctionTest", "[Adam]", arma::mat)
{
  Adam optimizer(0.001, 2, 0.7, 0.999, 1e-8, 500000, 1e-7, false);
  FunctionTest<BealeFunction, TestType>(optimizer, 0.1, 0.01);
}

TEMPLATE_TEST_CASE("AdamGoldsteinPriceFunctionTest", "[Adam]", arma::mat)
{
  Adam optimizer(0.0001, 2, 0.7, 0.999, 1e-8, 500000, 1e-9, false);
  FunctionTest<GoldsteinPriceFunction, TestType>(optimizer, 0.1, 0.01);
}

TEMPLATE_TEST_CASE("AdamLevyFunctionTest", "[Adam]", arma::mat)
{
  Adam optimizer(0.001, 2, 0.7, 0.999, 1e-8, 500000, 1e-9, false);
  FunctionTest<LevyFunctionN13, TestType>(optimizer, 0.1, 0.01);
}

TEMPLATE_TEST_CASE("AdamHimmelblauFunctionTest", "[Adam]", arma::mat)
{
  HimmelblauFunction f;
  Adam optimizer(0.001, 2, 0.7, 0.999, 1e-8, 500000, 1e-9, false);

  arma::mat coordinates = arma::mat("2.9; 1.9");
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(3.0).margin(0.05));
  REQUIRE(coordinates(1) == Approx(2.0).margin(0.05));
}

TEMPLATE_TEST_CASE("AdamThreeHumpCamelFunctionTest", "[Adam]", arma::mat)
{
  Adam optimizer(0.001, 2, 0.7, 0.999, 1e-8, 500000, 1e-9, false);
  FunctionTest<ThreeHumpCamelFunction, TestType>(optimizer, 0.1, 0.01);
}

#if ARMA_VERSION_MAJOR > 9 ||\
    (ARMA_VERSION_MAJOR == 9 && ARMA_VERSION_MINOR >= 400)

TEMPLATE_TEST_CASE("AdamSphereFunctionTestSpMat", "[Adam]", arma::sp_mat)
{
  Adam optimizer(0.5, 2, 0.7, 0.999, 1e-8, 500000, 1e-3, false);
  FunctionTest<SphereFunction<TestType, arma::Row<size_t>>, TestType>(
      optimizer, 0.5, 0.2);
}

TEST_CASE("AdamSphereFunctionTestSpMatDenseGradient", "[Adam]")
{
  SphereFunction<arma::mat, arma::Row<size_t>> f(2);
  Adam optimizer(0.5, 2, 0.7, 0.999, 1e-8, 500000, 1e-3, false);

  arma::sp_mat coordinates = f.GetInitialPoint<arma::sp_mat>();
  optimizer.Optimize<decltype(f), arma::sp_mat, arma::mat>(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
}

TEST_CASE("AMSGradSphereFunctionTestSpMat", "[Adam]")
{
  AMSGrad optimizer(1e-3, 1, 0.9, 0.999, 1e-8, 500000, 1e-11, true);
  FunctionTest<SphereFunction<arma::sp_mat, arma::Row<size_t>>,
      arma::sp_mat>(optimizer, 0.5, 0.1);
}

TEST_CASE("AMSGradSphereFunctionTestSpMatDenseGradient", "[Adam]")
{
  SphereFunction<arma::sp_mat, arma::Row<size_t>> f(2);
  AMSGrad optimizer(1e-3, 1, 0.9, 0.999, 1e-8, 500000, 1e-11, true);

  arma::sp_mat coordinates = f.GetInitialPoint<arma::sp_mat>();
  optimizer.Optimize<decltype(f), arma::sp_mat, arma::mat>(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
}

TEST_CASE("QHAdamLogisticRegressionSpMatTest", "[Adam]")
{
  QHAdam optimizer;
  LogisticRegressionFunctionTest<arma::sp_mat>(optimizer, 0.003, 0.006);
}

#endif

#ifdef ENS_HAS_COOT

TEMPLATE_TEST_CASE("AdamSphereFunctionTest", "[Adam]", coot::mat, coot::fmat)
{
  Adam optimizer(0.5, 2, 0.7, 0.999, 1e-8, 500000, 1e-3, false);
  FunctionTest<SphereFunction<TestType, coot::Row<size_t>>, TestType>(
      optimizer, 0.5, 0.2);
}

TEMPLATE_TEST_CASE("AdamStyblinskiTangFunctionTest", "[Adam]", coot::mat)
{
  Adam optimizer(0.5, 2, 0.7, 0.999, 1e-8, 500000, 1e-3, false);
  FunctionTest<StyblinskiTangFunction<TestType, coot::Row<size_t>>, TestType>(
      optimizer, 0.5, 0.1);
}

TEMPLATE_TEST_CASE("AdamMcCormickFunctionTest", "[Adam]", coot::mat)
{
  Adam optimizer(0.5, 1, 0.7, 0.999, 1e-8, 500000, 1e-5, false);
  FunctionTest<McCormickFunction, TestType>(optimizer, 0.5, 0.1);
}

TEMPLATE_TEST_CASE("AdamMatyasFunctionTest", "[Adam]", coot::mat)
{
  Adam optimizer(0.5, 1, 0.7, 0.999, 1e-8, 500000, 1e-5, false);
  FunctionTest<MatyasFunction, TestType>(optimizer, 0.1, 0.01);
}

TEMPLATE_TEST_CASE("AdamEasomFunctionTest", "[Adam]", coot::mat)
{
  Adam optimizer(0.2, 1, 0.7, 0.999, 1e-8, 500000, 1e-5, false);
  FunctionTest<EasomFunction, TestType>(optimizer, 1.5, 0.01);
}

TEMPLATE_TEST_CASE("AdamBoothFunctionTest", "[Adam]", coot::mat)
{
  Adam optimizer(1e-1, 1, 0.7, 0.999, 1e-8, 500000, 1e-9, true);
  FunctionTest<BoothFunction, TestType>(optimizer);
}

TEMPLATE_TEST_CASE("AMSGradSphereFunctionTestFMat", "[Adam]", coot::fmat)
{
  AMSGrad optimizer(1e-3, 1, 0.9, 0.999, 1e-8, 500000, 1e-11, true);
  FunctionTest<SphereFunction<TestType, coot::Row<size_t>>, TestType>(
      optimizer, 0.5, 0.1);
}

TEMPLATE_TEST_CASE("AdamLogisticRegressionTest", "[Adam]", coot::mat)
{
  Adam adam;
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      adam, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("AdaMaxLogisticRegressionTest", "[Adam]", coot::mat)
{
  AdaMax adamax(1e-3, 1, 0.9, 0.999, 1e-8, 5000000, 1e-9, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      adamax, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("AMSGradLogisticRegressionTest", "[Adam]", coot::mat)
{
  AMSGrad amsgrad(1e-3, 1, 0.9, 0.999, 1e-8, 500000, 1e-11, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      amsgrad, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("NadamLogisticRegressionTest", "[Adam]", coot::mat)
{
  Nadam nadam;
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      nadam, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("NadaMaxLogisticRegressionTest", "[Adam]", coot::mat)
{
  NadaMax nadamax(1e-3, 1, 0.9, 0.999, 1e-8, 5000000, 1e-9, true);
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      nadamax, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("OptimisticAdamLogisticRegressionTest", "[Adam]", coot::mat)
{
  OptimisticAdam optimisticAdam;
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      optimisticAdam, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("PadamLogisticRegressionTest", "[Adam]", coot::mat)
{
  Padam optimizer;
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      optimizer, 0.003, 0.006);
}

TEMPLATE_TEST_CASE("QHAdamLogisticRegressionTest", "[Adam]",
    coot::mat, coot::fmat)
{
  QHAdam optimizer;
  LogisticRegressionFunctionTest<TestType, coot::Row<size_t>>(
      optimizer, 0.003, 0.006);
}

#endif