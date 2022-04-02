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

/**
 * Test the Adam optimizer on the Sphere function.
 */
TEST_CASE("AdamSphereFunctionTest", "[AdamTest]")
{
  Adam optimizer(0.5, 2, 0.7, 0.999, 1e-8, 500000, 1e-3, false);
  FunctionTest<SphereFunction>(optimizer, 0.5, 0.2);
}

/**
 * Test the Adam optimizer on the Sphere function with arma::fmat.
 */
TEST_CASE("AdamSphereFunctionTestFMat", "[AdamTest]")
{
  Adam optimizer(0.5, 2, 0.7, 0.999, 1e-8, 500000, 1e-3, false);
  FunctionTest<SphereFunction, arma::fmat>(optimizer, 0.5, 0.2);
}

#if ARMA_VERSION_MAJOR > 9 ||\
    (ARMA_VERSION_MAJOR == 9 && ARMA_VERSION_MINOR >= 400)

/**
 * Test the AMSGrad optimizer on the Sphere function with arma::sp_mat.
 */
TEST_CASE("AdamSphereFunctionTestSpMat", "[AdamTest]")
{
  Adam optimizer(0.5, 2, 0.7, 0.999, 1e-8, 500000, 1e-3, false);
  FunctionTest<SphereFunction, arma::sp_mat>(optimizer, 0.5, 0.2);
}

/**
 * Test the AMSGrad optimizer on the Sphere function with arma::sp_mat but a
 * dense (arma::mat) gradient.
 */
TEST_CASE("AdamSphereFunctionTestSpMatDenseGradient", "[AdamTest]")
{
  SphereFunction f(2);
  Adam optimizer(0.5, 2, 0.7, 0.999, 1e-8, 500000, 1e-3, false);

  arma::sp_mat coordinates = f.GetInitialPoint<arma::sp_mat>();
  optimizer.Optimize<decltype(f), arma::sp_mat, arma::mat>(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
}

#endif

/**
 * Test the Adam optimizer on the Wood function.
 */
TEST_CASE("AdamStyblinskiTangFunctionTest", "[AdamTest]")
{
  Adam optimizer(0.5, 2, 0.7, 0.999, 1e-8, 500000, 1e-3, false);
  FunctionTest<StyblinskiTangFunction>(optimizer, 0.5, 0.1);
}

/**
 * Test the Adam optimizer on the McCormick function.
 */
TEST_CASE("AdamMcCormickFunctionTest", "[AdamTest]")
{
  Adam optimizer(0.5, 1, 0.7, 0.999, 1e-8, 500000, 1e-5, false);
  FunctionTest<McCormickFunction>(optimizer, 0.5, 0.1);
}

/**
 * Test the Adam optimizer on the Matyas function.
 */
TEST_CASE("AdamMatyasFunctionTest", "[AdamTest]")
{
  Adam optimizer(0.5, 1, 0.7, 0.999, 1e-8, 500000, 1e-5, false);
  FunctionTest<MatyasFunction>(optimizer, 0.1, 0.01);
}

/**
 * Test the Adam optimizer on the Easom function.
 */
TEST_CASE("AdamEasomFunctionTest", "[AdamTest]")
{
  Adam optimizer(0.2, 1, 0.7, 0.999, 1e-8, 500000, 1e-5, false);
  FunctionTest<EasomFunction>(optimizer, 1.5, 0.01);
}

/**
 * Test the Adam optimizer on the Booth function.
 */
TEST_CASE("AdamBoothFunctionTest", "[AdamTest]")
{
  Adam optimizer(1e-1, 1, 0.7, 0.999, 1e-8, 500000, 1e-9, true);
  FunctionTest<BoothFunction>(optimizer);
}


/**
 * Test the AMSGrad optimizer on the Sphere function with arma::fmat.
 */
TEST_CASE("AMSGradSphereFunctionTestFMat", "[AdamTest]")
{
  AMSGrad optimizer(1e-3, 1, 0.9, 0.999, 1e-8, 500000, 1e-11, true);
  FunctionTest<SphereFunction, arma::fmat>(optimizer, 0.5, 0.1);
}

#if ARMA_VERSION_MAJOR > 9 || \
    (ARMA_VERSION_MAJOR == 9 && ARMA_VERSION_MINOR >= 400)

/**
 * Test the AMSGrad optimizer on the Sphere function with arma::sp_mat.
 */
TEST_CASE("AMSGradSphereFunctionTestSpMat", "[AdamTest]")
{
  AMSGrad optimizer(1e-3, 1, 0.9, 0.999, 1e-8, 500000, 1e-11, true);
  FunctionTest<SphereFunction, arma::sp_mat>(optimizer, 0.5, 0.1);
}

/**
 * Test the AMSGrad optimizer on the Sphere function with arma::sp_mat but a
 * dense (arma::mat) gradient.
 */
TEST_CASE("AMSGradSphereFunctionTestSpMatDenseGradient", "[AdamTest]")
{
  SphereFunction f(2);
  AMSGrad optimizer(1e-3, 1, 0.9, 0.999, 1e-8, 500000, 1e-11, true);

  arma::sp_mat coordinates = f.GetInitialPoint<arma::sp_mat>();
  optimizer.Optimize<decltype(f), arma::sp_mat, arma::mat>(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
}

#endif

/**
 * Run Adam on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("AdamLogisticRegressionTest", "[AdamTest]")
{
  Adam adam;
  LogisticRegressionFunctionTest(adam, 0.003, 0.006);
}

/**
 * Run AdaMax on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("AdaMaxLogisticRegressionTest", "[AdamTest]")
{
  AdaMax adamax(1e-3, 1, 0.9, 0.999, 1e-8, 5000000, 1e-9, true);
  LogisticRegressionFunctionTest(adamax, 0.003, 0.006);
}

/**
 * Run AMSGrad on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("AMSGradLogisticRegressionTest", "[AdamTest]")
{
  AMSGrad amsgrad(1e-3, 1, 0.9, 0.999, 1e-8, 500000, 1e-11, true);
  LogisticRegressionFunctionTest(amsgrad, 0.003, 0.006);
}

/**
 * Run Nadam on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("NadamLogisticRegressionTest", "[AdamTest]")
{
  Nadam nadam;
  LogisticRegressionFunctionTest(nadam, 0.003, 0.006);
}

/**
 * Run NadaMax on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("NadaMaxLogisticRegressionTest", "[AdamTest]")
{
  NadaMax nadamax(1e-3, 1, 0.9, 0.999, 1e-8, 5000000, 1e-9, true);
  LogisticRegressionFunctionTest(nadamax, 0.003, 0.006);
}

/**
 * Run OptimisticAdam on logistic regression and make sure the results are
 * acceptable.
 */
TEST_CASE("OptimisticAdamLogisticRegressionTest", "[AdamTest]")
{
  OptimisticAdam optimisticAdam;
  LogisticRegressionFunctionTest(optimisticAdam, 0.003, 0.006);
}

/**
 * Run Padam on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("PadamLogisticRegressionTest", "[AdamTest]")
{
  Padam optimizer;
  LogisticRegressionFunctionTest(optimizer, 0.003, 0.006);
}

/**
 * Run QHAdam on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("QHAdamLogisticRegressionTest", "[AdamTest]")
{
  QHAdam optimizer;
  LogisticRegressionFunctionTest(optimizer, 0.003, 0.006);
}

/**
 * Run QHAdam on logistic regression and make sure the results are acceptable,
 * using arma::fmat.
 */
TEST_CASE("QHAdamLogisticRegressionFMatTest", "[AdamTest]")
{
  QHAdam optimizer;
  LogisticRegressionFunctionTest<arma::fmat>(optimizer, 0.03, 0.06);
}

#if ARMA_VERSION_MAJOR > 9 ||\
    (ARMA_VERSION_MAJOR == 9 && ARMA_VERSION_MINOR >= 400)

/**
 * Run QHAdam on logistic regression and make sure the results are acceptable,
 * using arma::sp_mat.
 */
TEST_CASE("QHAdamLogisticRegressionSpMatTest", "[AdamTest]")
{
  QHAdam optimizer;
  LogisticRegressionFunctionTest<arma::sp_mat>(optimizer, 0.003, 0.006);
}

#endif

/**
 * Test the Adam optimizer on the Ackley function.
 * This is to test the Ackley function and not Adam.
 * This test will be removed later.
 */
TEST_CASE("AdamAckleyFunctionTest", "[AdamTest]")
{
  Adam optimizer(0.001, 2, 0.7, 0.999, 1e-8, 500000, 1e-7, false);
  FunctionTest<AckleyFunction>(optimizer);
}

/**
 * Test the Adam optimizer on the Beale function.
 * This is to test the Beale function and not Adam.
 * This test will be removed later.
 */
TEST_CASE("AdamBealeFunctionTest", "[AdamTest]")
{
  Adam optimizer(0.001, 2, 0.7, 0.999, 1e-8, 500000, 1e-7, false);
  FunctionTest<BealeFunction>(optimizer, 0.1, 0.01);
}

/**
 * Test the Adam optimizer on the Goldstein-Price function.
 * This is to test the Goldstein-Price function and not Adam.
 * This test will be removed later.
 */
TEST_CASE("AdamGoldsteinPriceFunctionTest", "[AdamTest]")
{
  Adam optimizer(0.0001, 2, 0.7, 0.999, 1e-8, 500000, 1e-9, false);
  FunctionTest<GoldsteinPriceFunction>(optimizer, 0.1, 0.01);
}

/**
 * Test the Adam optimizer on the Levi function.
 * This is to test the Levi function and not Adam.
 * This test will be removed later.
 */
TEST_CASE("AdamLevyFunctionTest", "[AdamTest]")
{
  Adam optimizer(0.001, 2, 0.7, 0.999, 1e-8, 500000, 1e-9, false);
  FunctionTest<LevyFunctionN13>(optimizer, 0.1, 0.01);
}

/**
 * Test the Adam optimizer on the Himmelblau function.
 * This is to test the Himmelblau function and not Adam.
 * This test will be removed later.
 */
TEST_CASE("AdamHimmelblauFunctionTest", "[AdamTest]")
{
  HimmelblauFunction f;
  Adam optimizer(0.001, 2, 0.7, 0.999, 1e-8, 500000, 1e-9, false);

  arma::mat coordinates = arma::mat("2.9; 1.9");
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(3.0).margin(0.05));
  REQUIRE(coordinates(1) == Approx(2.0).margin(0.05));
}

/**
 * Test the Adam optimizer on the Three-hump camel function.
 * This is to test the Three-hump camel function and not Adam.
 * This test will be removed later.
 */
TEST_CASE("AdamThreeHumpCamelFunctionTest", "[AdamTest]")
{
  Adam optimizer(0.001, 2, 0.7, 0.999, 1e-8, 500000, 1e-9, false);
  FunctionTest<ThreeHumpCamelFunction>(optimizer, 0.1, 0.01);
}

/**
 * Test that multiple runs of the Adam optimizer result in the exact same
 * result.  This specifically tests that the update policy is successfully
 * reset at the start of each optimization.
 */
TEST_CASE("AdamResetPolicyTest", "[AdamTest]")
{
  Adam optimizer(0.5, 2, 0.7, 0.999, 1e-8, 5, 1e-3, false);
  optimizer.ResetPolicy() = true;

  SphereFunction f(2);

  arma::mat coordinatesA = f.GetInitialPoint();
  optimizer.Optimize(f, coordinatesA);

  // A second run should produce the exact same results.
  arma::mat coordinatesB = f.GetInitialPoint();
  optimizer.Optimize(f, coordinatesB);

  CheckMatrices(coordinatesA, coordinatesB);
}
