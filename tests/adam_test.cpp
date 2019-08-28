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
  SphereFunction f(2);
  Adam optimizer(0.5, 2, 0.7, 0.999, 1e-8, 500000, 1e-3, false);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
}

/**
 * Test the Adam optimizer on the Sphere function with arma::fmat.
 */
TEST_CASE("AdamSphereFunctionTestFMat", "[AdamTest]")
{
  SphereFunction f(2);
  Adam optimizer(0.5, 2, 0.7, 0.999, 1e-8, 500000, 1e-3, false);

  arma::fmat coordinates = f.GetInitialPoint<arma::fmat>();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
}

#if ARMA_VERSION_MAJOR > 9 ||\
    (ARMA_VERSION_MAJOR == 9 && ARMA_VERSION_MINOR >= 400)

/**
 * Test the AMSGrad optimizer on the Sphere function with arma::sp_mat.
 */
TEST_CASE("AdamSphereFunctionTestSpMat", "[AdamTest]")
{
  SphereFunction f(2);
  Adam optimizer(0.5, 2, 0.7, 0.999, 1e-8, 500000, 1e-3, false);

  arma::sp_mat coordinates = f.GetInitialPoint<arma::sp_mat>();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
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
  StyblinskiTangFunction f(2);
  Adam optimizer(0.5, 2, 0.7, 0.999, 1e-8, 500000, 1e-3, false);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(-2.9).epsilon(0.01)); // 1% error tolerance.
  REQUIRE(coordinates(1) == Approx(-2.9).epsilon(0.01)); // 1% error tolerance.
}

/**
 * Test the Adam optimizer on the McCormick function.
 */
TEST_CASE("AdamMcCormickFunctionTest", "[AdamTest]")
{
  McCormickFunction f;
  Adam optimizer(0.5, 1, 0.7, 0.999, 1e-8, 500000, 1e-5, false);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  // 3% error tolerance.
  REQUIRE(coordinates(0) == Approx(-0.547).epsilon(0.03));
  REQUIRE(coordinates(1) == Approx(-1.547).epsilon(0.03));
}

/**
 * Test the Adam optimizer on the Matyas function.
 */
TEST_CASE("AdamMatyasFunctionTest", "[AdamTest]")
{
  MatyasFunction f;
  Adam optimizer(0.5, 1, 0.7, 0.999, 1e-8, 500000, 1e-5, false);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  // 3% error tolerance.
  REQUIRE((std::trunc(100.0 * coordinates(0)) / 100.0) ==
      Approx(0.0).epsilon(0.003));
  REQUIRE((std::trunc(100.0 * coordinates(1)) / 100.0) ==
      Approx(0.0).epsilon(0.003));
}

/**
 * Test the Adam optimizer on the Easom function.
 */
TEST_CASE("AdamEasomFunctionTest", "[AdamTest]")
{
  EasomFunction f;
  Adam optimizer(0.2, 1, 0.7, 0.999, 1e-8, 500000, 1e-5, false);

  arma::mat coordinates = arma::mat("2.9; 2.9");
  optimizer.Optimize(f, coordinates);

  // 5% error tolerance.
  REQUIRE((std::trunc(100.0 * coordinates(0)) / 100.0) ==
      Approx(3.14).epsilon(0.005));
  REQUIRE((std::trunc(100.0 * coordinates(1)) / 100.0) ==
      Approx(3.14).epsilon(0.005));
}

/**
 * Test the Adam optimizer on the Booth function.
 */
TEST_CASE("AdamBoothFunctionTest", "[AdamTest]")
{
  BoothFunction f;
  Adam optimizer(1e-1, 1, 0.7, 0.999, 1e-8, 500000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(1.0).epsilon(0.002));
  REQUIRE(coordinates(1) == Approx(3.0).epsilon(0.002));
}

/**
 * Tests the Adam optimizer using a simple test function.
 */
TEST_CASE("SimpleAdamTestFunction", "[AdamTest]")
{
  SGDTestFunction f;
  Adam optimizer(1e-3, 1, 0.9, 0.999, 1e-8, 500000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.3));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.3));
  REQUIRE(coordinates(2) == Approx(0.0).margin(0.3));
}

/**
 * Tests the AdaMax optimizer using a simple test function.
 */
TEST_CASE("SimpleAdaMaxTestFunction", "[AdamTest]")
{
  SGDTestFunction f;
  AdaMax optimizer(2e-3, 1, 0.9, 0.999, 1e-8, 500000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.3));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.3));
  REQUIRE(coordinates(2) == Approx(0.0).margin(0.3));
}

/**
 * Tests the AMSGrad optimizer using a simple test function.
 */
TEST_CASE("SimpleAMSGradTestFunction", "[AdamTest]")
{
  SGDTestFunction f;
  AMSGrad optimizer(1e-3, 1, 0.9, 0.999, 1e-8, 500000, 1e-11, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.3));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.3));
  REQUIRE(coordinates(2) == Approx(0.0).margin(0.3));
}

/**
 * Test the AMSGrad optimizer on the Sphere function with arma::fmat.
 */
TEST_CASE("AMSGradSphereFunctionTestFMat", "[AdamTest]")
{
  SphereFunction f(2);
  AMSGrad optimizer(1e-3, 1, 0.9, 0.999, 1e-8, 500000, 1e-11, true);

  arma::fmat coordinates = f.GetInitialPoint<arma::fmat>();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
}

#if ARMA_VERSION_MAJOR > 9 || \
    (ARMA_VERSION_MAJOR == 9 && ARMA_VERSION_MINOR >= 400)

/**
 * Test the AMSGrad optimizer on the Sphere function with arma::sp_mat.
 */
TEST_CASE("AMSGradSphereFunctionTestSpMat", "[AdamTest]")
{
  SphereFunction f(2);
  AMSGrad optimizer(1e-3, 1, 0.9, 0.999, 1e-8, 500000, 1e-11, true);

  arma::sp_mat coordinates = f.GetInitialPoint<arma::sp_mat>();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
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
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

  Adam adam;
  arma::mat coordinates = lr.GetInitialPoint();
  adam.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}

/**
 * Run AdaMax on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("AdaMaxLogisticRegressionTest", "[AdamTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

  AdaMax adamax(1e-3, 1, 0.9, 0.999, 1e-8, 5000000, 1e-9, true);
  arma::mat coordinates = lr.GetInitialPoint();
  adamax.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}

/**
 * Run AMSGrad on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("AMSGradLogisticRegressionTest", "[AdamTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

  AMSGrad amsgrad(1e-3, 1, 0.9, 0.999, 1e-8, 500000, 1e-11, true);
  arma::mat coordinates = lr.GetInitialPoint();
  amsgrad.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}

/**
 * Tests the Nadam optimizer using a simple test function.
 */
TEST_CASE("SimpleNadamTestFunction", "[AdamTest]")
{
  SGDTestFunction f;
  Nadam optimizer(1e-3, 1, 0.9, 0.99, 1e-8, 500000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.3));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.3));
  REQUIRE(coordinates(2) == Approx(0.0).margin(0.3));
}

/**
 * Run Nadam on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("NadamLogisticRegressionTest", "[AdamTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

  Nadam nadam;
  arma::mat coordinates = lr.GetInitialPoint();
  nadam.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}

/**
 * Tests the NadaMax optimizer using a simple test function.
 */
TEST_CASE("SimpleNadaMaxTestFunction", "[AdamTest]")
{
  SGDTestFunction f;
  NadaMax optimizer(1e-3, 1, 0.9, 0.99, 1e-8, 500000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.3));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.3));
  REQUIRE(coordinates(2) == Approx(0.0).margin(0.3));
}

/**
 * Run NadaMax on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("NadaMaxLogisticRegressionTest", "[AdamTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

  NadaMax nadamax(1e-3, 1, 0.9, 0.999, 1e-8, 5000000, 1e-9, true);
  arma::mat coordinates = lr.GetInitialPoint();
  nadamax.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}

/**
 * Tests the OptimisticAdam optimizer using a simple test function.
 */
TEST_CASE("SimpleOptimisticAdamTestFunction", "[AdamTest]")
{
  // Sometimes this test can fail randomly, so we allow it to run up to three
  // times.
  bool success = false;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    SGDTestFunction f;
    OptimisticAdam optimizer(1e-2, 1, 0.9, 0.99, 1e-8);

    arma::mat coordinates = f.GetInitialPoint();
    optimizer.Optimize(f, coordinates);

    success = (coordinates(0) == Approx(0.0).margin(0.3)) &&
              (coordinates(1) == Approx(0.0).margin(0.3)) &&
              (coordinates(2) == Approx(0.0).margin(0.3));
    if (success)
      break;
  }

  REQUIRE(success == true);
}

/**
 * Run OptimisticAdam on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("OptimisticAdamLogisticRegressionTest", "[AdamTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

  OptimisticAdam optimisticAdam;
  arma::mat coordinates = lr.GetInitialPoint();
  optimisticAdam.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}

/**
 * Tests the Padam optimizer using a simple test function.
 */
TEST_CASE("SimplePadamTestFunction", "[AdamTest]")
{
  // Sometimes this test can fail randomly, so we allow it to run up to three
  // times.
  bool success = false;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    SGDTestFunction f;
    Padam optimizer(1e-2, 1, 0.9, 0.99, 0.25, 1e-8);

    arma::mat coordinates = f.GetInitialPoint();
    optimizer.Optimize(f, coordinates);

    success = (coordinates(0) == Approx(0.0).margin(0.3)) &&
              (coordinates(1) == Approx(0.0).margin(0.3)) &&
              (coordinates(2) == Approx(0.0).margin(0.3));
    if (success)
      break;
  }

  REQUIRE(success == true);
}

/**
 * Run Padam on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("PadamLogisticRegressionTest", "[AdamTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

  Padam optimizer;
  arma::mat coordinates = lr.GetInitialPoint();
  optimizer.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}

/**
 * Tests the QHadam optimizer using a simple test function.
 */
TEST_CASE("SimpleQHAdamTestFunction", "[AdamTest]")
{
  SGDTestFunction f;
  QHAdam optimizer(0.02, 32, 0.6, 0.9, 0.9, 0.999, 1e-8, 200000, 1e-7, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  bool success = (coordinates(0) == Approx(0.0).margin(0.3)) &&
                 (coordinates(1) == Approx(0.0).margin(0.3)) &&
                 (coordinates(2) == Approx(0.0).margin(0.3));
  REQUIRE(success == true);
}

/**
 * Run QHAdam on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("QHAdamLogisticRegressionTest", "[AdamTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

  QHAdam optimizer;
  arma::mat coordinates = lr.GetInitialPoint();
  optimizer.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}

/**
 * Run QHAdam on logistic regression and make sure the results are acceptable,
 * using arma::fmat.
 */
TEST_CASE("QHAdamLogisticRegressionFMatTest", "[AdamTest]")
{
  arma::fmat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<arma::fmat> lr(shuffledData, shuffledResponses, 0.5);

  QHAdam optimizer;
  arma::fmat coordinates = lr.GetInitialPoint();
  optimizer.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const float acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.03)); // 3% error tolerance.

  const float testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.06)); // 6% error tolerance.
}

#if ARMA_VERSION_MAJOR > 9 ||\
    (ARMA_VERSION_MAJOR == 9 && ARMA_VERSION_MINOR >= 400)

/**
 * Run QHAdam on logistic regression and make sure the results are acceptable,
 * using arma::sp_mat.
 */
TEST_CASE("QHAdamLogisticRegressionSpMatTest", "[AdamTest]")
{
  arma::sp_mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);
  LogisticRegression<arma::sp_mat> lr(shuffledData, shuffledResponses, 0.5);

  QHAdam optimizer;
  arma::sp_mat coordinates = lr.GetInitialPoint();
  optimizer.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}

#endif

/**
 * Test the Adam optimizer on the Ackley function.
 * This is to test the Ackley function and not Adam.
 * This test will be removed later.
 */
TEST_CASE("AdamAckleyFunctionTest", "[AdamTest]")
{
  AckleyFunction f;
  Adam optimizer(0.001, 2, 0.7, 0.999, 1e-8, 500000, 1e-7, false);

  arma::mat coordinates = arma::mat("0.02; 0.02");
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.001));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.001));
}

/**
 * Test the Adam optimizer on the Beale function.
 * This is to test the Beale function and not Adam.
 * This test will be removed later.
 */
TEST_CASE("AdamBealeFunctionTest", "[AdamTest]")
{
  BealeFunction f;
  Adam optimizer(0.001, 2, 0.7, 0.999, 1e-8, 500000, 1e-7, false);

  arma::mat coordinates = arma::mat("2.8; 0.35");
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(3.0).margin(0.01));
  REQUIRE(coordinates(1) == Approx(0.5).margin(0.01));
}

/**
 * Test the Adam optimizer on the Goldstein-Price function.
 * This is to test the Goldstein-Price function and not Adam.
 * This test will be removed later.
 */
TEST_CASE("AdamGoldsteinPriceFunctionTest", "[AdamTest]")
{
  GoldsteinPriceFunction f;
  Adam optimizer(0.0001, 2, 0.7, 0.999, 1e-8, 500000, 1e-9, false);

  arma::mat coordinates = arma::mat("0.2; -0.5");
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0).margin(0.01));
  REQUIRE(coordinates(1) == Approx(-1).margin(0.01));
}

/**
 * Test the Adam optimizer on the Levi function.
 * This is to test the Levi function and not Adam.
 * This test will be removed later.
 */
TEST_CASE("AdamLevyFunctionTest", "[AdamTest]")
{
  LevyFunctionN13 f;
  Adam optimizer(0.001, 2, 0.7, 0.999, 1e-8, 500000, 1e-9, false);

  arma::mat coordinates = arma::mat("0.9; 1.1");
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(1).margin(0.01));
  REQUIRE(coordinates(1) == Approx(1).margin(0.01));
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
  ThreeHumpCamelFunction f;
  Adam optimizer(0.001, 2, 0.7, 0.999, 1e-8, 500000, 1e-9, false);

  arma::mat coordinates = arma::mat("1; 1");
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.01));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.01));
}

/**
 * Test the Adam optimizer on Schaffer function N.2.
 * This is to test Schaffer function N.2 and not Adam.
 * This test will be removed later.
 */
TEST_CASE("AdamSchafferFunctionN2Test", "[AdamTest]")
{
  SchafferFunctionN2 f;
  Adam optimizer(0.001, 2, 0.7, 0.999, 1e-8, 500000, 1e-9, false);

  arma::mat coordinates = arma::mat("1; 1");
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.01));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.01));
}
