// Copyright (c) 2018 ensmallen developers.
// 
// Licensed under the 3-clause BSD license (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.opensource.org/licenses/BSD-3-Clause

#include <ensmallen.hpp>
#include "catch.hpp"

using namespace arma;
using namespace ens;
using namespace ens::test;

// #include <mlpack/core.hpp>
// #include <mlpack/core/optimizers/adam/adam.hpp>
// #include <mlpack/core/optimizers/problems/sgd_test_function.hpp>
// #include <mlpack/core/optimizers/problems/colville_function.hpp>
// #include <mlpack/core/optimizers/problems/booth_function.hpp>
// #include <mlpack/core/optimizers/problems/sphere_function.hpp>
// #include <mlpack/core/optimizers/problems/styblinski_tang_function.hpp>
// #include <mlpack/core/optimizers/problems/mc_cormick_function.hpp>
// #include <mlpack/core/optimizers/problems/matyas_function.hpp>
// #include <mlpack/core/optimizers/problems/easom_function.hpp>
// #include <mlpack/methods/logistic_regression/logistic_regression.hpp>
// 
// using namespace mlpack::optimization;
// using namespace mlpack::optimization::test;
// using namespace mlpack::distribution;
// using namespace mlpack::regression;
// using namespace mlpack;

/**
 * Test the Adam optimizer on the Sphere function.
 */
TEST_CASE("AdamSphereFunctionTest", "[AdamTest]")
{
  SphereFunction f(2);
  Adam optimizer(0.5, 2, 0.7, 0.999, 1e-8, 500000, 1e-3, false);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates[0] == Approx(0.0).margin(0.1));
  REQUIRE(coordinates[1] == Approx(0.0).margin(0.1));
}

/**
 * Test the Adam optimizer on the Wood function.
 */
TEST_CASE("AdamStyblinskiTangFunctionTest", "[AdamTest]")
{
  StyblinskiTangFunction f(2);
  Adam optimizer(0.5, 2, 0.7, 0.999, 1e-8, 500000, 1e-3, false);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates[0] == Approx(-2.9).epsilon(0.01)); // 1% error tolerance.
  REQUIRE(coordinates[1] == Approx(-2.9).epsilon(0.01)); // 1% error tolerance.
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

  REQUIRE(coordinates[0] == Approx(-0.547).epsilon(0.03)); // 3% error tolerance.
  REQUIRE(coordinates[0] == Approx(-1.547).epsilon(0.03)); // 3% error tolerance.
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
  REQUIRE((std::trunc(100.0 * coordinates[0]) / 100.0) == Approx(0.0).epsilon(0.003));
  REQUIRE((std::trunc(100.0 * coordinates[1]) / 100.0) == Approx(0.0).epsilon(0.003));
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
  REQUIRE(*std::trunc(100.0 * coordinates[0]) / 100.0) == Approx(3.14).epsilon(0.003));
  REQUIRE(*std::trunc(100.0 * coordinates[1]) / 100.0) == Approx(3.14).epsilon(0.003));
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

  REQUIRE(coordinates[0] == Approx(1.0).epsilon(0.002));
  REQUIRE(coordinates[1] == Approx(3.0).epsilon(0.002));
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

  REQUIRE(coordinates[0] == Approx(0.0).margin(0.1));
  REQUIRE(coordinates[1] == Approx(0.0).margin(0.1));
  REQUIRE(coordinates[2] == Approx(0.0).margin(0.1));
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

  REQUIRE(coordinates[0] == Approx(0.0).margin(0.1));
  REQUIRE(coordinates[1] == Approx(0.0).margin(0.1));
  REQUIRE(coordinates[2] == Approx(0.0).margin(0.1));
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

  REQUIRE(coordinates[0] == Approx(0.0).margin(0.1));
  REQUIRE(coordinates[1] == Approx(0.0).margin(0.1));
  REQUIRE(coordinates[2] == Approx(0.0).margin(0.1));
}

/**
 * Run Adam on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("AdamLogisticRegressionTest", "[AdamTest]")
{
  // Generate a two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"), arma::eye<arma::mat>(3, 3));

  arma::mat data(3, 1000);
  arma::Row<size_t> responses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }

  // Shuffle the dataset.
  arma::uvec indices = arma::shuffle(arma::linspace<arma::uvec>(0,
      data.n_cols - 1, data.n_cols));
  arma::mat shuffledData(3, 1000);
  arma::Row<size_t> shuffledResponses(1000);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    shuffledData.col(i) = data.col(indices[i]);
    shuffledResponses[i] = responses[indices[i]];
  }

  // Create a test set.
  arma::mat testData(3, 1000);
  arma::Row<size_t> testResponses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    testData.col(i) = g1.Random();
    testResponses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    testData.col(i) = g2.Random();
    testResponses[i] = 1;
  }

  Adam adam;
  LogisticRegression<> lr(shuffledData, shuffledResponses, adam, 0.5);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}

/**
 * Run AdaMax on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("AdaMaxLogisticRegressionTest", "[AdamTest]")
{
  // Generate a two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"), arma::eye<arma::mat>(3, 3));

  arma::mat data(3, 1000);
  arma::Row<size_t> responses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }

  // Shuffle the dataset.
  arma::uvec indices = arma::shuffle(arma::linspace<arma::uvec>(0,
      data.n_cols - 1, data.n_cols));
  arma::mat shuffledData(3, 1000);
  arma::Row<size_t> shuffledResponses(1000);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    shuffledData.col(i) = data.col(indices[i]);
    shuffledResponses[i] = responses[indices[i]];
  }

  // Create a test set.
  arma::mat testData(3, 1000);
  arma::Row<size_t> testResponses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    testData.col(i) = g1.Random();
    testResponses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    testData.col(i) = g2.Random();
    testResponses[i] = 1;
  }

  AdaMax adamax(1e-3, 1, 0.9, 0.999, 1e-8, 5000000, 1e-9, true);
  LogisticRegression<> lr(shuffledData, shuffledResponses, adamax, 0.5);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}

/**
 * Run AMSGrad on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("AMSGradLogisticRegressionTest", "[AdamTest]")
{
  // Generate a two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"), arma::eye<arma::mat>(3, 3));

  arma::mat data(3, 1000);
  arma::Row<size_t> responses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }

  // Shuffle the dataset.
  arma::uvec indices = arma::shuffle(arma::linspace<arma::uvec>(0,
      data.n_cols - 1, data.n_cols));
  arma::mat shuffledData(3, 1000);
  arma::Row<size_t> shuffledResponses(1000);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    shuffledData.col(i) = data.col(indices[i]);
    shuffledResponses[i] = responses[indices[i]];
  }

  // Create a test set.
  arma::mat testData(3, 1000);
  arma::Row<size_t> testResponses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    testData.col(i) = g1.Random();
    testResponses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    testData.col(i) = g2.Random();
    testResponses[i] = 1;
  }

  AMSGrad amsgrad(1e-3, 1, 0.9, 0.999, 1e-8, 500000, 1e-11, true);
  LogisticRegression<> lr(shuffledData, shuffledResponses, amsgrad, 0.5);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses);
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

  REQUIRE(coordinates[0] == Approx(0.0).margin(0.1));
  REQUIRE(coordinates[1] == Approx(0.0).margin(0.1));
  REQUIRE(coordinates[2] == Approx(0.0).margin(0.1));
}

/**
 * Run Nadam on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("NadamLogisticRegressionTest", "[AdamTest]")
{
  // Generate a two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"),
  arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"),
  arma::eye<arma::mat>(3, 3));

  arma::mat data(3, 1000);
  arma::Row<size_t> responses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }

  // Shuffle the dataset.
  arma::uvec indices = arma::shuffle(arma::linspace<arma::uvec>(0,
  data.n_cols - 1, data.n_cols));
  arma::mat shuffledData(3, 1000);
  arma::Row<size_t> shuffledResponses(1000);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    shuffledData.col(i) = data.col(indices[i]);
    shuffledResponses[i] = responses[indices[i]];
  }

  // Create a test set.
  arma::mat testData(3, 1000);
  arma::Row<size_t> testResponses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    testData.col(i) = g1.Random();
    testResponses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    testData.col(i) = g2.Random();
    testResponses[i] = 1;
  }

  Nadam nadam;
  LogisticRegression<> lr(shuffledData, shuffledResponses, nadam, 0.5);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses);
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

  REQUIRE(coordinates[0] == Approx(0.0).margin(0.1));
  REQUIRE(coordinates[1] == Approx(0.0).margin(0.1));
  REQUIRE(coordinates[2] == Approx(0.0).margin(0.1));
}

/**
 * Run NadaMax on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("NadaMaxLogisticRegressionTest", "[AdamTest]")
{
  // Generate a two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"),
  arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"),
  arma::eye<arma::mat>(3, 3));

  arma::mat data(3, 1000);
  arma::Row<size_t> responses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }

  // Shuffle the dataset.
  arma::uvec indices = arma::shuffle(arma::linspace<arma::uvec>(0,
  data.n_cols - 1, data.n_cols));
  arma::mat shuffledData(3, 1000);
  arma::Row<size_t> shuffledResponses(1000);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    shuffledData.col(i) = data.col(indices[i]);
    shuffledResponses[i] = responses[indices[i]];
  }

  // Create a test set.
  arma::mat testData(3, 1000);
  arma::Row<size_t> testResponses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    testData.col(i) = g1.Random();
    testResponses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    testData.col(i) = g2.Random();
    testResponses[i] = 1;
  }

  NadaMax nadamax;
  LogisticRegression<> lr(shuffledData, shuffledResponses, nadamax, 0.5);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}

/**
 * Tests the OptimisticAdam optimizer using a simple test function.
 */
TEST_CASE("SimpleOptimisticAdamTestFunction", "[AdamTest]")
{
  SGDTestFunction f;
  OptimisticAdam optimizer(1e-2, 1, 0.9, 0.99, 1e-8);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(coordinates[0] == Approx(0.0).margin(0.1));
  REQUIRE(coordinates[1] == Approx(0.0).margin(0.1));
  REQUIRE(coordinates[2] == Approx(0.0).margin(0.1));
}

/**
 * Run OptimisticAdam on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("OptimisticAdamLogisticRegressionTest", "[AdamTest]")
{
  // Generate a two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"),
  arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"),
  arma::eye<arma::mat>(3, 3));

  arma::mat data(3, 1000);
  arma::Row<size_t> responses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }

  // Shuffle the dataset.
  arma::uvec indices = arma::shuffle(arma::linspace<arma::uvec>(0,
  data.n_cols - 1, data.n_cols));
  arma::mat shuffledData(3, 1000);
  arma::Row<size_t> shuffledResponses(1000);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    shuffledData.col(i) = data.col(indices[i]);
    shuffledResponses[i] = responses[indices[i]];
  }

  // Create a test set.
  arma::mat testData(3, 1000);
  arma::Row<size_t> testResponses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    testData.col(i) = g1.Random();
    testResponses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    testData.col(i) = g2.Random();
    testResponses[i] = 1;
  }

  OptimisticAdam optimisticAdam;
  LogisticRegression<> lr(shuffledData, shuffledResponses, optimisticAdam, 0.5);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}
