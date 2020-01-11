/**
 * @file katyusha_test.cpp
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
 * Run Katyusha on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("KatyushaLogisticRegressionTest", "[KatyushaTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  // Now run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 30; batchSize < 45; batchSize += 5)
  {
    Katyusha optimizer(1.0, 10.0, batchSize, 100, 0, 1e-10, true);
    LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

    arma::mat coordinates = lr.GetInitialPoint();
    optimizer.Optimize(lr, coordinates);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses, coordinates);
    REQUIRE(acc == Approx(100.0).epsilon(0.015)); // 1.5% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses,
        coordinates);
    REQUIRE(testAcc == Approx(100.0).epsilon(0.015)); // 1.5% error tolerance.
  }
}

/**
 * Run Proximal Katyusha on logistic regression and make sure the results are
 * acceptable.
 */
TEST_CASE("KatyushaProximalLogisticRegressionTest", "[KatyushaTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  // Now run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 30; batchSize < 45; batchSize += 5)
  {
    KatyushaProximal optimizer(1.0, 10.0, batchSize, 100, 0, 1e-10, true);
    LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

    arma::mat coordinates = lr.GetInitialPoint();
    optimizer.Optimize(lr, coordinates);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses, coordinates);
    REQUIRE(acc == Approx(100.0).epsilon(0.015)); // 1.5% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses,
        coordinates);
    REQUIRE(testAcc == Approx(100.0).epsilon(0.015)); // 1.5% error tolerance.
  }
}

/**
 * Run Katyusha on logistic regression and make sure the results are acceptable.
 * Use arma::fmat.
 */
TEST_CASE("KatyushaLogisticRegressionFMatTest", "[KatyushaTest]")
{
  arma::fmat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  // Now run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 30; batchSize < 45; batchSize += 5)
  {
    Katyusha optimizer(1.0, 10.0, batchSize, 100, 0, 1e-10, true);
    LogisticRegression<arma::fmat> lr(shuffledData, shuffledResponses, 0.5);

    arma::fmat coordinates = lr.GetInitialPoint();
    optimizer.Optimize(lr, coordinates);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses, coordinates);
    REQUIRE(acc == Approx(100.0).epsilon(0.015)); // 1.5% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses,
        coordinates);
    REQUIRE(testAcc == Approx(100.0).epsilon(0.015)); // 1.5% error tolerance.
  }
}

/**
 * Run Proximal Katyusha on logistic regression and make sure the results are
 * acceptable.  Use arma::fmat.
 */
TEST_CASE("KatyushaProximalLogisticRegressionFMatTest", "[KatyushaTest]")
{
  arma::fmat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  // Now run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 30; batchSize < 45; batchSize += 5)
  {
    KatyushaProximal optimizer(1.0, 10.0, batchSize, 100, 0, 1e-10, true);
    LogisticRegression<arma::fmat> lr(shuffledData, shuffledResponses, 0.5);

    arma::fmat coordinates = lr.GetInitialPoint();
    optimizer.Optimize(lr, coordinates);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses, coordinates);
    REQUIRE(acc == Approx(100.0).epsilon(0.015)); // 1.5% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses,
        coordinates);
    REQUIRE(testAcc == Approx(100.0).epsilon(0.015)); // 1.5% error tolerance.
  }
}

#if ARMA_VERSION_MAJOR > 9 ||\
    (ARMA_VERSION_MAJOR == 9 && ARMA_VERSION_MINOR >= 400)

/**
 * Run Katyusha on logistic regression and make sure the results are acceptable.
 * Use arma::sp_mat.
 */
TEST_CASE("KatyushaLogisticRegressionSpMatTest", "[KatyushaTest]")
{
  arma::sp_mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  // Now run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 30; batchSize < 45; batchSize += 5)
  {
    Katyusha optimizer(1.0, 10.0, batchSize, 100, 0, 1e-10, true);
    LogisticRegression<arma::sp_mat> lr(shuffledData, shuffledResponses, 0.5);

    arma::sp_mat coordinates = lr.GetInitialPoint();
    optimizer.Optimize(lr, coordinates);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses, coordinates);
    REQUIRE(acc == Approx(100.0).epsilon(0.015)); // 1.5% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses,
        coordinates);
    REQUIRE(testAcc == Approx(100.0).epsilon(0.015)); // 1.5% error tolerance.
  }
}

/**
 * Run Proximal Katyusha on logistic regression and make sure the results are
 * acceptable.  Use arma::sp_mat.
 */
TEST_CASE("KatyushaProximalLogisticRegressionSpMatTest", "[KatyushaTest]")
{
  arma::sp_mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  // Now run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 30; batchSize < 45; batchSize += 5)
  {
    KatyushaProximal optimizer(1.0, 10.0, batchSize, 100, 0, 1e-10, true);
    LogisticRegression<arma::sp_mat> lr(shuffledData, shuffledResponses, 0.5);

    arma::sp_mat coordinates = lr.GetInitialPoint();
    optimizer.Optimize(lr, coordinates);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses, coordinates);
    REQUIRE(acc == Approx(100.0).epsilon(0.015)); // 1.5% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses,
        coordinates);
    REQUIRE(testAcc == Approx(100.0).epsilon(0.015)); // 1.5% error tolerance.
  }
}

#endif
