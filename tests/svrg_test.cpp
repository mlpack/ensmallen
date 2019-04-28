/**
 * @file svrg_test.cpp
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
 * Run SVRG on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("SVRGLogisticRegressionTest", "[SVRGTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  // Now run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 50; batchSize += 5)
  {
    SVRG optimizer(0.005, batchSize, 300, 0, 1e-5, true);
    LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

    arma::mat coordinates = lr.GetInitialPoint();
    optimizer.Optimize(lr, coordinates);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses, coordinates);
    REQUIRE(acc == Approx(100.0).epsilon(0.015)); // 1.5% error tolerance.
    // REQUIRE(acc == Approx(100.0).scale(0.015)); // 1.5% error tolerance.
    // TODO: not sure whether .epsilon() or .scale() is more appropriate

    const double testAcc = lr.ComputeAccuracy(testData, testResponses,
        coordinates);
    REQUIRE(testAcc == Approx(100.0).epsilon(0.015)); // 1.5% error tolerance.
  }
}

/**
 * Run SVRG_BB on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("SVRGBBLogisticRegressionTest", "[SVRGTest]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  // Now run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 50; batchSize += 5)
  {
    SVRG_BB optimizer(0.005, batchSize, 300, 0, 1e-5, true, false,
        SVRGUpdate(), BarzilaiBorweinDecay(0.1));
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
