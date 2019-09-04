/**
 * @file snorms3_test.cpp
 * @author Vivek Pal
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
 * Tests the SMORMS3 optimizer using a simple test function.
 */
TEST_CASE("SimpleSMORMS3TestFunction","[SMORMS3Test]")
{
  SGDTestFunction f;
  SMORMS3 s(0.001, 1, 1e-16, 5000000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  s.Optimize(f, coordinates);

  REQUIRE(coordinates(0) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(1) == Approx(0.0).margin(0.1));
  REQUIRE(coordinates(2) == Approx(0.0).margin(0.1));
}

/**
 * Run SMORMS3 on logistic regression and make sure the results are acceptable.
 */
TEST_CASE("SMORMS3LogisticRegressionTest","[SMORMS3Test]")
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  SMORMS3 smorms3;
  LogisticRegression<> lr(shuffledData, shuffledResponses, 0.5);

  arma::mat coordinates = lr.GetInitialPoint();
  smorms3.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}

/**
 * Run SMORMS3 on logistic regression and make sure the results are acceptable.
 * Use arma::fmat.
 */
TEST_CASE("SMORMS3LogisticRegressionFMatTest","[SMORMS3Test]")
{
  arma::fmat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  SMORMS3 smorms3;
  LogisticRegression<arma::fmat> lr(shuffledData, shuffledResponses, 0.5);

  arma::fmat coordinates = lr.GetInitialPoint();
  smorms3.Optimize(lr, coordinates);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses, coordinates);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses,
      coordinates);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}
