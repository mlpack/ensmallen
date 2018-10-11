// Copyright (c) 2018 ensmallen developers.
//
// Licensed under the 3-clause BSD license (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.opensource.org/licenses/BSD-3-Clause

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

  REQUIRE(coordinates[0] == Approx(0.0).margin(0.1));
  REQUIRE(coordinates[1] == Approx(0.0).margin(0.1));
  REQUIRE(coordinates[2] == Approx(0.0).margin(0.1));
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
