/**
 * @file spalera_sgd_test.cpp
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
 * Run SPALeRA SGD on logistic regression and make sure the results are
 * acceptable.
 */
TEST_CASE("LogisticRegressionTest","[SPALeRASGDTest]")
{
  // Run SPALeRA SGD with a couple of batch sizes.
  for (size_t batchSize = 30; batchSize < 50; batchSize += 5)
  {
    SPALeRASGD<> optimizer(0.05 / batchSize, batchSize, 10000, 1e-4);
    LogisticRegressionFunctionTest(optimizer, 0.015, 0.024, 3);
  }
}

/**
 * Run SPALeRA SGD on logistic regression and make sure the results are
 * acceptable.  Use arma::fmat.
 */
TEST_CASE("LogisticRegressionFMatTest","[SPALeRASGDTest]")
{
  // Run SPALeRA SGD with a couple of batch sizes.
  for (size_t batchSize = 30; batchSize < 50; batchSize += 5)
  {
    SPALeRASGD<> optimizer(0.05 / batchSize, batchSize, 10000, 1e-4);
    LogisticRegressionFunctionTest<arma::fmat>(optimizer, 0.015, 0.024, 3);
  }
}
