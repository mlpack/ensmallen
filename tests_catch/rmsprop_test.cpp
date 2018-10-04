/**
 * @file rmsprop_test.cpp
 * @author Marcus Edel
 *
 * Tests the RMSProp optimizer.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "test_tools.hpp"

TEST_CASE("rmsprop_sgd_function ", "[rmsprop]")
{
  SGDTestFunction f;
  RMSProp optimizer(1e-3, 1, 0.99, 1e-8, 5000000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(std::abs(coordinates[0]) <= 0.1);
  REQUIRE(std::abs(coordinates[1]) <= 0.1);
  REQUIRE(std::abs(coordinates[2]) <= 0.1);
}
