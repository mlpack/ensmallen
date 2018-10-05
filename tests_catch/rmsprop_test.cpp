// Copyright (c) 2018 ensmallen developers.
// 
// Licensed under the 3-clause BSD license (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.opensource.org/licenses/BSD-3-Clause

#include <ensmallen.hpp>
#include "catch.hpp"

using namespace ens;

TEST_CASE("rmsprop_sgd_function", "[rmsprop]")
{
  SGDTestFunction f;
  RMSProp optimizer(1e-3, 1, 0.99, 1e-8, 5000000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  REQUIRE(std::abs(coordinates[0]) <= 0.1);
  REQUIRE(std::abs(coordinates[1]) <= 0.1);
  REQUIRE(std::abs(coordinates[2]) <= 0.1);
}
