/**
 * @file pso_test.cpp
 * @author Chintan Soni
 * @author Suryoday Basak
 *
 * Test file for PSO optimizer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <ensmallen.hpp>
#include "catch.hpp"
#include "test_function_tools.hpp"

using namespace ens;
using namespace ens::test;

/**
 * Tests the LBest optimizer using simple test functions.
 */

TEST_CASE("RastriginFunctionTest", "[PSOTest]")
{
  RastriginFunction f(4);
  LBestPSO s;

  arma::vec coordinates = f.GetInitialPoint();
  double result = s.Optimize(f, coordinates);

  //REQUIRE(coordinates[j] == Approx(0.0).margin(0.003)); REFERENCE
  //BOOST_REQUIRE_SMALL(result, 1e-5);
  REQUIRE(result <= 1e-5); //work on this
  for (size_t j = 0; j < 4; ++j)
	REQUIRE(coordinates[j] <= 1e-3); //work on this
    //BOOST_REQUIRE_SMALL(coordinates[j], 1e-3);
}

TEST_CASE("RosenbrockFunctionTest", "[PSOTest]")
{
  RosenbrockFunction f;
  LBestPSO s;

  arma::vec coordinates = f.GetInitialPoint();
  double result = s.Optimize(f, coordinates);

  //BOOST_REQUIRE_SMALL(result, 1e-5);
  REQUIRE(result <= 1e-5);
  for (size_t j = 0; j < 2; ++j)
	REQUIRE(coordinates[j] == Approx(1.0).margin(0.0001)); //work on this
    //BOOST_REQUIRE_CLOSE(coordinates[j], (double) 1.0, 1e-3);
}

TEST_CASE("SphereFunctionTest", "[PSOTest]")
{
  SphereFunction f(4);
  LBestPSO s;

  arma::vec coordinates = f.GetInitialPoint();
  double result = s.Optimize(f, coordinates);

  //BOOST_REQUIRE_SMALL(result, 1e-5);
  REQUIRE(result <= 1e-5);
  for (size_t j = 0; j < 4; ++j)
    //BOOST_REQUIRE_SMALL(coordinates[j], 1e-3);
	REQUIRE(coordinates[j] <= 1e-3); //work on this
}

//BOOST_AUTO_TEST_SUITE_END();
