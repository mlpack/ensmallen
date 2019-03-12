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

using namespace ens;
using namespace ens::test;

TEST_CASE("LBestPSORosenbrockFunctionTest", "[PSOTest]")
{
  RosenbrockFunction f;
  LBestPSO s;

  arma::vec coords = f.GetInitialPoint();
  if (!s.Optimize(f, coords))
    FAIL("LBest PSO optimization reported failure for Rosenbrock Function.");

  double finalValue = f.Evaluate(coords);

  REQUIRE(finalValue == Approx(0.0).margin(1e-5));
  REQUIRE(coords[0] == Approx(1.0).epsilon(1e-7));
  REQUIRE(coords[1] == Approx(1.0).epsilon(1e-7));
}

TEST_CASE("LBestPSOSphereFunctionTest", "[PSOTest]")
{
  SphereFunction f(4);
  LBestPSO s;

  arma::vec coords = f.GetInitialPoint();
  if (!s.Optimize(f, coords))
    FAIL("LBest PSO optimization reported failure for Sphere Function.");
  
  double finalValue = f.Evaluate(coords);
  //BOOST_REQUIRE_SMALL(result, 1e-5);
  REQUIRE(finalValue <= 1e-5);
  for (size_t j = 0; j < 4; ++j)
    //BOOST_REQUIRE_SMALL(coordinates[j], 1e-3);
	REQUIRE(coords[j] <= 1e-3); //work on this
}


TEST_CASE("ConstrainedSquaredFunctionTest", "[PSOTest]")
{
  class SquaredFunction
  {
    public:
    // This returns f(x) = 2 |x|^2.
    double Evaluate(const arma::mat& x)
    {
      if (x.at(0,0) < 1.0)
	return std::numeric_limits<double>::max();
      else
        return std::pow(arma::norm(x), 2.0);
    }
  };
  arma::mat x("2.0");

  LBestPSO s;
  SquaredFunction f;

  arma::vec coords = x;
  if (!s.Optimize(f, coords))
    FAIL("LBest PSO optimization reported failure for Linear Constrained Square Function.");
  
  double finalValue = f.Evaluate(coords);
  //BOOST_REQUIRE_SMALL(result, 1e-5);
  REQUIRE(finalValue == Approx(1.0).epsilon(1e-5));
  REQUIRE(coords[0] == Approx(1.0).epsilon(1e-5));
}

TEST_CASE("ConstrainedAffineFunctionTest", "[PSOTest]")
{
  class AffineFunction
  {
    public:
    // This returns f(x) = 2 |x|^2.
    double Evaluate(const arma::mat& x)
    {
      if (x.at(0,0) < -3.0)
	return std::numeric_limits<double>::max();
      else
        return x.at(0,0);
    }
  };
  arma::mat x("2.0");

  LBestPSO s;
  AffineFunction f;

  arma::vec coords = x;
  if (!s.Optimize(f, coords))
    FAIL("LBest PSO optimization reported failure for Linear Constrained Affine Function.");
  
  double finalValue = f.Evaluate(coords);
  //BOOST_REQUIRE_SMALL(result, 1e-5);
  REQUIRE(finalValue == Approx(-3.0).epsilon(1e-5));
  REQUIRE(coords[0] == Approx(-3.0).epsilon(1e-5));
}

//BOOST_AUTO_TEST_SUITE_END();
