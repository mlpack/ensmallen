/**
 * @file gradient_descent_test.cpp
 * @author Sumedh Ghaisas
 *
 * Test file for Gradient Descent optimizer.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <armadillo>
#include <ensmallen.hpp>
#include <ensmallen_bits/problems/problems.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace std;
using namespace arma;
using namespace ens;
using namespace ens::test;

BOOST_AUTO_TEST_SUITE(GradientDescentTest);

BOOST_AUTO_TEST_CASE(SimpleGDTestFunction)
{
  GDTestFunction f;
  GradientDescent s(0.01, 5000000, 1e-9);

  arma::vec coordinates = f.GetInitialPoint();
  double result = s.Optimize(f, coordinates);

  BOOST_REQUIRE_SMALL(result, 1e-4);
  BOOST_REQUIRE_SMALL(coordinates[0], 1e-2);
  BOOST_REQUIRE_SMALL(coordinates[1], 1e-2);
  BOOST_REQUIRE_SMALL(coordinates[2], 1e-2);
}

BOOST_AUTO_TEST_CASE(RosenbrockTest)
{
  // Create the Rosenbrock function.
  RosenbrockFunction f;

  GradientDescent s(0.001, 0, 1e-15);

  arma::mat coordinates = f.GetInitialPoint();
  double result = s.Optimize(f, coordinates);

  BOOST_REQUIRE_SMALL(result, 1e-10);
  for (size_t j = 0; j < 2; ++j)
    BOOST_REQUIRE_CLOSE(coordinates[j], (double) 1.0, 1e-3);
}

BOOST_AUTO_TEST_SUITE_END();
