/**
 * @file grid_search_test.cpp
 * @author Ryan Curtin
 *
 * Test file for the GridSearch optimizer.
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

// An implementation of a simple categorical function.  The parameters can be
// understood as x = [c1 c2 c3].  When c1 = 0, c2 = 2, and c3 = 1, the value of
// f(x) is 0.  In any other case, the value of f(x) is 10.  Therefore, the
// optimum is found at [0, 2, 1].
class SimpleCategoricalFunction
{
 public:
  // Return the objective function f(x) as described above.
  template<typename MatType>
  typename MatType::elem_type Evaluate(const MatType& x)
  {
    if (size_t(x(0)) == 0 &&
        size_t(x(1)) == 2 &&
        size_t(x(2)) == 1)
      return 0.0;
    else
      return 10.0;
  }
};

TEST_CASE("GridSearchTest", "[GridSearchTest]")
{
  // Create and optimize the categorical function with the GridSearch
  // optimizer.  We must also create a std::vector<bool> that holds the types
  // of each dimension, and an arma::Row<size_t> that holds the number of
  // categories in each dimension.
  SimpleCategoricalFunction c;

  // We have three categorical dimensions only.
  std::vector<bool> categoricalDimensions;
  categoricalDimensions.push_back(true);
  categoricalDimensions.push_back(true);
  categoricalDimensions.push_back(true);

  // The first category can take 5 values; the second can take 3; the third can
  // take 12.
  arma::Row<size_t> numCategories("5 3 12");

  // The initial point for our optimization will be to set all categories to 0.
  arma::mat params("0 0 0");

  // Now create the GridSearch optimizer with default parameters, and run the
  // optimization.
  // The GridSearch type can be replaced with any ensmallen optimizer that
  // is able to handle categorical functions.
  GridSearch gs;
  gs.Optimize(c, params, categoricalDimensions, numCategories);

  REQUIRE(params(0) == 0);
  REQUIRE(params(1) == 2);
  REQUIRE(params(2) == 1);
}

TEST_CASE("GridSearchFMatTest", "[GridSearchTest]")
{
  // Create and optimize the categorical function with the GridSearch
  // optimizer.  We must also create a std::vector<bool> that holds the types
  // of each dimension, and an arma::Row<size_t> that holds the number of
  // categories in each dimension.
  SimpleCategoricalFunction c;

  // We have three categorical dimensions only.
  std::vector<bool> categoricalDimensions;
  categoricalDimensions.push_back(true);
  categoricalDimensions.push_back(true);
  categoricalDimensions.push_back(true);

  // The first category can take 5 values; the second can take 3; the third can
  // take 12.
  arma::Row<size_t> numCategories("5 3 12");

  // The initial point for our optimization will be to set all categories to 0.
  arma::fmat params("0 0 0");

  // Now create the GridSearch optimizer with default parameters, and run the
  // optimization.
  // The GridSearch type can be replaced with any ensmallen optimizer that
  // is able to handle categorical functions.
  GridSearch gs;
  gs.Optimize(c, params, categoricalDimensions, numCategories);

  REQUIRE(params(0) == 0);
  REQUIRE(params(1) == 2);
  REQUIRE(params(2) == 1);
}

TEST_CASE("GridSearchIMatTest", "[GridSearchTest]")
{
  // Create and optimize the categorical function with the GridSearch
  // optimizer.  We must also create a std::vector<bool> that holds the types
  // of each dimension, and an arma::Row<size_t> that holds the number of
  // categories in each dimension.
  SimpleCategoricalFunction c;

  // We have three categorical dimensions only.
  std::vector<bool> categoricalDimensions;
  categoricalDimensions.push_back(true);
  categoricalDimensions.push_back(true);
  categoricalDimensions.push_back(true);

  // The first category can take 5 values; the second can take 3; the third can
  // take 12.
  arma::Row<size_t> numCategories("5 3 12");

  // The initial point for our optimization will be to set all categories to 0.
  arma::imat params("0 0 0");

  // Now create the GridSearch optimizer with default parameters, and run the
  // optimization.
  // The GridSearch type can be replaced with any ensmallen optimizer that
  // is able to handle categorical functions.
  GridSearch gs;
  gs.Optimize(c, params, categoricalDimensions, numCategories);

  REQUIRE(params(0) == 0);
  REQUIRE(params(1) == 2);
  REQUIRE(params(2) == 1);
}
