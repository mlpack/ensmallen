/**
 * @file frankwolfe_test.cpp
 * @author Chenzhe Diao
 * @author Marcus Edel
 *
 * Test file for Frank-Wolfe type optimizer.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <ensmallen.hpp>
#include "catch.hpp"
#include "test_function_tools.hpp"

using namespace arma;
using namespace ens;
using namespace ens::test;

/**
 * Simple test of Orthogonal Matching Pursuit algorithm.
 */
TEST_CASE("FWOMPTest", "[FrankWolfeTest]")
{
  const int k = 5;
  mat B1 = eye(3, 3);
  mat B2 = 0.1 * randn(3, k);
  mat A = join_horiz(B1, B2); // The dictionary is input as columns of A.
  vec b = {1.0, 1.0, 0.0}; // Vector to be sparsely approximated.

  FuncSq f(A, b);
  ConstrLpBallSolver linearConstrSolver(1);
  UpdateSpan updateRule;

  OMP s(linearConstrSolver, updateRule);

  mat coordinates = zeros<mat>(k + 3, 1);
  double result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(0.0).margin(1e-10));
  REQUIRE(coordinates(0) - 1 == Approx(0.0).margin(1e-10));
  REQUIRE(coordinates(1) - 1 == Approx(0.0).margin(1e-10));
  REQUIRE(coordinates(2) == Approx(0.0).margin(1e-10));
  for (int ii = 0; ii < k; ++ii)
  {
    REQUIRE(coordinates[ii + 3] == Approx(0.0).margin(1e-10));
  }
}

/**
 * Simple test of Orthogonal Matching Pursuit with regularization.
 */
TEST_CASE("FWRegularizedOMP", "[FrankWolfeTest]")
{
  const int k = 10;
  mat B1 = 0.1 * eye(k, k);
  mat B2 = 100 * randn(k, k);
  mat A = join_horiz(B1, B2); // The dictionary is input as columns of A.
  vec b(k, arma::fill::zeros); // Vector to be sparsely approximated.
  b(0) = 1;
  b(1) = 1;
  vec lambda(A.n_cols);
  for (size_t ii = 0; ii < A.n_cols; ii++)
    lambda(ii) = norm(A.col(ii), 2);

  FuncSq f(A, b);
  ConstrLpBallSolver linearConstrSolver(1, lambda);
  UpdateSpan updateRule;

  OMP s(linearConstrSolver, updateRule);

  mat coordinates = zeros<mat>(2 * k, 1);
  double result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(0.0).margin(1e-10));
}

/**
 * Simple test of Orthogonal Matching Pursuit with support prune.
 */
TEST_CASE("FWPruneSupportOMP", "[FrankWolfeTest]")
{
  // The dictionary is input as columns of A.
  const int k = 3;
  mat B1 = { { 1.0, 0.0, 1.0 },
             { 0.0, 1.0, 1.0 },
             { 0.0, 0.0, 1.0 } };
  mat B2 = randu(k, k);
  mat A = join_horiz(B1, B2); // The dictionary is input as columns of A.
  vec b = { 1.0, 1.0, 0.0 }; // Vector to be sparsely approximated.

  FuncSq f(A, b);
  ConstrLpBallSolver linearConstrSolver(1);
  UpdateSpan updateRule(true);

  OMP s(linearConstrSolver, updateRule);

  mat coordinates = zeros<mat>(k + 3, 1);
  double result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(0.0).margin(1e-10));
}

/**
 * Simple test of sparse soluton in atom domain with atom norm constraint.
 */
TEST_CASE("FWAtomNormConstraint", "[FrankWolfeTest]")
{
  const int k = 5;
  mat B1 = eye(3, 3);
  mat B2 = 0.1 * randn(3, k);
  mat A = join_horiz(B1, B2); // The dictionary is input as columns of A.
  vec b = { 1.0, 1.0, 0.0 }; // Vector to be sparsely approximated.

  FuncSq f(A, b);
  ConstrLpBallSolver linearConstrSolver(1);
  UpdateFullCorrection updateRule(2, 0.2);

  FrankWolfe<ConstrLpBallSolver, UpdateFullCorrection>
    s(linearConstrSolver, updateRule);

  mat coordinates = zeros<mat>(k + 3, 1);
  double result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(0.0).margin(1e-10));
}


/**
 * A very simple test of classic Frank-Wolfe algorithm.
 * The constrained domain used is unit lp ball.
 */
TEST_CASE("ClassicFW", "[FrankWolfeTest]")
{
  TestFuncFW<> f;
  double p = 2;   // Constraint set is unit lp ball.
  ConstrLpBallSolver linearConstrSolver(p);
  UpdateClassic updateRule;

  FrankWolfe<ConstrLpBallSolver, UpdateClassic>
      s(linearConstrSolver, updateRule);

  mat coordinates = randu<mat>(3, 1);
  double result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(0.0).margin(1e-4));
  REQUIRE(coordinates(0) - 0.1 == Approx(0.0).margin(1e-4));
  REQUIRE(coordinates(1) - 0.2 == Approx(0.0).margin(1e-4));
  REQUIRE(coordinates(2) - 0.3 == Approx(0.0).margin(1e-4));
}

/**
 * A very simple test of classic Frank-Wolfe algorithm.
 * The constrained domain used is unit lp ball.
 * Use arma::fmat.
 */
TEST_CASE("ClassicFWFMat", "[FrankWolfeTest]")
{
  TestFuncFW<arma::fmat> f;
  double p = 2;   // Constraint set is unit lp ball.
  ConstrLpBallSolver linearConstrSolver(p);
  UpdateClassic updateRule;

  FrankWolfe<ConstrLpBallSolver, UpdateClassic>
      s(linearConstrSolver, updateRule);

  fmat coordinates = randu<fmat>(3, 1);
  float result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(0.0).margin(1e-4));
  REQUIRE(coordinates(0) - 0.1 == Approx(0.0).margin(1e-4));
  REQUIRE(coordinates(1) - 0.2 == Approx(0.0).margin(1e-4));
  REQUIRE(coordinates(2) - 0.3 == Approx(0.0).margin(1e-4));
}

/**
 * Exactly the same problem with ClassicFW.
 * The update step performs a line search now.
 * It converges much faster.
 */
TEST_CASE("FWLineSearch", "[FrankWolfeTest]")
{
  TestFuncFW<> f;
  double p = 2;   // Constraint set is unit lp ball.
  ConstrLpBallSolver linearConstrSolver(p);
  UpdateLineSearch updateRule;

  FrankWolfe<ConstrLpBallSolver, UpdateLineSearch>
      s(linearConstrSolver, updateRule);

  mat coordinates = randu<mat>(3);
  double result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(0.0).margin(1e-4));
  REQUIRE(coordinates(0) - 0.1 == Approx(0.0).margin(1e-4));
  REQUIRE(coordinates(1) - 0.2 == Approx(0.0).margin(1e-4));
  REQUIRE(coordinates(2) - 0.3 == Approx(0.0).margin(1e-4));
}

/**
 * Exactly the same problem with ClassicFW.
 * The update step performs a line search now.
 * It converges much faster.
 * Use arma::fmat.
 */
TEST_CASE("FWLineSearchFMat", "[FrankWolfeTest]")
{
  TestFuncFW<arma::fmat> f;
  double p = 2;   // Constraint set is unit lp ball.
  ConstrLpBallSolver linearConstrSolver(p);
  UpdateLineSearch updateRule;

  FrankWolfe<ConstrLpBallSolver, UpdateLineSearch>
      s(linearConstrSolver, updateRule);

  fmat coordinates = randu<fmat>(3);
  float result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(0.0).margin(1e-4));
  REQUIRE(coordinates(0) - 0.1 == Approx(0.0).margin(1e-4));
  REQUIRE(coordinates(1) - 0.2 == Approx(0.0).margin(1e-4));
  REQUIRE(coordinates(2) - 0.3 == Approx(0.0).margin(1e-4));
}
