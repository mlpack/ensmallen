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
#if defined(ENS_USE_COOT)
  #include <armadillo>
  #include <bandicoot>
#endif
#include <ensmallen.hpp>
#include "catch.hpp"
#include "test_function_tools.hpp"
#include "test_types.hpp"

using namespace arma;
using namespace ens;
using namespace ens::test;

/**
 * Simple test of Orthogonal Matching Pursuit algorithm.
 */
TEMPLATE_TEST_CASE("FrankWolfe_OMP", "[FrankWolfe]", arma::mat)
{
  typedef typename TestType::elem_type ElemType;

  const int k = 5;
  TestType B1 = arma::eye<TestType>(3, 3);
  TestType B2 = 0.1 * arma::randn<TestType>(3, k);
  // The dictionary is input as columns of A.
  TestType A = join_horiz(B1, B2);
  // Vector to be sparsely approximated.
  arma::Col<ElemType> b = {1.0, 1.0, 0.0};

  FuncSq f(A, b);
  ConstrLpBallSolver linearConstrSolver(1);
  UpdateSpan updateRule;

  OMP s(linearConstrSolver, updateRule);

  TestType coordinates = zeros<TestType>(k + 3, 1);
  ElemType result = s.Optimize(f, coordinates);

  const double margin = Tolerances<TestType>::Coord;

  REQUIRE(result == Approx(0.0).margin(margin));
  REQUIRE(coordinates(0) - 1 == Approx(0.0).margin(margin));
  REQUIRE(coordinates(1) - 1 == Approx(0.0).margin(margin));
  REQUIRE(coordinates(2) == Approx(0.0).margin(margin));
  for (int ii = 0; ii < k; ++ii)
  {
    REQUIRE(coordinates[ii + 3] == Approx(0.0).margin(margin));
  }
}

/**
 * Simple test of Orthogonal Matching Pursuit with regularization.
 */
TEMPLATE_TEST_CASE("FrankWolfe_RegularizedOMP", "[FrankWolfe]", arma::mat)
{
  typedef typename TestType::elem_type ElemType;

  const int k = 10;
  TestType B1 = 0.1 * arma::eye<TestType>(k, k);
  TestType B2 = 100 * arma::randn<TestType>(k, k);
  // The dictionary is input as columns of A.
  TestType A = join_horiz(B1, B2);
  // Vector to be sparsely approximated.
  arma::Col<ElemType> b(k, arma::fill::zeros);
  b(0) = 1;
  b(1) = 1;
  arma::Col<ElemType> lambda(A.n_cols);
  for (size_t ii = 0; ii < A.n_cols; ii++)
    lambda(ii) = norm(A.col(ii), 2);

  FuncSq f(A, b);
  ConstrLpBallSolver linearConstrSolver(1, lambda);
  UpdateSpan updateRule;

  OMP s(linearConstrSolver, updateRule);

  TestType coordinates = zeros<TestType>(2 * k, 1);
  ElemType result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(0.0).margin(Tolerances<TestType>::Coord));
}

/**
 * Simple test of Orthogonal Matching Pursuit with support prune.
 */
TEMPLATE_TEST_CASE("FrankWolfe_PruneSupportOMP", "[FrankWolfe]", arma::mat)
{
  typedef typename TestType::elem_type ElemType;

  // The dictionary is input as columns of A.
  const int k = 3;
  TestType B1 = { { 1.0, 0.0, 1.0 },
                  { 0.0, 1.0, 1.0 },
                  { 0.0, 0.0, 1.0 } };
  TestType B2 = arma::randu<TestType>(k, k);
  // The dictionary is input as columns of A.
  TestType A = join_horiz(B1, B2);
  // Vector to be sparsely approximated.
  arma::Col<ElemType> b = { 1.0, 1.0, 0.0 };

  FuncSq f(A, b);
  ConstrLpBallSolver linearConstrSolver(1);
  UpdateSpan updateRule(true);

  OMP s(linearConstrSolver, updateRule);

  TestType coordinates = zeros<TestType>(k + 3, 1);
  ElemType result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(0.0).margin(Tolerances<TestType>::Coord));
}

/**
 * Simple test of sparse soluton in atom domain with atom norm constraint.
 */
TEMPLATE_TEST_CASE("FrankWolfe_AtomNormConstraint", "[FrankWolfe]",
    arma::mat)
{
  typedef typename TestType::elem_type ElemType;

  const int k = 5;
  TestType B1 = arma::eye<TestType>(3, 3);
  TestType B2 = 0.1 * arma::randn<TestType>(3, k);
  // The dictionary is input as columns of A.
  TestType A = join_horiz(B1, B2);
  // Vector to be sparsely approximated.
  arma::Col<ElemType> b = { 1.0, 1.0, 0.0 };

  FuncSq f(A, b);
  ConstrLpBallSolver linearConstrSolver(1);
  UpdateFullCorrection updateRule(2, 0.2);

  FrankWolfe<ConstrLpBallSolver, UpdateFullCorrection>
      s(linearConstrSolver, updateRule);

  TestType coordinates = zeros<TestType>(k + 3, 1);
  ElemType result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(0.0).margin(Tolerances<TestType>::Coord));
}

/**
 * A very simple test of classic Frank-Wolfe algorithm.
 * The constrained domain used is unit lp ball.
 */
TEMPLATE_TEST_CASE("FrankWolfe_Classic", "[FrankWolfe]", arma::mat)
{
  typedef typename TestType::elem_type ElemType;

  TestFuncFW<TestType> f;
  double p = 2;   // Constraint set is unit lp ball.
  ConstrLpBallSolver linearConstrSolver(p);
  UpdateClassic updateRule;

  FrankWolfe<ConstrLpBallSolver, UpdateClassic>
      s(linearConstrSolver, updateRule);

  TestType coordinates = arma::randu<TestType>(3, 1);
  ElemType result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(0.0).margin(Tolerances<TestType>::Obj));
  const double coordTol = Tolerances<TestType>::Coord;
  REQUIRE(coordinates(0) - 0.1 == Approx(0.0).margin(coordTol));
  REQUIRE(coordinates(1) - 0.2 == Approx(0.0).margin(coordTol));
  REQUIRE(coordinates(2) - 0.3 == Approx(0.0).margin(coordTol));
}

/**
 * Exactly the same problem with ClassicFW.
 * The update step performs a line search now.
 * It converges much faster.
 */
TEMPLATE_TEST_CASE("FrankWolfe_LineSearch", "[FrankWolfe]", arma::mat)
{
  typedef typename TestType::elem_type ElemType;

  TestFuncFW<TestType> f;
  double p = 2;   // Constraint set is unit lp ball.
  ConstrLpBallSolver linearConstrSolver(p);
  UpdateLineSearch updateRule;

  FrankWolfe<ConstrLpBallSolver, UpdateLineSearch>
      s(linearConstrSolver, updateRule);

  TestType coordinates = arma::randu<TestType>(3);
  ElemType result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(0.0).margin(Tolerances<TestType>::Obj));
  const double coordTol = Tolerances<TestType>::Coord;
  REQUIRE(coordinates(0) - 0.1 == Approx(0.0).margin(coordTol));
  REQUIRE(coordinates(1) - 0.2 == Approx(0.0).margin(coordTol));
  REQUIRE(coordinates(2) - 0.3 == Approx(0.0).margin(coordTol));
}

#ifdef ENS_HAVE_COOT

TEMPLATE_TEST_CASE("FrankWolfe_LineSearch", "[FrankWolfe]",
    coot::mat, coot::fmat)
{
  typedef typename TestType::elem_type ElemType;

  TestFuncFW<TestType> f;
  double p = 2;   // Constraint set is unit lp ball.
  ConstrLpBallSolverType<coot::Row<ElemType> > linearConstrSolver(p);
  UpdateLineSearch updateRule;

  FrankWolfe<decltype(linearConstrSolver), UpdateLineSearch>
      s(linearConstrSolver, updateRule);

  TestType coordinates = coot::randu<TestType>(3);
  ElemType result = s.Optimize(f, coordinates);

  REQUIRE(result == Approx(0.0).margin(1e-4));
  REQUIRE(coordinates(0) - 0.1 == Approx(0.0).margin(1e-4));
  REQUIRE(coordinates(1) - 0.2 == Approx(0.0).margin(1e-4));
  REQUIRE(coordinates(2) - 0.3 == Approx(0.0).margin(1e-4));
}

#endif
