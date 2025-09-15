/**
 * @file lrsdp_test.cpp
 * @author Ryan Curtin
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
#include "test_types.hpp"

using namespace arma;
using namespace ens;
using namespace ens::test;

/**
 * Create a Lovasz-Theta initial point.
 */
template<typename MatType>
void CreateLovaszThetaInitialPoint(const MatType& edges,
                                   MatType& coordinates)
{
  // Get the number of vertices in the problem.
  const size_t vertices = max(max(edges)) + 1;

  const size_t m = edges.n_cols + 1;
  float r = 0.5 + sqrt(0.25 + 2 * m);
  if (ceil(r) > vertices)
    r = vertices; // An upper bound on the dimension.

  coordinates.set_size(vertices, ceil(r));

  // Now we set the entries of the initial matrix according to the formula given
  // in Section 4 of Monteiro and Burer.
  for (size_t i = 0; i < vertices; ++i)
  {
    for (size_t j = 0; j < ceil(r); ++j)
    {
      if (i == j)
        coordinates(i, j) = sqrt(1.0 / r) + sqrt(1.0 / (vertices * m));
      else
        coordinates(i, j) = sqrt(1.0 / (vertices * m));
    }
  }
}

/**
 * Prepare an LRSDP object to solve the Lovasz-Theta SDP in the manner detailed
 * in Monteiro + Burer 2004.  The list of edges in the graph must be given; that
 * is all that is necessary to set up the problem.  A matrix which will contain
 * initial point coordinates should be given also.
 */
template<typename MatType>
void SetupLovaszTheta(const MatType& edges,
                      LRSDP<SDP<MatType>>& lovasz,
                      typename ForwardType<MatType>::bvec& lambda)
{
  // Get the number of vertices in the problem.
  const size_t vertices = max(max(edges)) + 1;

  // C = -(e e^T) = -ones().
  lovasz.SDP().C().ones(vertices, vertices);
  lovasz.SDP().C() *= -1;

  // b_0 = 1; else = 0.
  lovasz.SDP().SparseB().zeros(edges.n_cols + 1);
  lovasz.SDP().SparseB()[0] = 1;

  // A_0 = I_n.
  lovasz.SDP().SparseA()[0].eye(vertices, vertices);

  // A_ij only has ones at (i, j) and (j, i) and 0 elsewhere.
  for (size_t i = 0; i < edges.n_cols; ++i)
  {
    lovasz.SDP().SparseA()[i + 1].zeros(vertices, vertices);
    lovasz.SDP().SparseA()[i + 1](edges(0, i), edges(1, i)) = 1.;
    lovasz.SDP().SparseA()[i + 1](edges(1, i), edges(0, i)) = 1.;
  }

  // Set the Lagrange multipliers right.
  lambda.ones(edges.n_cols + 1);
  lambda *= -1;
  lambda[0] = -((typename MatType::elem_type) vertices);
}

/**
 * johnson8-4-4.co test case for Lovasz-Theta LRSDP.
 * See Monteiro and Burer 2004.
 */
TEMPLATE_TEST_CASE("Johnson844LovaszThetaSDP", "[LRSDP]", ENS_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  // Load the edges.
  TestType edges;

  if (edges.load("data/johnson8-4-4.csv", csv_ascii) == false)
  {
    FAIL("couldn't load data");
    return;
  }

  edges = edges.t();

  // The LRSDP itself and the initial point.
  TestType coordinates;

  CreateLovaszThetaInitialPoint(edges, coordinates);

  LRSDP<SDP<TestType>> lovasz(edges.n_cols + 1, 0, coordinates);

  typedef typename ForwardType<TestType>::bvec VecType;
  VecType lambda;
  SetupLovaszTheta(edges, lovasz, lambda);

  double sigma = 10;
  ElemType finalValue = lovasz.Optimize(coordinates, lambda, sigma);

  // Final value taken from Monteiro + Burer 2004.
  REQUIRE(finalValue == Approx(-14.0).epsilon(Tolerances<TestType>::Obj));

  // Now ensure that all the constraints are satisfied.
  TestType rrt = coordinates * trans(coordinates);
  REQUIRE(trace(rrt) == Approx(1.0).epsilon(Tolerances<TestType>::Obj));

  // All those edge constraints...
  for (size_t i = 0; i < edges.n_cols; ++i)
  {
    REQUIRE(rrt(edges(0, i), edges(1, i)) ==
        Approx(0.0).margin(100 * Tolerances<TestType>::Obj));
    REQUIRE(rrt(edges(1, i), edges(0, i)) ==
        Approx(0.0).margin(100 * Tolerances<TestType>::Obj));
  }
}

/**
 * Create an unweighted graph laplacian from the edges.
 */
template<typename ElemType>
void CreateSparseGraphLaplacian(const Mat<ElemType>& edges,
                                SpMat<ElemType>& laplacian)
{
  // Get the number of vertices in the problem.
  const size_t vertices = max(max(edges)) + 1;

  laplacian.zeros(vertices, vertices);

  for (size_t i = 0; i < edges.n_cols; ++i)
  {
    laplacian(edges(0, i), edges(1, i)) = ElemType(-1);
    laplacian(edges(1, i), edges(0, i)) = ElemType(-1);
  }

  for (size_t i = 0; i < vertices; ++i)
  {
    laplacian(i, i) = -accu(laplacian.row(i));
  }
}

TEMPLATE_TEST_CASE("ErdosRenyiRandomGraphMaxCutSDP", "[LRSDP]", ENS_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  // Load the edges.
  TestType edges;

  if (edges.load("data/erdosrenyi-n100.csv", csv_ascii) == false)
  {
    FAIL("couldn't load data");
    return;
  }

  edges = edges.t();

  SpMat<ElemType> laplacian;
  CreateSparseGraphLaplacian(edges, laplacian);

  float r = 0.5 + sqrt(0.25 + 2 * edges.n_cols);
  if (ceil(r) > laplacian.n_rows)
    r = laplacian.n_rows;

  // initialize coordinates to a feasible point
  TestType coordinates(laplacian.n_rows, ceil(r));
  coordinates.zeros();
  for (size_t i = 0; i < coordinates.n_rows; ++i)
  {
    coordinates(i, i % coordinates.n_cols) = ElemType(1);
  }

  LRSDP<SDP<SpMat<ElemType>>> maxcut(laplacian.n_rows, 0, coordinates);
  maxcut.SDP().C() = laplacian;
  maxcut.SDP().C() *= -1.; // need to minimize the negative
  maxcut.SDP().SparseB().ones(laplacian.n_rows);
  for (size_t i = 0; i < laplacian.n_rows; ++i)
  {
    maxcut.SDP().SparseA()[i].zeros(laplacian.n_rows, laplacian.n_rows);
    maxcut.SDP().SparseA()[i](i, i) = ElemType(1);
  }

  const ElemType finalValue = maxcut.Optimize(coordinates);
  const TestType rrt = coordinates * trans(coordinates);

  for (size_t i = 0; i < laplacian.n_rows; ++i)
  {
    REQUIRE(rrt(i, i) == Approx(1.0).epsilon(Tolerances<TestType>::Obj));
  }

  // Final value taken by solving with Mosek
  REQUIRE(finalValue ==
      Approx(-3672.7).epsilon(100 * Tolerances<TestType>::Coord));
}

/*
 * Test a nuclear norm minimization SDP.
 *
 * Specifically, fix an unknown m x n matrix X. Our goal is to recover X from p
 * measurements of X, where the i-th measurement is of the form
 *
 *    b_i = dot(A_i, X)
 *
 * where the A_i's have iid entries from Normal(0, 1/p). We do this by solving
 * the the following semi-definite program
 *
 *    min ||X||_* subj to dot(A_i, X) = b_i, i=1,...,p
 *
 * where ||X||_* denotes the nuclear norm (sum of singular values) of X. The
 * equivalent SDP is
 *
 *    min tr(W1) + tr(W2) : [ W1, X ; X', W2 ] is PSD,
 *                          dot(A_i, X) = b_i, i = 1, ..., p
 *
 * For more details on matrix sensing and nuclear norm minimization, see
 *
 *    Guaranteed Minimum-Rank Solutions of Linear Matrix Equations via Nuclear
 *    Norm Minimization.
 *    Benjamin Recht, Maryam Fazel, Pablo Parrilo.
 *    SIAM Review 2010.
 *
 */
TEMPLATE_TEST_CASE("GaussianMatrixSensingSDP", "[LRSDP]", ENS_TEST_TYPES)
{
  typedef typename TestType::elem_type ElemType;

  TestType Xorig, A;

  // read the unknown matrix X and the measurement matrices A_i in

  if (Xorig.load("data/sensing_X.csv", csv_ascii) == false)
  {
    FAIL("couldn't load data");
  }

  if (A.load("data/sensing_A.csv", csv_ascii) == false)
  {
    FAIL("couldn't load data");
  }

  const size_t m = Xorig.n_rows;
  const size_t n = Xorig.n_cols;
  const size_t p = A.n_rows;
  assert(A.n_cols == m * m);

  Col<ElemType> b(p);
  for (size_t i = 0; i < p; ++i)
  {
    const TestType Ai = reshape(A.row(i), n, m);
    b(i) = dot(trans(Ai), Xorig);
  }

  float r = 0.5 + sqrt(0.25 + 2 * p);
  if (ceil(r) > m + n)
    r = m + n;

  TestType coordinates;
  coordinates.eye(m + n, ceil(r));

  LRSDP<SDP<SpMat<ElemType>>> sensing(0, p, coordinates, 15);
  sensing.SDP().C().eye(m + n, m + n);
  sensing.SDP().DenseB() = 2. * b;

  const span blockRows(0, m - 1);
  const span blockCols(m, m + n - 1);

  for (size_t i = 0; i < p; ++i)
  {
    const TestType Ai = reshape(A.row(i), n, m);
    sensing.SDP().DenseA()[i].zeros(m + n, m + n);
    sensing.SDP().DenseA()[i](blockRows, blockCols) = trans(Ai);
    sensing.SDP().DenseA()[i](blockCols, blockRows) = Ai;
  }

  ElemType finalValue = sensing.Optimize(coordinates);
  REQUIRE(finalValue == Approx(44.7550132629).epsilon(
      Tolerances<TestType>::LargeObj));

  const TestType rrt = coordinates * trans(coordinates);
  for (size_t i = 0; i < p; ++i)
  {
    const TestType Ai = reshape(A.row(i), n, m);
    const ElemType measurement = dot(trans(Ai), rrt(blockRows, blockCols));
    // Custom tolerances because floats can do very bad here.
    const ElemType eps = std::is_same<ElemType, float>::value ? 0.1 : 0.001;
    REQUIRE(measurement == Approx(b(i)).epsilon(eps));
  }

  // check matrix recovery
  const ElemType err = norm(Xorig - rrt(blockRows, blockCols), "fro") /
      norm(Xorig, "fro");
  REQUIRE(err == Approx(0.0).margin(10 * Tolerances<TestType>::LargeObj));
}
