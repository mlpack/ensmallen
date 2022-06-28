/**
 * @file sdp_primal_dual_test.cpp
 * @author Stephen Tu
 * @author Marcus Edel
 * @author Conrad Sanderson
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#if (ARMA_VERSION_MAJOR < 11)
  #define ARMA_DONT_PRINT_ERRORS
#endif

#include <ensmallen.hpp>
#include "catch.hpp"

using namespace ens;
using namespace ens::test;

class UndirectedGraph
{
 public:
  UndirectedGraph() : numVertices(0) { }

  size_t NumVertices() const { return numVertices; }
  size_t NumEdges() const { return edges.n_cols; }

  const arma::umat& Edges() const { return edges; }
  const arma::vec& Weights() const { return weights; }

  void Laplacian(arma::sp_mat& laplacian) const
  {
    laplacian.zeros(numVertices, numVertices);

    for (size_t i = 0; i < edges.n_cols; ++i)
    {
      laplacian(edges(0, i), edges(1, i)) = -weights(i);
      laplacian(edges(1, i), edges(0, i)) = -weights(i);
    }

    for (size_t i = 0; i < numVertices; ++i)
    {
      laplacian(i, i) = -arma::accu(laplacian.row(i));
    }
  }

  static void LoadFromEdges(UndirectedGraph& g,
                            const std::string& edgesFilename,
                            bool transposeEdges)
  {
    
    // data::Load(edgesFilename, g.edges, true, transposeEdges);
    if (g.edges.load(edgesFilename) == false)  { FAIL("couldn't load data"); }
    if (transposeEdges)  { g.edges = g.edges.t(); }
    
    if (g.edges.n_rows != 2)
      FAIL("Invalid datafile");
    g.weights.ones(g.edges.n_cols);
    g.ComputeVertices();
  }

  static void LoadFromEdgesAndWeights(UndirectedGraph& g,
                                      const std::string& edgesFilename,
                                      bool transposeEdges,
                                      const std::string& weightsFilename,
                                      bool transposeWeights)
  {
    // data::Load(edgesFilename, g.edges, true, transposeEdges);
    if (g.edges.load(edgesFilename) == false)  { FAIL("couldn't load data"); }
    if (transposeEdges)  { g.edges = g.edges.t(); }

    if (g.edges.n_rows != 2)
      FAIL("Invalid datafile");
    
    // data::Load(weightsFilename, g.weights, true, transposeWeights);
    if (g.weights.load(weightsFilename) == false)  { FAIL("couldn't load data"); }
    if (transposeWeights)  { g.weights = g.weights.t(); }
    
    if (g.weights.n_elem != g.edges.n_cols)
      FAIL("Size mismatch");
    g.ComputeVertices();
  }

  static void ErdosRenyiRandomGraph(UndirectedGraph& g,
                                    size_t numVertices,
                                    double edgeProbability,
                                    bool weighted,
                                    bool selfLoops = false)
  {
    if (edgeProbability < 0. || edgeProbability > 1.)
      FAIL("edgeProbability not in [0, 1]");

    std::vector<std::pair<size_t, size_t>> edges;
    std::vector<double> weights;

    for (size_t i = 0; i < numVertices; i ++)
    {
      for (size_t j = (selfLoops ? i : i + 1); j < numVertices; j++)
      {
        if (arma::as_scalar(arma::randu(1)) > edgeProbability)
          continue;
        edges.emplace_back(i, j);
        weights.push_back(weighted ? double(arma::as_scalar(arma::randu(1))) : double(1));
      }
    }

    g.edges.set_size(2, edges.size());
    for (size_t i = 0; i < edges.size(); i++)
    {
      g.edges(0, i) = edges[i].first;
      g.edges(1, i) = edges[i].second;
    }
    g.weights = arma::vec(weights);

    g.numVertices = numVertices;
  }

 private:
  void ComputeVertices()
  {
    numVertices = max(max(edges)) + 1;
  }

  arma::umat edges;
  arma::vec weights;
  size_t numVertices;
};

static inline SDP<arma::sp_mat>
ConstructMaxCutSDPFromGraph(const UndirectedGraph& g)
{
  SDP<arma::sp_mat> sdp(g.NumVertices(), g.NumVertices(), 0);
  g.Laplacian(sdp.C());
  sdp.C() *= -1;
  for (size_t i = 0; i < g.NumVertices(); i++)
  {
    sdp.SparseA()[i].zeros(g.NumVertices(), g.NumVertices());
    sdp.SparseA()[i](i, i) = 1.;
  }
  sdp.SparseB().ones();
  return sdp;
}

static inline SDP<arma::mat>
ConstructLovaszThetaSDPFromGraph(const UndirectedGraph& g)
{
  SDP<arma::mat> sdp(g.NumVertices(), g.NumEdges() + 1, 0);
  sdp.C().ones();
  sdp.C() *= -1.;
  sdp.SparseA()[0].eye(g.NumVertices(), g.NumVertices());
  for (size_t i = 0; i < g.NumEdges(); i++)
  {
    sdp.SparseA()[i + 1].zeros(g.NumVertices(), g.NumVertices());
    sdp.SparseA()[i + 1](g.Edges()(0, i), g.Edges()(1, i)) = 1.;
    sdp.SparseA()[i + 1](g.Edges()(1, i), g.Edges()(0, i)) = 1.;
  }
  sdp.SparseB().zeros();
  sdp.SparseB()[0] = 1.;
  return sdp;
}

static inline SDP<arma::sp_mat>
ConstructMaxCutSDPFromLaplacian(const std::string& laplacianFilename)
{
  arma::mat laplacian;
  
  // data::Load(laplacianFilename, laplacian, true, false);
  if (laplacian.load(laplacianFilename) == false)  { FAIL("couldn't load data"); }
  
  if (laplacian.n_rows != laplacian.n_cols)
    FAIL("laplacian not square");
  SDP<arma::sp_mat> sdp(laplacian.n_rows, laplacian.n_rows, 0);
  sdp.C() = -arma::sp_mat(laplacian);
  for (size_t i = 0; i < laplacian.n_rows; i++)
  {
    sdp.SparseA()[i].zeros(laplacian.n_rows, laplacian.n_rows);
    sdp.SparseA()[i](i, i) = 1.;
  }
  sdp.SparseB().ones();
  return sdp;
}

static bool CheckPositiveSemiDefinite(const arma::mat& X)
{
  // TODO: Armadillo 9.300+ has .is_sympd()
  arma::vec evals;
  if (!arma::eig_sym(evals, X))
    return false;
  else
    return (evals(0) > 1e-20);
}

template <typename SDPType>
static bool CheckKKT(const SDPType& sdp,
                     const arma::mat& X,
                     const arma::vec& ysparse,
                     const arma::vec& ydense,
                     const arma::mat& Z)
{
  // Require that the KKT optimality conditions for sdp are satisfied
  // by the primal-dual pair (X, y, Z).

  if (!CheckPositiveSemiDefinite(X))
    return false;
  if (!CheckPositiveSemiDefinite(Z))
    return false;

  bool success = true;
  const double normXz = arma::norm(X * Z, "fro");
  success &= (std::abs(normXz) < 1e-5);

  for (size_t i = 0; i < sdp.NumSparseConstraints(); i++)
  {
    success &= (std::abs(
        arma::dot(sdp.SparseA()[i], X) - sdp.SparseB()[i]) < 1e-5);
  }

  for (size_t i = 0; i < sdp.NumDenseConstraints(); i++)
  {
    success &= (std::abs(
        arma::dot(sdp.DenseA()[i], X) - sdp.DenseB()[i]) < 1e-5);
  }

  arma::mat dualCheck = Z - sdp.C();
  for (size_t i = 0; i < sdp.NumSparseConstraints(); i++)
    dualCheck += ysparse(i) * sdp.SparseA()[i];
  for (size_t i = 0; i < sdp.NumDenseConstraints(); i++)
    dualCheck += ydense(i) * sdp.DenseA()[i];
  const double dualInfeas = arma::norm(dualCheck, "fro");
  success &= (dualInfeas < 1e-5);

  return success;
}

static void SolveMaxCutFeasibleSDP(const SDP<arma::sp_mat>& sdp)
{
  arma::mat X, Z;
  arma::mat ysparse, ydense;
  ydense.set_size(0);

  // strictly feasible starting point
  X.eye(sdp.N(), sdp.N());
  ysparse = -1.1 * arma::vec(arma::sum(arma::abs(sdp.C()), 0).t());
  Z = -arma::diagmat(ysparse) + sdp.C();

  PrimalDualSolver solver;

  solver.Optimize(sdp, X, ysparse, ydense, Z);
  CheckKKT(sdp, X, ysparse, ydense, Z);
}

static void SolveMaxCutPositiveSDP(const SDP<arma::sp_mat>& sdp)
{
  arma::mat X, Z;
  arma::mat ysparse, ydense;
  ydense.set_size(0);

  // infeasible, but positive starting point
  X = arma::eye<arma::mat>(sdp.N(), sdp.N());
  ysparse = arma::randu<arma::vec>(sdp.NumSparseConstraints());
  Z.eye(sdp.N(), sdp.N());

  PrimalDualSolver solver;
  solver.Optimize(sdp, X, ysparse, ydense, Z);
  CheckKKT(sdp, X, ysparse, ydense, Z);
}

TEST_CASE("SmallMaxCutSdp","[SdpPrimalDualTest]")
{
  auto sdp = ConstructMaxCutSDPFromLaplacian("data/r10.txt");
  SolveMaxCutFeasibleSDP(sdp);
  SolveMaxCutPositiveSDP(sdp);

  UndirectedGraph g;
  UndirectedGraph::ErdosRenyiRandomGraph(g, 10, 0.3, true);
  sdp = ConstructMaxCutSDPFromGraph(g);

  // the following was resulting in non-positive Z0 matrices on some
  // random instances.
  // SolveMaxCutFeasibleSDP(sdp);

  SolveMaxCutPositiveSDP(sdp);
}

// This test is deprecated and can be removed in ensmallen 2.10.0.
TEST_CASE("DeprecatedSmallLovaszThetaSdp", "[SdpPrimalDualTest]")
{
  UndirectedGraph g;
  UndirectedGraph::LoadFromEdges(g, "data/johnson8-4-4.csv", true);
  auto sdp = ConstructLovaszThetaSDPFromGraph(g);

  PrimalDualSolver solver;

  arma::mat X, Z;
  arma::mat ysparse, ydense;
  sdp.GetInitialPoints(X, ysparse, ydense, Z);
  solver.Optimize(sdp, X, ysparse, ydense, Z);
  CheckKKT(sdp, X, ysparse, ydense, Z);
}

TEST_CASE("SmallLovaszThetaSdp", "[SdpPrimalDualTest]")
{
  UndirectedGraph g;
  UndirectedGraph::LoadFromEdges(g, "data/johnson8-4-4.csv", true);
  auto sdp = ConstructLovaszThetaSDPFromGraph(g);

  PrimalDualSolver solver;

  arma::mat X, Z, ysparse, ydense;
  sdp.GetInitialPoints(X, ysparse, ydense, Z);
  solver.Optimize(sdp, X, ysparse, ydense, Z);
  CheckKKT(sdp, X, ysparse, ydense, Z);
}

static inline arma::sp_mat
RepeatBlockDiag(const arma::sp_mat& block, size_t repeat)
{
  assert(block.n_rows == block.n_cols);
  arma::sp_mat ret(block.n_rows * repeat, block.n_rows * repeat);
  ret.zeros();
  for (size_t i = 0; i < repeat; i++)
    ret(arma::span(i * block.n_rows, (i + 1) * block.n_rows - 1),
        arma::span(i * block.n_rows, (i + 1) * block.n_rows - 1)) = block;
  return ret;
}

static inline arma::sp_mat
BlockDiag(const std::vector<arma::sp_mat>& blocks)
{
  // assumes all blocks are the same size
  const size_t n = blocks.front().n_rows;
  assert(blocks.front().n_cols == n);
  arma::sp_mat ret(n * blocks.size(), n * blocks.size());
  ret.zeros();
  for (size_t i = 0; i < blocks.size(); i++)
    ret(arma::span(i * n, (i + 1) * n - 1),
        arma::span(i * n, (i + 1) * n - 1)) = blocks[i];
  return ret;
}

static inline SDP<arma::sp_mat>
ConstructLogChebychevApproxSdp(const arma::mat& A, const arma::vec& b)
{
  if (A.n_rows != b.n_elem)
    FAIL("A.n_rows != len(b)");
  const size_t p = A.n_rows;
  const size_t k = A.n_cols;

  // [0, 0, 0]
  // [0, 0, 1]
  // [0, 1, 0]
  arma::sp_mat cblock(3, 3);
  cblock(1, 2) = cblock(2, 1) = 1.;
  const arma::sp_mat C = RepeatBlockDiag(cblock, p);

  SDP<arma::sp_mat> sdp(C.n_rows, k + 1, 0);
  sdp.C() = C;
  sdp.SparseB().zeros();
  sdp.SparseB()[0] = -1;

  // [1, 0, 0]
  // [0, 0, 0]
  // [0, 0, 1]
  arma::sp_mat a0block(3, 3);
  a0block(0, 0) = a0block(2, 2) = 1.;
  sdp.SparseA()[0] = RepeatBlockDiag(a0block, p);
  sdp.SparseA()[0] *= -1.;

  for (size_t i = 0; i < k; i++)
  {
    std::vector<arma::sp_mat> blocks;
    for (size_t j = 0; j < p; j++)
    {
      arma::sp_mat block(3, 3);
      const double f = A(j, i) / b(j);
      // [ -a_j(i)/b_j     0        0 ]
      // [      0       a_j(i)/b_j  0 ]
      // [      0          0        0 ]
      block(0, 0) = -f;
      block(1, 1) = f;
      blocks.emplace_back(block);
    }
    sdp.SparseA()[i + 1] = BlockDiag(blocks);
    sdp.SparseA()[i + 1] *= -1;
  }

  return sdp;
}

static inline arma::mat
RandomOrthogonalMatrix(size_t rows, size_t cols)
{
  arma::mat Q, R;
  if (!arma::qr(Q, R, arma::randu<arma::mat>(rows, cols)))
    FAIL("could not compute QR decomposition");
  return Q;
}

static inline arma::mat
RandomFullRowRankMatrix(size_t rows, size_t cols)
{
  const arma::mat U = RandomOrthogonalMatrix(rows, rows);
  const arma::mat V = RandomOrthogonalMatrix(cols, cols);
  arma::mat S;
  S.zeros(rows, cols);
  for (size_t i = 0; i < std::min(rows, cols); i++)
  {
    S(i, i) = arma::as_scalar(arma::randu(1)) + 1e-3;
  }
  return U * S * V;
}

/**
 * See the examples section, Eq. 9, of
 *
 *   Semidefinite Programming.
 *   Lieven Vandenberghe and Stephen Boyd.
 *   SIAM Review. 1996.
 *
 * The logarithmic Chebychev approximation to Ax = b, A is p x k and b is
 * length p is given by the SDP:
 *
 *   min    t
 *   s.t.
 *          [ t - dot(a_i, x)          0             0 ]
 *          [       0           dot(a_i, x) / b_i    1 ]  >= 0, i=1,...,p
 *          [       0                  1             t ]
 *
 */
TEST_CASE("LogChebychevApproxSdp","[SdpPrimalDualTest]")
{
  // Sometimes, the optimization can fail randomly, so we will run the test
  // three times and make sure it succeeds at least once.
  bool success = false;
  for (size_t i = 0; i < 3; ++i)
  {
    const size_t p0 = 5;
    const size_t k0 = 10;
    const arma::mat A0 = RandomFullRowRankMatrix(p0, k0);
    const arma::vec b0 = arma::randu<arma::vec>(p0);
    const auto sdp0 = ConstructLogChebychevApproxSdp(A0, b0);
    PrimalDualSolver solver0;
    arma::mat X0, Z0;
    arma::mat ysparse0, ydense0;
    sdp0.GetInitialPoints(X0, ysparse0, ydense0, Z0);
    solver0.Optimize(sdp0, X0, ysparse0, ydense0, Z0);
    success = CheckKKT(sdp0, X0, ysparse0, ydense0, Z0);
    if (success)
      break;
  }

  REQUIRE(success == true);

  success = false;
  for (size_t i = 0; i < 3; ++i)
  {
    const size_t p1 = 10;
    const size_t k1 = 5;
    const arma::mat A1 = RandomFullRowRankMatrix(p1, k1);
    const arma::vec b1 = arma::randu<arma::vec>(p1);
    const auto sdp1 = ConstructLogChebychevApproxSdp(A1, b1);
    PrimalDualSolver solver1;
    arma::mat X1, Z1;
    arma::mat ysparse1, ydense1;
    sdp1.GetInitialPoints(X1, ysparse1, ydense1, Z1);
    solver1.Optimize(sdp1, X1, ysparse1, ydense1, Z1);
    success = CheckKKT(sdp1, X1, ysparse1, ydense1, Z1);
    if (success)
      break;
  }

  REQUIRE(success == true);
}

/**
 * Example 1 on the SDP wiki
 *
 *   min   x_13
 *   s.t.
 *         -0.2 <= x_12 <= -0.1
 *          0.4 <= x_23 <=  0.5
 *          x_11 = x_22 = x_33 = 1
 *          X >= 0
 *
 */
TEST_CASE("CorrelationCoeffToySdp","[SdpPrimalDualTest]")
{
  // The semi-definite constraint looks like:
  //
  // [ 1  x_12  x_13  0  0  0  0 ]
  // [     1    x_23  0  0  0  0 ]
  // [            1   0  0  0  0 ]
  // [               s1  0  0  0 ]  >= 0
  // [                  s2  0  0 ]
  // [                     s3  0 ]
  // [                        s4 ]


  // x_11 == 0
  arma::sp_mat A0(7, 7); A0.zeros();
  A0(0, 0) = 1.;

  // x_22 == 0
  arma::sp_mat A1(7, 7); A1.zeros();
  A1(1, 1) = 1.;

  // x_33 == 0
  arma::sp_mat A2(7, 7); A2.zeros();
  A2(2, 2) = 1.;

  // x_12 <= -0.1  <==>  x_12 + s1 == -0.1, s1 >= 0
  arma::sp_mat A3(7, 7); A3.zeros();
  A3(1, 0) = A3(0, 1) = 1.; A3(3, 3) = 2.;

  // -0.2 <= x_12  <==>  x_12 - s2 == -0.2, s2 >= 0
  arma::sp_mat A4(7, 7); A4.zeros();
  A4(1, 0) = A4(0, 1) = 1.; A4(4, 4) = -2.;

  // x_23 <= 0.5  <==>  x_23 + s3 == 0.5, s3 >= 0
  arma::sp_mat A5(7, 7); A5.zeros();
  A5(2, 1) = A5(1, 2) = 1.; A5(5, 5) = 2.;

  // 0.4 <= x_23  <==>  x_23 - s4 == 0.4, s4 >= 0
  arma::sp_mat A6(7, 7); A6.zeros();
  A6(2, 1) = A6(1, 2) = 1.; A6(6, 6) = -2.;

  std::vector<arma::sp_mat> ais({A0, A1, A2, A3, A4, A5, A6});

  SDP<arma::sp_mat> sdp(7, 7 + 4 + 4 + 4 + 3 + 2 + 1, 0);

  for (size_t j = 0; j < 3; j++)
  {
    // x_j4 == x_j5 == x_j6 == x_j7 == 0
    for (size_t i = 0; i < 4; i++)
    {
      arma::sp_mat A(7, 7); A.zeros();
      A(i + 3, j) = A(j, i + 3) = 1;
      ais.emplace_back(A);
    }
  }

  // x_45 == x_46 == x_47 == 0
  for (size_t i = 0; i < 3; i++)
  {
    arma::sp_mat A(7, 7); A.zeros();
    A(i + 4, 3) = A(3, i + 4) = 1;
    ais.emplace_back(A);
  }

  // x_56 == x_57 == 0
  for (size_t i = 0; i < 2; i++)
  {
    arma::sp_mat A(7, 7); A.zeros();
    A(i + 5, 4) = A(4, i + 5) = 1;
    ais.emplace_back(A);
  }

  // x_67 == 0
  arma::sp_mat A(7, 7); A.zeros();
  A(6, 5) = A(5, 6) = 1;
  ais.emplace_back(A);

  std::swap(sdp.SparseA(), ais);

  sdp.SparseB().zeros();

  sdp.SparseB()[0] = sdp.SparseB()[1] = sdp.SparseB()[2] = 1.;

  sdp.SparseB()[3] = -0.2; sdp.SparseB()[4] = -0.4;

  sdp.SparseB()[5] = 1.; sdp.SparseB()[6] = 0.8;

  sdp.C().zeros();
  sdp.C()(0, 2) = sdp.C()(2, 0) = 1.;

  PrimalDualSolver solver;
  arma::mat X, Z;
  arma::mat ysparse, ydense;
  sdp.GetInitialPoints(X, ysparse, ydense, Z);
  const double obj = solver.Optimize(sdp, X, ysparse, ydense, Z);
  bool success = CheckKKT(sdp, X, ysparse, ydense, Z);
  REQUIRE(success == true);
  REQUIRE(obj == Approx(2 * (-0.978)).epsilon(1e-5));
}
