/**
 * @file primal_dual_impl.hpp
 * @author Stephen Tu
 *
 * Contains an implementation of the "XZ+ZX" primal-dual infeasible interior
 * point method with a Mehrotra predictor-corrector update step presented and
 * analyzed in:
 *
 *   Primal-dual interior-point methods for semidefinite programming:
 *   Convergence rates, stability and numerical results.
 *   Farid Alizadeh, Jean-Pierre Haeberly, and Michael Overton.
 *   SIAM J. Optim. 1998.
 *   https://www.cs.nyu.edu/overton/papers/pdffiles/pdsdp.pdf
 *
 * We will refer to this paper as [AHO98] in this file.
 *
 * Note there are many optimizations that still need to be implemented. See the
 * code comments for more details.
 *
 * Also note the current implementation assumes the SDP problem has a strictly
 * feasible primal/dual point (and therefore the duality gap is zero), and
 * that the constraint matrices are linearly independent.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SDP_PRIMAL_DUAL_IMPL_HPP
#define ENSMALLEN_SDP_PRIMAL_DUAL_IMPL_HPP

#include "primal_dual.hpp"
#include "lin_alg.hpp"

namespace ens {
inline PrimalDualSolver::PrimalDualSolver(const size_t maxIterations,
                                          const double tau,
                                          const double normXzTol,
                                          const double primalInfeasTol,
                                          const double dualInfeasTol) :
    maxIterations(maxIterations),
    tau(tau),
    normXzTol(normXzTol),
    primalInfeasTol(primalInfeasTol),
    dualInfeasTol(dualInfeasTol)
{
  // Nothing to do.
}

/**
 * Compute
 *
 *     alpha = min(1, tau * alphahat(A, dA))
 *
 * where
 *
 *     alphahat = sup{ alphahat : A + dA is psd }
 *
 * See (2.18) of [AHO98] for more details.
 */
template<typename MatType>
static inline bool
Alpha(const MatType& a, const MatType& dA, double tau, double& alpha)
{
  arma::mat l;
  if (!arma::chol(l, a, "lower"))
    return false;

  arma::mat lInv;
  if (!arma::inv(lInv, arma::trimatl(l)))
    return false;

  // TODO(stephentu): We only want the top eigenvalue, we should
  // be able to do better than full eigen-decomposition.
  arma::Col<typename MatType::elem_type> evals;
  if (!arma::eig_sym(evals, -lInv * dA * lInv.t()))
    return false;
  const double alphahatInv = evals(evals.n_elem - 1);
  double alphahat = 1. / alphahatInv;

  if (alphahat < 0.)
    // dA is PSD already
    alphahat = 1.;
  alpha = std::min(1., tau * alphahat);
  return true;
}

/**
 * Solve the following Lyapunov equation (for X)
 *
 *   AX + XA = H
 *
 * where A, H are symmetric matrices.
 *
 * TODO(stephentu): Note this method current uses arma's builtin arma::syl
 * method, which is overkill for this situation. See Lemma 7.2 of [AHO98] for
 * how to solve this Lyapunov equation using an eigenvalue decomposition of A.
 *
 */
template<typename MatType, typename AType, typename BType>
static inline void
SolveLyapunov(MatType& x, const AType& a, const BType& h)
{
  arma::syl(x, a, a, -h);
}

/**
 * Solve the following KKT system (2.10) of [AHO98]:
 *
 *     [ 0  A^T  I ] [ dsx ] = [ rd ]
 *     [ A   0   0 ] [  dy ] = [ rp ]
 *     [ E   0   F ] [ dsz ] = [ rc ]
 *     \---- M ----/
 *
 * where
 *
 *     A  = [ Asparse ]
 *          [ Adense  ]
 *     dy = [ dysparse  dydense ]
 *     E  = Z sym I
 *     F  = X sym I
 *
 */
template<typename MatType,
         typename SparseConstraintType,
         typename DenseConstraintType>
static inline void
SolveKKTSystem(const SparseConstraintType& aSparse,
               const DenseConstraintType& aDense,
               const MatType& dualCoordinates,
               const MatType& m,
               const MatType& fMat,
               const MatType& rp,
               const MatType& rd,
               const MatType& rc,
               MatType& dsX,
               MatType& dySparse,
               MatType& dyDense,
               MatType& dsZ)
{
  MatType frdRcMat, eInvFrdRcMat, eInvFrdATdyRcMat, frdATdyRcMat;
  MatType eInvFrdRc, eInvFrdATdyRc, dy;

  // Note: Whenever a formula calls for E^(-1) v for some v, we solve Lyapunov
  // equations instead of forming an explicit inverse.

  // Compute the RHS of (2.12)
  math::Smat(fMat * rd - rc, frdRcMat);
  SolveLyapunov(eInvFrdRcMat, dualCoordinates, 2. * frdRcMat);
  math::Svec(eInvFrdRcMat, eInvFrdRc);

  MatType rhs = rp;
  const size_t numConstraints = aSparse.n_rows + aDense.n_rows;
  if (aSparse.n_rows)
    rhs(arma::span(0, aSparse.n_rows - 1), 0) += aSparse * eInvFrdRc;
  if (aDense.n_rows)
    rhs(arma::span(aSparse.n_rows, numConstraints - 1), 0) += aDense * eInvFrdRc;

  if (!arma::solve(dy, m, rhs, arma::solve_opts::fast))
  {
    throw std::logic_error("PrimalDualSolver::SolveKKTSystem(): Could not "
        "solve KKT system.");
  }

  MatType subTerm(aSparse.n_cols, 1, arma::fill::zeros);
  
  if (aSparse.n_rows)
  {
    dySparse = dy(arma::span(0, aSparse.n_rows - 1), 0);
    subTerm += aSparse.t() * dySparse;
  }
  if (aDense.n_rows)
  {
    dyDense = dy(arma::span(aSparse.n_rows, numConstraints - 1), 0);
    subTerm += aDense.t() * dyDense;
  }

  // Compute dx from (2.13)
  math::Smat(fMat * (rd - subTerm) - rc,
      frdATdyRcMat);
  SolveLyapunov(eInvFrdATdyRcMat, dualCoordinates, 2. * frdATdyRcMat);
  math::Svec(eInvFrdATdyRcMat, eInvFrdATdyRc);
  dsX = -eInvFrdATdyRc;

  // Compute dz from (2.14)
  dsZ = rd - subTerm;
}

template<typename SDPType, typename MatType, typename... CallbackTypes>
typename MatType::elem_type PrimalDualSolver::Optimize(
    const SDPType& sdp,
    MatType& coordinates,
    CallbackTypes&&... callbacks)
{
  // Initialize the other parameters and then call the other overload.
  MatType ySparse(arma::ones<MatType>(sdp.NumSparseConstraints(), 1));
  MatType yDense(arma::ones<MatType>(sdp.NumDenseConstraints(), 1));
  MatType z(arma::eye<MatType>(sdp.N(), sdp.N()));

  return Optimize(sdp, coordinates, ySparse, yDense, z, callbacks...);
}

template<typename SDPType, typename MatType, typename... CallbackTypes>
typename MatType::elem_type PrimalDualSolver::Optimize(
    const SDPType& sdp,
    MatType& coordinates,
    MatType& ySparse,
    MatType& yDense,
    MatType& dualCoordinates,
    CallbackTypes&&... callbacks)
{
  MatType tmp;

  // Note that the algorithm we implement requires primal iterate X and
  // dual multiplier Z to be positive definite (but not feasible).
  if (coordinates.n_rows != sdp.N() || coordinates.n_cols != sdp.N())
  {
    throw std::logic_error("PrimalDualSolver::Optimize(): coordinates needs to "
        "be square n x n matrix.");
  }

  if (!arma::chol(tmp, coordinates))
  {
    throw std::logic_error("PrimalDualSolver::Optimize(): coordinates needs to "
        "be symmetric positive definite.");
  }

  if (ySparse.n_cols != 1)
  {
    throw std::logic_error("PrimalDualSolver::Optimize(): ySparse must have "
        "only one column.");
  }

  if (ySparse.n_rows != sdp.NumSparseConstraints())
  {
    throw std::logic_error("PrimalDualSolver::Optimize(): ySparse needs to have"
        " the same length as the number of sparse constraints.");
  }

  if (yDense.n_cols != 1)
  {
    throw std::logic_error("PrimalDualSolver::Optimize(): yDense must have only"
        " one column.");
  }

  if (yDense.n_rows != sdp.NumDenseConstraints())
  {
    throw std::logic_error("PrimalDualSolver::Optimize(): yDense needs to have "
        "the same length as the number of dense constraints.");
  }

  if (dualCoordinates.n_rows != sdp.N() || dualCoordinates.n_cols != sdp.N())
  {
    throw std::logic_error("PrimalDualSolver::Optimize(): dualCoordinates needs"
        " to be square n x n matrix.");
  }

  if (!arma::chol(tmp, dualCoordinates))
  {
    throw std::logic_error("PrimalDualSolver::Optimize(): dualCoordinates needs"
        " to be symmetric positive definite.");
  }

  const size_t n = sdp.N();
  const size_t n2bar = sdp.N2bar();

  // Form the A matrix in (2.7). Note we explicitly handle
  // sparse and dense constraints separately.

  typename SDPType::SparseConstraintType aSparse(sdp.NumSparseConstraints(),
                                                 n2bar);
  typename SDPType::SparseConstraintType aiSparse;

  for (size_t i = 0; i < sdp.NumSparseConstraints(); i++)
  {
    math::Svec(sdp.SparseA()[i], aiSparse);
    aSparse.row(i) = aiSparse.t();
  }

  typename SDPType::DenseConstraintType aDense(sdp.NumDenseConstraints(),
                                               n2bar);
  typename SDPType::DenseConstraintType aiDense;
  for (size_t i = 0; i < sdp.NumDenseConstraints(); i++)
  {
    math::Svec(sdp.DenseA()[i], aiDense);
    aDense.row(i) = aiDense.t();
  }

  typename SDPType::ObjectiveType sc;
  math::Svec(sdp.C(), sc);

  MatType sx, sz, dySparse, dyDense, dsx, dsz, dX, dZ;

  math::Svec(coordinates, sx);
  math::Svec(dualCoordinates, sz);

  MatType rp, rd, rc, gk;

  MatType rcMat, fMat, eInvFaSparseT, eInvFaDenseT, gkMat,
      m, dualCheck;

  rp.set_size(sdp.NumConstraints(), 1);

  eInvFaSparseT.set_size(n2bar, sdp.NumSparseConstraints());
  eInvFaDenseT.set_size(n2bar, sdp.NumDenseConstraints());
  m.zeros(sdp.NumConstraints(), sdp.NumConstraints());

  // Controls early termination of the optimization process.
  bool terminate = false;

  typename SDPType::ElemType primalObj = 0., alpha, beta;
  Callback::BeginOptimization(*this, sdp, coordinates, callbacks...);
  for (size_t iteration = 1; iteration != maxIterations && !terminate;
      iteration++)
  {
    // Note: The Mehrotra PC algorithm works like this at a high level.
    // We first solve a KKT system with mu=0. Then, we use the results
    // of this KKT system to get a better estimate of mu and solve
    // the KKT system again. Empirically, this PC step has been shown to
    // significantly reduce the number of required iterations (and is used
    // by most practical solver implementations).
    if (sdp.NumSparseConstraints())
    {
      rp(arma::span(0, sdp.NumSparseConstraints() - 1), 0) =
          sdp.SparseB() - aSparse * sx;
    }
    if (sdp.NumDenseConstraints())
    {
      rp(arma::span(sdp.NumSparseConstraints(), sdp.NumConstraints() - 1), 0) =
          sdp.DenseB() - aDense * sx;
    }

    // Rd = C - Z - smat A^T y
    rd = sc - sz - aSparse.t() * ySparse - aDense.t() * yDense;

    math::SymKronId(coordinates, fMat);

    // We compute E^(-1) F A^T by solving Lyapunov equations.
    // See (2.16).
    for (size_t i = 0; i < sdp.NumSparseConstraints(); i++)
    {
      SolveLyapunov(gkMat, dualCoordinates, coordinates * sdp.SparseA()[i] +
          sdp.SparseA()[i] * coordinates);
      math::Svec(gkMat, gk);
      eInvFaSparseT.col(i) = gk;
    }

    for (size_t i = 0; i < sdp.NumDenseConstraints(); i++)
    {
      SolveLyapunov(gkMat, dualCoordinates, coordinates * sdp.DenseA()[i] +
          sdp.DenseA()[i] * coordinates);
      math::Svec(gkMat, gk);
      eInvFaDenseT.col(i) = gk;
    }

    // Form the M = A E^(-1) F A^T matrix (2.15)
    //
    // Since we split A up into its sparse and dense components,
    // we have to handle each block separately.
    if (sdp.NumSparseConstraints())
    {
      m.submat(arma::span(0, sdp.NumSparseConstraints() - 1),
               arma::span(0, sdp.NumSparseConstraints() - 1)) =
          aSparse * eInvFaSparseT;
      if (sdp.NumDenseConstraints())
      {
        m.submat(arma::span(0, sdp.NumSparseConstraints() - 1),
                 arma::span(sdp.NumSparseConstraints(),
                            sdp.NumConstraints() - 1)) =
            aSparse * eInvFaDenseT;
      }
    }
    if (sdp.NumDenseConstraints())
    {
      if (sdp.NumSparseConstraints())
      {
        m.submat(arma::span(sdp.NumSparseConstraints(),
                            sdp.NumConstraints() - 1),
                 arma::span(0,
                            sdp.NumSparseConstraints() - 1)) =
            aDense * eInvFaSparseT;
      }
      m.submat(arma::span(sdp.NumSparseConstraints(),
                          sdp.NumConstraints() - 1),
               arma::span(sdp.NumSparseConstraints(),
                          sdp.NumConstraints() - 1)) =
          aDense * eInvFaDenseT;
    }

    const typename MatType::elem_type sxdotsz = arma::dot(sx, sz);

    // TODO(stephentu): computing these alphahats should take advantage of
    // the cholesky decomposition of X and Z which we should have available
    // when we use more efficient methods above.

    // This solves step (1) of Section 7, the "predictor" step.
    rcMat = -0.5 * (coordinates * dualCoordinates + dualCoordinates * coordinates);
    math::Svec(rcMat, rc);
    SolveKKTSystem(aSparse, aDense, dualCoordinates, m, fMat, rp, rd, rc, dsx,
        dySparse, dyDense, dsz);
    math::Smat(dsx, dX);
    math::Smat(dsz, dZ);

    // Step (2), determine step size lengths (alpha, beta)
    bool success = Alpha(coordinates, dX, tau, alpha);
    if (!success)
    {
      Warn << "PrimalDualSolver::Optimize(): cholesky decomposition of X "
          << "failed!  Terminating optimization.";

      Callback::EndOptimization(*this, sdp, coordinates, callbacks...);
      return primalObj;
    }

    success = Alpha(dualCoordinates, dZ, tau, beta);
    if (!success)
    {
      Warn << "PrimalDualSolver::Optimize(): cholesky decomposition of Z "
          << "failed!  Terminating optimization.";

      Callback::EndOptimization(*this, sdp, coordinates, callbacks...);
      return primalObj;
    }

    // See (7.1)
    const double sigma = std::pow(arma::dot(coordinates + alpha * dX,
                                            dualCoordinates + beta * dZ) /
        sxdotsz, 3);
    const double mu = sigma * sxdotsz / n;

    // Step (3), the "corrector" step.
    rcMat = mu * arma::eye<MatType>(n, n) - 0.5 *
        (coordinates * dualCoordinates +
         dualCoordinates * coordinates +
         dX * dZ +
         dZ * dX);
    math::Svec(rcMat, rc);
    SolveKKTSystem(aSparse, aDense, dualCoordinates, m, fMat, rp, rd, rc, dsx,
        dySparse, dyDense, dsz);
    math::Smat(dsx, dX);
    math::Smat(dsz, dZ);
    if (!Alpha(coordinates, dX, tau, alpha))
    {
      Warn << "PrimalDualSolver::Optimize(): cholesky decomposition of X "
          << "failed!  Terminating optimization.";

      Callback::EndOptimization(*this, sdp, coordinates, callbacks...);
      return primalObj;
    }
    if (!Alpha(dualCoordinates, dZ, tau, beta))
    {
      Warn << "PrimalDualSolver::Optimize(): cholesky decomposition of Z "
          << "failed!  Terminating optimization.";

      Callback::EndOptimization(*this, sdp, coordinates, callbacks...);
      return primalObj;
    }

    // Iterate update
    coordinates += alpha * dX;
    terminate |= Callback::StepTaken(*this, sdp, coordinates, callbacks...);

    math::Svec(coordinates, sx);
    if (dySparse.n_cols != 0)
      ySparse += beta * dySparse;
    if (dyDense.n_cols != 0)
      yDense += beta * dyDense;
    dualCoordinates += beta * dZ;
    math::Svec(dualCoordinates, sz);

    // Below, we check the KKT conditions. Recall the KKT conditions are
    //
    // (1) Primal feasibility
    // (2) Dual feasibility
    // (3) XZ = 0 (slackness condition)
    //
    // If the KKT conditions are satisfied to a certain degree of precision,
    // then we consider this a valid certificate of optimality and terminate.
    // Otherwise, we proceed onwards.

    const double normXZ = arma::norm(coordinates * dualCoordinates, "fro");

    const double sparsePrimalInfeas = arma::norm(sdp.SparseB() - aSparse * sx,
        2);
    const double densePrimalInfeas = arma::norm(sdp.DenseB() - aDense * sx, 2);
    const double primalInfeas = sqrt(sparsePrimalInfeas * sparsePrimalInfeas +
        densePrimalInfeas * densePrimalInfeas);

    primalObj = arma::dot(sdp.C(), coordinates);

    // const double dualObj = arma::dot(sdp.SparseB(), ySparse) +
    //      arma::dot(sdp.DenseB(), yDense);
    // TODO: dualObj seems to be unused

    // const double dualityGap = primalObj - dualObj;
    // TODO: dualityGap seems to be unused

    // TODO(stephentu): this dual check is quite expensive,
    // maybe make it optional?
    dualCheck = dualCoordinates - sdp.C();
    for (size_t i = 0; i < sdp.NumSparseConstraints(); i++)
      dualCheck += ySparse(i) * sdp.SparseA()[i];
    for (size_t i = 0; i < sdp.NumDenseConstraints(); i++)
      dualCheck += yDense(i) * sdp.DenseA()[i];
    const double dualInfeas = arma::norm(dualCheck, "fro");

    if (normXZ <= normXzTol && primalInfeas <= primalInfeasTol &&
        dualInfeas <= dualInfeasTol)
      return primalObj;
  }

  Warn << "PrimalDualSolver::Optimizer(): Did not converge after "
      << maxIterations << " iterations!" << std::endl;

  Callback::EndOptimization(*this, sdp, coordinates, callbacks...);
  return primalObj;
}

} // namespace ens

#endif
