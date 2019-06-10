/**
 * @file primal_dual.hpp
 * @author Stephen Tu
 *
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SDP_PRIMAL_DUAL_HPP
#define ENSMALLEN_SDP_PRIMAL_DUAL_HPP

#include "sdp.hpp"

namespace ens {

/**
 * PrimalDualSolver is a primal dual interior point solver for semidefinite
 * programs.
 *
 * PrimalDualSolver can optimize semidefinite programs.  For more details, see the
 * documentation on function types included with this distribution or on the
 * ensmallen website.
 *
 * @tparam DeprecatedSDPType Type of SDP to solve.  This parameter is deprecated
 *      and will be removed in ensmallen 2.10.0.
 */
template<typename DeprecatedSDPType = SDP<arma::mat>>
class PrimalDualSolver
{
 public:
  /**
   * Construct a new solver instance with the given optimization parameters.
   *
   * @param maxIterations Maximum number of iterations.
   * @param tau Step size modulating factor (between 0 and 1).
   * @param normXzTol Tolerance of norm(X*Z) required before terminating.  (X is
   *      the primal coordinates, Z is the dual coordinates.)
   * @param primalInfeasTol Primal infeasibility tolerance for termination.
   * @param dualInfeasTol Dual infeasibility tolerance for termination.
   */
  PrimalDualSolver(const size_t maxIterations = 1000,
                   const double tau = 0.99,
                   const double normXzTol = 1e-7,
                   const double primalInfeasTol = 1e-7,
                   const double dualInfeasTol = 1e-7);

  /**
   * Construct a new solver instance from a given SDP instance.  Uses a random,
   * positive initialization point.
   *
   * This constructor is deprecated.  Use the constructor that does not take an
   * SDPType and then call Optimize() with the SDP to be solved.
   *
   * This constructor will be removed in ensmallen 2.10.0.
   *
   * @param sdp Initialized SDP to be solved.
   */
  ens_deprecated PrimalDualSolver(const DeprecatedSDPType& sdp);

  /**
   * Construct a new solver instance with the given SDP instance and initial
   * points for optimization.
   *
   * This constructor is deprecated.  Use the constructor that does not take an
   * SDPType and then call Optimize() with the SDP to be solved.
   *
   * This constructor will be removed in ensmallen 2.10.0.
   *
   * @param sdp Initialized SDP to be solved.
   * @param initialX Initial primal point for optimization.
   * @param initialYSparse Initial y values for sparse constraints.
   * @param initialYDense Initial y values for dense constraints.
   * @param initialZ Initial dual point for optimization.
   */
  ens_deprecated PrimalDualSolver(const DeprecatedSDPType& sdp,
                                  const arma::mat& initialX,
                                  const arma::vec& initialYSparse,
                                  const arma::vec& initialYDense,
                                  const arma::mat& initialZ);

  /**
   * Optimize the stored SDP instance, storing the primal coordinates in X and
   * returning the primal objective value.  Any initial point in X will be
   * ignored.
   *
   * This function is deprecated and will be removed in ensmallen 2.10.0.  Use
   * the overload of Optimize() that takes an SDP.
   *
   * @param X Matrix to store final primal coordinates for optimization.
   * @return Primal objective value.
   */
  ens_deprecated double Optimize(arma::mat& X);

  /**
   * Optimize the stored SDP instance, storing the primal coordinates, dual
   * coordinates, and sparse and dense y values into the given matrices and
   * vectors.  The primal objective is returned.  Any initial setting of the
   * given matrices will be ignored.
   *
   * This function is deprecated and will be rmeoved in ensmallen 2.10.0.  Use
   * the overload of Optimize() that takes an SDP.
   *
   * @param X Matrix to store final primal coordinates into.
   * @param ySparse Vector to store final sparse y values into.
   * @param yDense Vector to store final dense y values into.
   * @param Z Matrix to store final dual coordinates into.
   */
  ens_deprecated double Optimize(arma::mat& X,
                                 arma::vec& ySparse,
                                 arma::vec& yDense,
                                 arma::mat& Z);

  /**
   * Optimize the given SDP with the given initial coordinates.  To get a set of
   * initial coordinates from an SDP class, consider calling
   * SDP::GetInitialPoint().  The primal objective is returned, and the final
   * coordinates are stored in the given coordinates matrix.
   *
   * @tparam SDPType Type of SDP to optimize.
   * @tparam MatType Type of matrix to optimize with.
   * @tparam CallbackTypes Types of callback functions.
   * @param sdp The SDP to optimize.
   * @param coordinates The primal SDP coordinates to optimize.
   * @param callbacks Callback functions.
   */
  template<typename SDPType, typename MatType, typename... CallbackTypes>
  typename MatType::elem_type Optimize(const SDPType& sdp,
                                       MatType& coordinates,
                                       CallbackTypes&&... callbacks);

  /**
   * Optimize the given SDP with the given initial primal and dual coordinates.
   * To get a set of primal and dual initial coordinates from an SDP class,
   * consider calling SDP::GetInitialPoints().  The primal objective is
   * returned, and the final primal and dual variables are stored in the given
   * matrices.  Both ySparse and yDense should be vector shaped (one column).
   *
   * @tparam SDPType Type of SDP to optimize.
   * @tparam MatType Type of matrix to optimize with.
   * @tparam CallbackTypes Types of callback functions.
   * @param sdp The SDP to optimize.
   * @param coordinates The initial primal SDP coordinates to optimize.
   * @param ySparse The initial ySparse to optimize.
   * @param yDense The initial yDense to optimize.
   * @param dualCoordinates The initial dual SDP coordinates to optimize.
   * @param callbacks Callback functions.
   */
  template<typename SDPType, typename MatType, typename... CallbackTypes>
  typename MatType::elem_type Optimize(const SDPType& sdp,
                                       MatType& coordinates,
                                       MatType& ySparse,
                                       MatType& yDense,
                                       MatType& dualCoordinates,
                                       CallbackTypes&&... callbacks);

  //! Get the maximum number of iterations to run before converging.
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations to run before converging.
  size_t& MaxIterations() { return maxIterations; }

  //! Get tau.
  double Tau() const { return tau; }
  //! Modify tau. Typical values are 0.99.
  double& Tau() { return tau; }

  //! Get the XZ tolerance.
  double NormXzTol() const { return normXzTol; }
  //! Modify the XZ tolerance.
  double& NormXzTol() { return normXzTol; }

  //! Get the primal infeasibility tolerance.
  double PrimalInfeasTol() const { return primalInfeasTol; }
  //! Modify the primal infeasibility tolerance.
  double& PrimalInfeasTol() { return primalInfeasTol; }

  //! Get the dual infeasibility tolerance.
  double DualInfeasTol() const { return dualInfeasTol; }
  //! Modify the dual infeasibility tolerance.
  double& DualInfeasTol() { return dualInfeasTol; }

 private:
  /**
   * These are deprecated and will be removed in ensmallen 2.10.0.
   */
  DeprecatedSDPType deprecatedSDP;
  arma::mat initialX;
  arma::vec initialYSparse;
  arma::vec initialYDense;
  arma::mat initialZ;

  //! Maximum number of iterations to run. Set to 0 for no limit.
  size_t maxIterations;

  //! The step size modulating factor. Needs to be a scalar in (0, 1).
  double tau;

  //! The tolerance on the norm of XZ required before terminating.
  double normXzTol;

  //! The tolerance required on the primal constraints required before
  //! terminating.
  double primalInfeasTol;

  //! The tolerance required on the dual constraint required before terminating.
  double dualInfeasTol;
};

} // namespace ens

// Include implementation.
#include "primal_dual_impl.hpp"

#endif
