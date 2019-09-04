/**
 * @file lrsdp.hpp
 * @author Ryan Curtin
 *
 * An implementation of Monteiro and Burer's formulation of low-rank
 * semidefinite programs (LR-SDP).
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SDP_LRSDP_HPP
#define ENSMALLEN_SDP_LRSDP_HPP

#include <ensmallen_bits/aug_lagrangian/aug_lagrangian.hpp>

#include "lrsdp_function.hpp"

namespace ens {

/**
 * LRSDP is the implementation of Monteiro and Burer's formulation of low-rank
 * semidefinite programs (LR-SDP).  This solver uses the augmented Lagrangian
 * optimizer to solve low-rank semidefinite programs.
 *
 * LRSDP can optimize semidefinite programs.  For more details, see the
 * documentation on function types included with this distribution or on the
 * ensmallen website.
 */
template <typename SDPType>
class LRSDP
{
 public:
  /**
   * Create an LRSDP to be optimized.  The solution will end up being a matrix
   * of size (rows) x (rank).  To construct each constraint and the objective
   * function, use the function SDP() in order to access the SDPType object
   * associated with this optimizer.
   *
   * @param numSparseConstraints Number of sparse constraints in the problem.
   * @param numDenseConstraints Number of dense constraints in the problem.
   * @param initialPoint Initial point of the optimization.
   * @param maxIterations Maximum number of iterations.
   */
  LRSDP(const size_t numSparseConstraints,
        const size_t numDenseConstraints,
        const arma::Mat<typename SDPType::ElemType>& initialPoint,
        const size_t maxIterations = 1000);

  /**
   * Create an LRSDP object with the given SDP problem to be solved, and the
   * given initial point.  Note that the SDP may be modified later by calling
   * SDP() to access the object.
   *
   * TODO: this is currently not implemented.
   *
   * @param sdp SDP to be solved.
   * @param initialPoint Initial point of the optimization.
   * @param maxIterations Maximum number of iterations.
   *
  LRSDP(const SDPType& sdp,
        const arma::mat& initialPoint,
        const size_t maxIterations = 1000);
   */

  /**
   * Optimize the LRSDP and return the final objective value.  The given
   * coordinates will be modified to contain the final solution.
   *
   * @param coordinates Starting coordinates for the optimization.
   * @param callbacks Callback functions.
   */
  template<typename MatType, typename... CallbackTypes>
  typename MatType::elem_type Optimize(MatType& coordinates,
                                       CallbackTypes&&... callbacks);

  //! Return the SDP that will be solved.
  const SDPType& SDP() const { return function.SDP(); }
  //! Modify the SDP that will be solved.
  SDPType& SDP() { return function.SDP(); }

  //! Return the function to be optimized.
  const LRSDPFunction<SDPType>& Function() const { return function; }
  //! Modify the function to be optimized.
  LRSDPFunction<SDPType>& Function() { return function; }

  //! Return the augmented Lagrangian object.
  const AugLagrangian& AugLag() const { return augLag; }
  //! Modify the augmented Lagrangian object.
  AugLagrangian& AugLag() { return augLag; }

  //! Get the maximum number of iterations.
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations.
  size_t& MaxIterations() { return maxIterations; }

 private:
  //! Augmented lagrangian optimizer.
  AugLagrangian augLag;
  //! Function to optimize, which the AugLagrangian object holds.
  LRSDPFunction<SDPType> function;
  //! The maximum number of iterations for optimization.
  size_t maxIterations;
};

} // namespace ens

// Include implementation
#include "lrsdp_impl.hpp"

#endif
