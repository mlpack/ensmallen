/**
 * @file l1_penalty.hpp
 * @author Ryan Curtin
 *
 * An implementation of the proximal operator for the L1 penalty (also known as
 * the shrinkage operator).
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_FBS_L1_PENALTY_HPP
#define ENSMALLEN_FBS_L1_PENALTY_HPP

namespace ens {

/**
 * The L1Penalty applies a non-differentiable L1-norm penalty to the coordinates
 * during optimization:
 *
 *   `lambda * || coordinates ||_1`
 *
 * This class is meant to be used with the FBS optimizer, and any other
 * optimizer that uses a proximal operator/step.
 */
class L1Penalty
{
 public:
  /**
   * Construct an L1Penalty object with a given penalty `lambda`.
   */
  L1Penalty(const double lambda);

  /**
   * Evaluate the L1 penalty function: `lambda * || coordinates ||_1`.
   */
  template<typename MatType>
  typename MatType::elem_type Evaluate(const MatType& coordinates) const;

  /**
   * After taking a forward step of size `stepSize`, apply a backwards step /
   * proximal operator that applies the L1 penalty to `coordinates`.
   */
  template<typename MatType>
  void ProximalStep(MatType& coordinates, const double stepSize) const;

  //! Get the L1 penalty to use when applying the proximal step.
  double Lambda() const { return lambda; }
  //! Modify the L1 penalty to use when applying the proximal step.
  double& Lambda() { return lambda; }

 private:
  //! The L1 penalty value to use.
  double lambda;
};

} // namespace ens

// Include implementation.
#include "l1_penalty_impl.hpp"

#endif
