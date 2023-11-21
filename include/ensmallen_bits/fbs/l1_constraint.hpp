/**
 * @file l1_constraint.hpp
 * @author Ryan Curtin
 *
 * An implementation of the proximal operator for the L1 constraint.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_FBS_L1_CONSTRAINT_HPP
#define ENSMALLEN_FBS_L1_CONSTRAINT_HPP

namespace ens {

/**
 * The L1Constraint applies a specific constraint that the L1 norm of the
 * parameters must be less than or equal to the given lambda value.
 *
 * Implementationally, this means that the proximal step is a projection onto
 * the L1 ball of radius lambda.  If the constraint is satisfied, `Evaluate()`
 * will return 0.  Otherwise, it will return infinity.
 *
 * This class is meant to be used with the FBS optimizer, and any other
 * optimizer that uses a proximal operator/step.
 */
class L1Constraint
{
 public:
  /**
   * Construct an L1Constraint with the given maximum L1 norm for the
   * coordinates (lambda).
   */
  L1Constraint(const double lambda);

  /**
   * If the L1 norm of the coordinates is less than or equal to lambda, this
   * returns 0.  Otherwise, it returns infinity.
   */
  template<typename MatType>
  typename MatType::elem_type Evaluate(const MatType& coordinates) const;

  /**
   * Apply a proximal step to the given `coordinates`, assuming that the forward
   * step took a step of size `stepSize`.  This projects `coordinates` back onto
   * the surface of the L1-ball with radius `lambda`, if the L1 norm of
   * `coordinates` is greater than `lambda`.
   *
   * This may apply the proximal step multiple times to account for numerical
   * stability issues during projection.
   */
  template<typename MatType>
  void ProximalStep(MatType& coordinates, const double stepSize) const;

  //! Get the L1 constraint to use when applying the proximal step.
  double Lambda() const { return lambda; }
  //! Modify the L1 constraint to use when applying the proximal step.
  double& Lambda() { return lambda; }

 private:
  //! The L1 constraint value to use.
  double lambda;

  //! Helper function: extract only nonzero elements from sparse objects, or
  //! extract the entire dense object.
  template<typename MatType>
  inline arma::Col<typename MatType::elem_type> ExtractNonzeros(
      const MatType& coordinates) const;

  template<typename eT>
  inline arma::Col<eT> ExtractNonzeros(const arma::SpMat<eT>& coordinates)
      const;
};

} // namespace ens

// Include implementation.
#include "l1_constraint_impl.hpp"

#endif
