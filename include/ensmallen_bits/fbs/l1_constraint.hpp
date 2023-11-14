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
 */
class L1Constraint
{
 public:
  /**
   */
  L1Constraint(const double lambda);

  template<typename MatType>
  typename MatType::elem_type Evaluate(const MatType& coordinates) const;

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
