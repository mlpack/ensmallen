/**
 * @file update_full_correction.hpp
 * @author Chenzhe Diao
 *
 * Update method for FrankWolfe algorithm, recalculate the coefficents of
 * of current atoms, while satisfying the norm constraint.
 * Used as UpdateRuleType.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_FW_UPDATE_FULL_CORRECTION_HPP
#define ENSMALLEN_FW_UPDATE_FULL_CORRECTION_HPP

#include "atoms.hpp"

namespace ens {

/**
 * Full correction approach to update the solution.
 *
 * UpdateSpan class reoptimize the solution in the span of all current atoms,
 * which is used in OMP, which only focus on sparsity.
 *
 * UpdateFullCorrection class reoptimize the solution in a similar way, however,
 * the solutions need to satisfy the constraint that the atom norm has to be
 * smaller than or equal to tau. This constraint optimization problem is solved
 * by projected gradient method. See Atoms.ProjectedEnhancement().
 *
 * Currently only works for function in FuncSq class.
 *
 */
class UpdateFullCorrection
{
 public:
  /**
   * Construct UpdateFullCorrection class.
   *
   * @param tau atom norm constraint.
   * @param stepSize step size used in projected gradient method.
   */
  UpdateFullCorrection(const double tau, const double stepSize) :
      tau(tau), stepSize(stepSize)
  { /* Do nothing. */ }

  /**
   * Update rule for FrankWolfe, recalculate the coefficents of of current
   * atoms, while satisfying the norm constraint.
   *
   * FuncSqType is an ignored type to match the requirements of the class.
   *
   * @param function Function to be optimized.
   * @param oldCoords Previous solution coords.
   * @param s Current linear_constr_solution result.
   * @param newCoords New output solution coords.
   * @param numIter Current iteration number.
   */
  template<typename FuncSqType, typename MatType, typename GradType>
  void Update(FuncSq& function,
              const MatType& oldCoords,
              const MatType& s,
              MatType& newCoords,
              const size_t /* numIter */)
  {
    // Line search, with explicit solution here.
    MatType v = tau * s - oldCoords;
    MatType b = function.Vectorb();
    MatType A = function.MatrixA();
    typename MatType::elem_type gamma = arma::dot(b - A * oldCoords, A * v);
    gamma = gamma / std::pow(arma::norm(A * v, "fro"), 2);
    gamma = std::min(gamma, 1.0);
    atoms.CurrentCoeffs() = (1.0 - gamma) * atoms.CurrentCoeffs();
    atoms.AddAtom(arma::mat(s), function, gamma * tau);

    // Projected gradient method for enhancement.
    atoms.ProjectedGradientEnhancement(function, tau, stepSize);
    arma::mat tmp;
    atoms.RecoverVector(tmp);
    newCoords = arma::conv_to<MatType>::from(tmp);
  }

 private:
  //! Atom norm constraint.
  double tau;

  //! Step size in projected gradient method.
  double stepSize;

  //! Atoms information.
  Atoms atoms;
};

} // namespace ens

#endif
