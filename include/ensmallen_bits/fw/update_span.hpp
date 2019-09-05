/**
 * @file update_span.hpp
 * @author Chenzhe Diao
 *
 * Update method for FrankWolfe algorithm, recalculate the optimal in the span
 * of previous solution space. Used as UpdateRuleType.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_FW_UPDATE_SPAN_HPP
#define ENSMALLEN_FW_UPDATE_SPAN_HPP

#include "func_sq.hpp"
#include "atoms.hpp"

namespace ens {

/**
 * Recalculate the optimal solution in the span of all previous solution space,
 * used as update step for FrankWolfe algorithm.
 *
 * Currently only works for function in FuncSq class.
 */
class UpdateSpan
{
 public:
  /**
   * Construct the span update rule. The function to be optimized is input here.
   *
   * @param function Function to be optimized in FrankWolfe algorithm.
   */
  UpdateSpan(const bool isPrune = false) : isPrune(isPrune)
  { /* Do nothing. */ }

  /**
   * Update rule for FrankWolfe, reoptimize in the span of current
   * solution space.
   *
   * @param function function to be optimized.
   * @param oldCoords previous solution coords.
   * @param s current linearConstrSolution result.
   * @param newCoords output new solution coords.
   * @param numIter current iteration number.
   */
  template<typename FuncSqType, typename MatType, typename GradType>
  void Update(FuncSq& function,
              const MatType& oldCoords,
              const MatType& s,
              MatType& newCoords,
              const size_t /* numIter */)
  {
    // Add new atom into soluton space.
    atoms.AddAtom(arma::mat(s), function);

    // Reoptimize the solution in the current space.
    arma::vec b = function.Vectorb();
    atoms.CurrentCoeffs() = solve(function.MatrixA() * atoms.CurrentAtoms(), b, arma::solve_opts::fast);

    // x has coords of only the current atoms, recover the solution
    // to the original size.
    arma::mat tmp;
    atoms.RecoverVector(tmp);
    newCoords = arma::conv_to<MatType>::from(tmp);

    // Prune the support.
    if (isPrune)
    {
      double oldF = function.Evaluate(oldCoords);
      double F = 0.25 * oldF + 0.75 * function.Evaluate(newCoords);
      atoms.PruneSupport(F, function);
      atoms.RecoverVector(tmp);
      newCoords = arma::conv_to<MatType>::from(tmp);
    }
  }

 private:
  //! Atoms information.
  Atoms atoms;

  //! Flag for support prune step.
  bool isPrune;
}; // class UpdateSpan

} // namespace ens

#endif
