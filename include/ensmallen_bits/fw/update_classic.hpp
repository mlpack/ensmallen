/**
 * @file update_classic.hpp
 * @author Chenzhe Diao
 *
 * Classic update method for FrankWolfe algorithm. Used as UpdateRuleType.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_FW_UPDATE_CLASSIC_HPP
#define ENSMALLEN_FW_UPDATE_CLASSIC_HPP

namespace ens {

/**
 * Use classic rule in the update step for FrankWolfe algorithm. That is,
 * take \f$ \gamma = \frac{2}{k+2} \f$, where \f$ k \f$ is the iteration
 * number. The update rule would be:
 * \f[
 * x_{k+1} = (1-\gamma) x_k + \gamma s
 * \f]
 *
 */
class UpdateClassic
{
 public:
  /**
   * Construct the classic update rule for FrankWolfe algorithm.
   */
  UpdateClassic() { /* Do nothing. */ }

  /**
   * Classic update rule for FrankWolfe.
   *
   * \f$ x_{k+1} = (1-\gamma)x_k + \gamma s \f$, where \f$ \gamma = 2/(k+2) \f$
   *
   * @param function Function to be optimized, not used in this update rule.
   * @param oldCoords Previous solution coords.
   * @param s Current linear_constr_solution result.
   * @param newCoords Output new solution coords.
   * @param numIter Current iteration number.
   */
  template<typename FunctionType, typename MatType, typename GradType>
  void Update(FunctionType& /* function */,
              const MatType& oldCoords,
              const MatType& s,
              MatType& newCoords,
              const size_t numIter)
  {
    typename MatType::elem_type gamma = 2.0 / (numIter + 2.0);
    newCoords = (1.0 - gamma) * oldCoords + gamma * s;
  }
};

} // namespace ens

#endif
