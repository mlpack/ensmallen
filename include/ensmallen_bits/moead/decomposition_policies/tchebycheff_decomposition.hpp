/**
 * @file tchebycheff_decomposition.hpp
 * @author Nanubala Gnana Sai
 *
 * The Tchebycheff Weight decomposition policy.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_MOEAD_TCHEBYCHEFF_HPP
#define ENSMALLEN_MOEAD_TCHEBYCHEFF_HPP

namespace ens {

/**
 * The Tchebycheff method works by taking the maximum of element-wise product
 * between reference direction and the line connecting objective vector and
 * ideal point.
 *
 * Under mild conditions, for each Pareto Optimal point there exists a reference
 * direction such that the given point is also the optimal solution
 * to this scalar objective.
 *
 * For more information, see the following:
 * @code
 * article{zhang2007moea,
 *   title={MOEA/D: A multiobjective evolutionary algorithm based on decomposition},
 *   author={Zhang, Qingfu and Li, Hui},
 *   journal={IEEE Transactions on evolutionary computation},
 *   pages={712--731},
 *   year={2007}
 * @endcode
 */
class Tchebycheff
{
 public:
  /**
   * Constructor for Tchebycheff decomposition policy.
   */
  Tchebycheff()
  {
    /* Nothing to do. */
  }

  /**
   * Decompose the weight vectors.
   *
   * @tparam VecType The type of the vector used in the decommposition.
   * @param weight The weight vector corresponding to a subproblem.
   * @param idealPoint The reference point in the objective space.
   * @param candidateFitness The objective vector of the candidate.
   */
  template<typename VecType>
  typename VecType::elem_type Apply(const VecType& weight,
                                    const VecType& idealPoint,
                                    const VecType& candidateFitness)
  {
    return arma::max(weight % arma::abs(candidateFitness - idealPoint));
  }
};

} // namespace ens

#endif
