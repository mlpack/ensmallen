/**
 * @file weighted_decomposition.hpp
 * @author Nanubala Gnana Sai
 *
 * The Weighted Average decomposition policy.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_MOEAD_WEIGHTED_HPP
#define ENSMALLEN_MOEAD_WEIGHTED_HPP

namespace ens {

/**
 * The Weighted average method of decomposition. The working principle is to
 * minimize the dot product between reference direction and the line connecting
 * objective vector and ideal point.
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
class WeightedAverage
{
 public:
  /**
   * Constructor for Weighted Average decomposition policy.
   */
  WeightedAverage()
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
                                    const VecType& /* idealPoint */,
                                    const VecType& candidateFitness)
  {
    return arma::dot(weight, candidateFitness);
  }
};

} // namespace ens

#endif
