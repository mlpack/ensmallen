/**
 * @file constant_step.hpp
 * @author Nanubala Gnana Sai
 *
 * The Weighted Average Decomposition policy.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_MOEAD_WEIGHTED_HPP
#define ENSMALLEN_MOEAD_WEIGHTED_HPP

namespace ens {

class WeightedAverage
{
 public:
  WeightedAverage()
  {
    /* Nothing to do. */
  }

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