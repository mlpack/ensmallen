/**
 * @file constant_step.hpp
 * @author Nanubala Gnana Sai
 *
 * The Tchebycheff Weight Decomposition policy.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_MOEAD_TCHEBYCHEFF_HPP
#define ENSMALLEN_MOEAD_TCHEBYCHEFF_HPP

namespace ens {

class Tchebycheff
{
 public:
  Tchebycheff()
  {
    /* Nothing to do. */
  }

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