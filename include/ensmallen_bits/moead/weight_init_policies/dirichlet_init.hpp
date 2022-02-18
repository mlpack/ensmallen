/**
 * @file dirichlet_init.hpp
 * @author Nanubala Gnana Sai
 *
 * The Dirichlet method of Weight Initialization.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_MOEAD_DIRICHLET_HPP
#define ENSMALLEN_MOEAD_DIRICHLET_HPP

namespace ens {

/**
 * The Dirichlet method for initializing weights. Sampling a 
 * Dirichlet distribution with parameters set to one returns 
 * point lying on unit simplex with uniform distribution.
 */
class Dirichlet
{
 public:
  /**
   * Constructor for Dirichlet policy.
   */
  Dirichlet()
  {
    /* Nothing to do. */
  }

  /**
   * Generate the reference direction matrix.
   *
   * @tparam MatType The type of the matrix used for constructing weights.
   * @param numObjectives The dimensionality of objective space.
   * @param numPoints The number of reference directions requested.
   * @param epsilon Handle numerical stability after weight initialization.
   */
  template<typename MatType>
  MatType Generate(const size_t numObjectives,
                   const size_t numPoints,
                   const double epsilon)
  {
    MatType weights = arma::randg<MatType>(numObjectives, numPoints,
        arma::distr_param(1.0, 1.0)) + epsilon;
    // Normalize each column.
    return arma::normalise(weights, 1, 0);
  }
};

} // namespace ens

#endif
