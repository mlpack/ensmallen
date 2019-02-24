/**
 * @file test_function.hpp
 * @author Adeel Ahmad
 *
 * Simple test function for Particle Swarm Optimization:
 *
 * This function returns a squared sum of each element
 * in the given vector.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PSO_TEST_FUNCTION_HPP
#define ENSMALLEN_PSO_TEST_FUNCTION_HPP

namespace ens {

/**
 * Simple test function for Particle Swarm Optimization:
 *
 * This function returns a squared sum of each element
 * in the given vector.
 */
class PSOTestFunction
{
 public:
  PSOTestFunction()
  { /* Nothing to do. */ }

  /**
   * Evaluation of the function.
   *
   * @param position Current position of the particles.
   */
  double Evaluate(const arma::mat& position)
  {
    return arma::accu(arma::square(position));
  }
};

}  // namespace ens

#endif // ENSMALLEN_PSO_TEST_FUNCTION_HPP
