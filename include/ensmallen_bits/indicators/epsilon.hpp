/**
 * @file epsilon.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Epsilon indicator
 * A binary quality indicator that is capable of detecting whether one
 * approximation set is better than another.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_INDICATORS_EPSILON_HPP
#define ENSMALLEN_INDICATORS_EPSILON_HPP

namespace ens {

/**
 * The epsilon indicator is a binary quality indicator that was proposed by
 * Zitzler et. al. in response to the observation that quality indicators could
 * only give information that one approximation set is not worse than another.
 * The binary epsilon indicator, however, is capable of detecting if an
 * approximation is better than another.
 *
 * \f[ I_c(A,B) = \max_{z_2 \in B} \
 *                \min_{z_1 \in A} \
 *                \max_{1 \leq i \leq n} \ \frac{z_1}{z_2}\
 *                \f]
 *
 * For more information, please see:
 *
 * @code
 * @article{1197687,
 *    author   = {E. Zitzler and L. Thiele and M. Laumanns and C. M. Fonseca and
 *                V. G. da Fonseca},
 *    title    = {Performance assessment of multiobjective optimizers: an
 *                analysis and review},
 *    journal  = {IEEE Transactions on Evolutionary Computation},
 *    year     = {2003},
 * }
 * @endcode
 */
class Epsilon
{
  /**
   * Find the epsilon value of the front with respect to the given reference
   * front.
   *
   * @param front The given approximation front.
   * @param referenceFront The given reference front.
   * @return The epsilon value of the front.
   */
  double Indicate(arma::cube& front,
  				        arma::cube& referenceFront)
  {
    double eps = 0;
    for (size_t i = 0; i < referenceFront.n_slices; i++)
    {
      double epsjMin = DBL_MAX;
      for (size_t j = 0; j < front.n_slices; j++)
      {
        double epsj = (front.slice(j) / referenceFront.slice(i)).max();
        if (epsj < epsjMin)
          epsjMin = epsj;
      }
      if (epsjMin > eps)
        eps = epsjMin;
    }

    return eps;
  }
};

} // namespace ens

#endif