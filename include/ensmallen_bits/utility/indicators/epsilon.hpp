/**
 * @file epsilon.hpp
 * @author Rahul Ganesh Prabhu
 * @author Nanubala Gnana Sai
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
 * The epsilon indicator is one of the binary quality indicators that was proposed by
 * Zitzler et. al.. The indicator originally calculates a weak dominance relation 
 * between two approximation sets. It returns "epsilon" which is the factor by which 
 * the given approximation set is worse than the reference front with respect to 
 * all the objectives.
 * 
 * \f[ I_{\epsilon}(A,B) = \max_{z^2 \in B} \
 *                \min_{z^1 \in A} \
 *                \max_{1 \leq i \leq n} \ \frac{z^1_i}{z^2_i}\
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
   public:
    /**
     * Default constructor does nothing, but is required to satisfy the Indicator
     * policy.
     */
    Epsilon() { }

    /**
     * Find the epsilon value of the front with respect to the given reference
     * front.
     *
     * @tparam CubeType The cube data type of front.
     * @param front The given approximation front.
     * @param referenceFront The given reference front.
     * @return The epsilon value of the front.
     */
    template<typename CubeType>
    static typename CubeType::elem_type Evaluate(const CubeType& front,
                                                 const CubeType& referenceFront)
    {
      // Convenience typedefs.
      typedef typename CubeType::elem_type ElemType;
      ElemType eps = 0;
      for (size_t i = 0; i < referenceFront.n_slices; i++)
      {
        ElemType epsjMin = std::numeric_limits<ElemType>::max();
        for (size_t j = 0; j < front.n_slices; j++)
        {
          arma::Mat<ElemType> frontRatio = front.slice(j) / referenceFront.slice(i);
          frontRatio.replace(arma::datum::inf, -1.); // Handle zero division case.
          ElemType epsj = frontRatio.max();
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
