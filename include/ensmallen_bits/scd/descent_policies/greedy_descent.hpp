/**
 * @file greedy_descent.hpp
 * @author Shikhar Bhardwaj
 *
 * Greedy descent policy for Stochastic Coordinate Descent (SCD).
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SCD_DESCENT_POLICIES_GREEDY_HPP
#define ENSMALLEN_SCD_DESCENT_POLICIES_GREEDY_HPP

namespace ens {

/**
 * Greedy descent policy for Stochastic Co-ordinate Descent(SCD). This
 * descent scheme picks a the co-ordinate for the descent with the maximum
 * guaranteed descent, according to the Gauss-Southwell rule. This is a
 * deterministic approach and is generally more expensive to calculate.
 *
 * For more information, refer to the following.
 * @code
 * @misc{Nutini2015,
 *   author = {Julie Nutini and Mark Schmidt and Issam H.
 *             Laradji and Michael Friedlander and Hoyt Koepke},
 *   title  = {Coordinate Descent Converges Faster with the Gauss-Southwell Rule
 *             Than Random Selection},
 *   year   = {2015},
 *   eprint = {arXiv:1506.00552}
 * }
 * @endcode
 */
class GreedyDescent
{
 public:
  /**
   * The DescentFeature method is used to get the descent coordinate for the
   * current iteration.
   *
   * @tparam ResolvableFunctionType The type of the function to be optimized.
   * @param iteration The iteration number for which the feature is to be
   *    obtained.
   * @param iterate The current value of the decision variable.
   * @param function The function to be optimized.
   * @return The index of the coordinate to be descended.
   */
  template<typename ResolvableFunctionType, typename MatType, typename GradType>
  static size_t DescentFeature(const size_t /* iteration */,
                               const MatType& iterate,
                               const ResolvableFunctionType& function)
  {
    typedef typename MatType::elem_type ElemType;

    size_t bestFeature = 0;
    ElemType bestDescent = 0;
    for (size_t i = 0; i < function.NumFeatures(); ++i)
    {
      GradType fGrad;

      function.PartialGradient(iterate, i, fGrad);

      ElemType descent = arma::accu(fGrad);
      if (descent > bestDescent)
      {
        bestFeature = i;
        bestDescent = descent;
      }
    }

    return bestFeature;
  }
};

} // namespace ens

#endif
