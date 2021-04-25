/**
 * @file igd_plus.hpp
 * @author Rahul Ganesh Prabhu
 * @author Nanubala Gnana Sai
 *
 * Inverse Generational Distance Plus (IGD+) indicator.
 * The average distance from each reference point to its nearest solution.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_INDICATORS_IGD_PLUS_HPP
#define ENSMALLEN_INDICATORS_IGD_PLUS_HPP

namespace ens {

/**
 * The IGD indicator returns the average distance from each point in the reference
 * front to the nearest point to it's solution. IGD+ is an improvement upon
 * the IGD indicator, which fixes misleading results given by IGD in certain
 * cases via a different distance metric:
 *
 * \f[ d^{+}(z,a) = \sqrt{\sum_{i = 1}^{n}( \max\{a_i - z_i, 0\})^2 \ } \
 *    \f]
 *
 * For more information see:
 *
 * @code
 * @article{10.1007/978-3-319-15892-1_8,
 *    author   = {Ishibuchi, Hisao and Masuda, Hiroyuki and Tanigaki, Yuki
 *                and Nojima, Yusuke},
 *    title    = {Modified Distance Calculation in Generational Distance
 *                and Inverted Generational Distance},
 *    book     = {Evolutionary Multi-Criterion Optimization}
 *    year     = {2015}
 * }
 * @endcode
 */
  class IGDPlus
  {
   public:
    /**
     * Default constructor does nothing, but is required to satisfy the Indicator
     * policy.
     */
    IGDPlus() { }

    /**
     * Find the IGD+ value of the front with respect to the given reference
     * front.
     *
     * @tparam CubeType The cube data type of front.
     * @param front The given approximation front.
     * @param referenceFront The given reference front.
     * @return The IGD value of the front.
     */
    template<typename CubeType>
    static typename CubeType::elem_type Evaluate(const CubeType& front,
                                                 const CubeType& referenceFront)
    {
      // Convenience typedefs.
      typedef typename CubeType::elem_type ElemType;
      ElemType igd = 0;
      for (size_t i = 0; i < referenceFront.n_slices; i++)
      {
        ElemType min = std::numeric_limits<ElemType>::max();
        for (size_t j = 0; j < front.n_slices; j++)
        {
          ElemType dist = 0;
          for (size_t k = 0; k < front.slice(j).n_rows; k++)
          {
            ElemType z = referenceFront(k, 0, i);
            ElemType a = front(k, 0, j);
            // Assuming minimization of all objectives.
            dist += std::pow(std::max<ElemType>(a - z, 0), 2);
          }
          dist = std::sqrt(dist);
          if (dist < min)
            min = dist;
        }
        igd += min;
      }
      igd /= referenceFront.n_slices;

      return igd;
    }
  };

} // namespace ens

#endif