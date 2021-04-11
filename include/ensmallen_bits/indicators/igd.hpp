/**
 * @file igd.hpp
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

#ifndef ENSMALLEN_INDICATORS_IGD_HPP
#define ENSMALLEN_INDICATORS_IGD_HPP

namespace ens {

  /**
   * The IGD indicator is the average distance from each point in the reference
   * front to the nearest point in it's solution. IGD+ is an improvement upon
   * the IGD indicator, which fixes misleading results given by IGD in certain
   * cases by using a different distance metric:
   *
   * \f[ d^{+}(z,a) = \sqrt{\sum_{i = 1}^{n} \left( \max{a - z}\ \right) \ } \
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
  class IGD
  {
    /**
     * Find the IGD value of the front with respect to the given reference
     * front.
     *
     * @param front The given approximation front.
     * @param referenceFront The given reference front.
     * @return The IGD value of the front.
     */
    double Indicate(arma::cube& front,
      arma::cube& referenceFront)
    {
      double igd = 0;
      for (size_t i = 0; i < referenceFront.n_slices; i++)
      {
        double min = DBL_MAX;
        for (size_t j = 0; j < front.n_slices; j++)
        {
          double dist = 0;
          for (size_t k = 0; k < front.slice(j).n_rows; k++)
          {
            double z = referenceFront(k, 0, i);
            double a = front(k, 0, j);
            dist += std::pow(std::max(z - a, 0.0), 2);
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