/**
 * @file igd.hpp
 * @author Satyam Shukla
 *
 * Inverse Generational Distance Plus (IGD) indicator.
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
 * The inverted generational distance( IGD) is a metric for assessing the quality
 * of approximations to the Pareto front obtained by multi-objective optimization
 * algorithms.The IGD indicator returns the average distance from each point in 
 * the reference front to the nearest point to it's solution. 
 *
 * \f[ d(z,a) = \sqrt{\sum_{i = 1}^{n}(a_i - z_i)^2 \ } \
 *    \f]
 *
 * For more information see:
 *
 * @code
 * @inproceedings{coello2004study,
 * title={A study of the parallelization of a coevolutionary multi-objective evolutionary algorithm},
 * author={Coello Coello, Carlos A and Reyes Sierra, Margarita},
 * booktitle={MICAI 2004: Advances in Artificial Intelligence: Third Mexican International Conference on Artificial Intelligence, Mexico City, Mexico, April 26-30, 2004. Proceedings 3},
 * pages={688--697},
 * year={2004},
 * organization={Springer}
 * }
 * @endcode
 */
  class IGD
  {
   public:
    /**
     * Default constructor does nothing, but is required to satisfy the Indicator
     * policy.
     */
    IGD() { }

    /**
     * Find the IGD value of the front with respect to the given reference
     * front.
     *
     * @tparam CubeType The cube data type of front.
     * @param front The given approximation front.
     * @param referenceFront The given reference front.
     * @param p The power constant in the distance formula. 
     * @return The IGD value of the front.
     */
    template<typename CubeType>
    static typename CubeType::elem_type Evaluate(const CubeType& front,
                                                 const CubeType& referenceFront,
                                                 double p)
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
            //! IGD does not clip negative differences to 0
            dist += std::pow(a - z, 2); 
          }
          dist = std::sqrt(dist);
          if (dist < min)
            min = dist;
        }
        igd += std::pow(min,p);
      }
      igd /= referenceFront.n_slices;
      igd = std::pow(igd, 1.0 / p);
      return igd;
    }
  };

} // namespace ens

#endif