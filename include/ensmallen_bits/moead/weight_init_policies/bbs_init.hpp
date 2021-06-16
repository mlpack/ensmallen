/**
 * @file bbs_init.hpp
 * @author Nanubala Gnana Sai
 *
 * The Bayesian Bootstrap (BBS) method of Weight Initialization.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_MOEAD_BBS_HPP
#define ENSMALLEN_MOEAD_BBS_HPP

namespace ens {

/**
 * The Bayesian Bootstrap method for initializing weights. Samples are randomly picked from uniform 
 * distribution followed by sorting and finding adjacent difference. This gives you a list of 
 * numbers which is guaranteed to sum up to 1.
 *
 * @code
 * @article{rubin1981bayesian,
 *   title={The bayesian bootstrap},
 *   author={Rubin, Donald B},
 *   journal={The annals of statistics},
 *   pages={130--134},
 *   year={1981},
 * @endcode
 *
 */
class BayesianBootstrap
{
 public:
  /**
   * Constructor for Bayesian Bootstrap policy.
   */
  BayesianBootstrap()
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
      typedef typename MatType::elem_type ElemType;
      typedef typename arma::Col<ElemType> VecType;

      MatType weights(numObjectives, numPoints);
      for (size_t pointIdx = 0; pointIdx < numPoints; ++pointIdx)
      {
        VecType referenceDirection(numObjectives + 1, arma::fill::randu);
        referenceDirection(0) = 0;
        referenceDirection(numObjectives) = 1;
        referenceDirection = arma::sort(referenceDirection);
        referenceDirection = arma::diff(referenceDirection);
        weights.col(pointIdx) = std::move(referenceDirection) + epsilon;
      }

      return weights;
  }
};

} // namespace ens

#endif
