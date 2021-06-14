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

class BayesianBootstrap
{
 public:
  BayesianBootstrap()
  {
    /* Nothing to do. */
  }

  template<typename MatType>
  MatType Generate(size_t numObjectives,
                   size_t numPoints,
                   double epsilon)
  {
      typedef typename MatType::elem_type ElemType;
      typedef typename arma::Col<ElemType> VecType;

      MatType weights(numObjectives, numPoints);
      for (size_t pointIdx = 0; pointIdx < numPoints; ++pointIdx)
      {
        VecType referenceDirection(numObjectives, arma::fill::randu);
        referenceDirection(numObjectives - 1) = 1;
        referenceDirection = arma::sort(referenceDirection);
        std::adjacent_difference(referenceDirection.begin(), referenceDirection.end(),
            referenceDirection.end());
        weights.col(pointIdx) = std::move(referenceDirection) + epsilon;
      }

      return weights;
  }
};

} // namespace ens

#endif