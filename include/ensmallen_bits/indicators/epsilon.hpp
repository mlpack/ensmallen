/**
 * @file epsilon.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Implementation of the epsilon indicator.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_INDICATORS_EPSILON_HPP
#define ENSMALLEN_INDICATORS_EPSILON_HPP

namespace ens {

class Epsilon
{
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