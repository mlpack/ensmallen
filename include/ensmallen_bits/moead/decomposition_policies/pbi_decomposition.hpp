/**
 * @file pbi_decomposition.hpp
 * @author Nanubala Gnana Sai
 *
 * The Penalty Based Boundary Intersection(PBI) Decomposition policy.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_MOEAD_PBI_HPP
#define ENSMALLEN_MOEAD_PBI_HPP

namespace ens {

class PenaltyBoundaryIntersection
{
 public:
  PenaltyBoundaryIntersection(const double theta = 5) :
      theta(theta)
  {
    /* Nothing to do. */
  }

  template<typename VecType>
  typename VecType::elem_type Apply(const VecType& weight,
                                    const VecType& idealPoint,
                                    const VecType& candidateFitness)
  {
    typedef typename VecType::elem_type ElemType;
    //! A unit vector in the same direction as the provided weight vector.
    const VecType referenceDirection = weight / arma::norm(weight, 1);
    //! Distance of F(x) from the idealPoint along the reference direction.
    const ElemType d1 = arma::dot(candidateFitness - idealPoint, referenceDirection);
    //! Length of projection line of F(x) on provided weight vector.
    const ElemType d2 = arma::norm(candidateFitness - (idealPoint + d1 * referenceDirection), 1);

    return d1 + static_cast<ElemType>(theta) * d2;
  }

  private:
    double theta;
};

} // namespace ens

#endif