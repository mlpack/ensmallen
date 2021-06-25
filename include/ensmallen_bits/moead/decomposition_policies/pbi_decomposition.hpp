/**
 * @file pbi_decomposition.hpp
 * @author Nanubala Gnana Sai
 *
 * The Penalty Based Boundary Intersection (PBI) decomposition policy.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_MOEAD_PBI_HPP
#define ENSMALLEN_MOEAD_PBI_HPP

namespace ens {

/**
 * Penalty Based Boundary Intersection (PBI) method is a weight decomposition method,
 * it tries to find the intersection between bottom-most boundary of the attainable
 * objective set with the reference directions.
 *
 * The goal is to minimize the distance between objective vectors with the ideal point
 * along the reference direction. To handle equality constraints, a penalty parameter
 * theta is used.
 *
 * For more information, see the following:
 * @code
 * article{zhang2007moea,
 *   title={MOEA/D: A multiobjective evolutionary algorithm based on decomposition},
 *   author={Zhang, Qingfu and Li, Hui},
 *   journal={IEEE Transactions on evolutionary computation},
 *   pages={712--731},
 *   year={2007}
 * @endcode
 */
class PenaltyBoundaryIntersection
{
 public:
  /**
   * Constructor for Penalty Based Boundary Intersection decomposition
   * policy.
   *
   * @param theta The penalty value.
   */
  PenaltyBoundaryIntersection(const double theta = 5) :
      theta(theta)
  {
    /* Nothing to do. */
  }

  /**
   * Decompose the weight vectors.
   *
   * @tparam VecType The type of the vector used in the decommposition.
   * @param weight The weight vector corresponding to a subproblem.
   * @param idealPoint The reference point in the objective space.
   * @param candidateFitness The objective vector of the candidate.
   */
  template<typename VecType>
  typename VecType::elem_type Apply(const VecType& weight,
                                    const VecType& idealPoint,
                                    const VecType& candidateFitness)
  {
    typedef typename VecType::elem_type ElemType;
    //! A unit vector in the same direction as the provided weight vector.
    const VecType referenceDirection = weight / arma::norm(weight);
    //! Distance of F(x) from the idealPoint along the reference direction.
    const ElemType d1 = arma::dot(candidateFitness - idealPoint, referenceDirection);
    //! The perpendicular distance of F(x) from reference direction.
    const ElemType d2 = arma::norm(candidateFitness - (idealPoint + d1 * referenceDirection));

    return d1 + static_cast<ElemType>(theta) * d2;
  }

  private:
    double theta;
};

} // namespace ens

#endif
