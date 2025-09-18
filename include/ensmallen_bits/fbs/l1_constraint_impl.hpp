/**
 * @file l1_constraint_impl.hpp
 * @author Ryan Curtin
 *
 * An implementation of the proximal operator for the L1 constraint.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_FBS_L1_CONSTRAINT_IMPL_HPP
#define ENSMALLEN_FBS_L1_CONSTRAINT_IMPL_HPP

// In case it hasn't been included yet.
#include "l1_constraint.hpp"

namespace ens {

inline L1Constraint::L1Constraint(const double lambda) : lambda(lambda)
{
  // Nothing to do.
}

template<typename MatType>
typename MatType::elem_type L1Constraint::Evaluate(const MatType& coordinates)
    const
{
  typedef typename MatType::elem_type eT;

  // Allow some amount of tolerance for floating-point errors.
  const eT l1Norm = norm(vectorise(coordinates), 1);
  if (l1Norm <= lambda)
    return eT(0);
  else if (std::numeric_limits<eT>::has_infinity)
    return std::numeric_limits<eT>::infinity();
  else
    return std::numeric_limits<eT>::max();
}

template<typename MatType>
void L1Constraint::ProximalStep(MatType& coordinates,
                                const double /* stepSize */)
    const
{
  // First determine whether projection is necessary.
  if (norm(vectorise(coordinates), 1) <= lambda)
  {
    return;
  }

  // An empty vector can't be projected.
  if (coordinates.n_elem == 0)
  {
    return;
  }

  // We use the algorithm denoted in Figure 2 of the following paper:
  //
  // ```
  // @inproceedings{duchi2008efficient,
  //   title={Efficient projections onto the L1-ball for learning in high
  //       dimensions},
  //   author={Duchi, John and Shalev-Shwartz, Shai and Singer, Yoram and
  //       Chandra, Tushar},
  //   booktitle={Proceedings of the 25th international conference on
  //       Machine learning},
  //   pages={272--279},
  //   year={2008}
  // }
  // ```
  //
  // This is an iterative algorithm that has a quicksort feel, where we try to
  // determine the "pivot" element that tells us how much we need to shrink.  In
  // the original paper, they maintain lists indicating whether a point is above
  // or below the pivot, but it is more expedient (and efficient) to simply copy
  // the coordinates array and partially sort it in-place.

  typedef typename MatType::elem_type eT;
  arma::Col<eT> work = ExtractNonzeros(coordinates);
  size_t firstUpperElement = 0;
  size_t lastUpperElement = work.n_elem;
  eT rho = eT(0); // This is the quantity we aim to find to perform the projection.
  eT s = eT(0);

  while (lastUpperElement > firstUpperElement)
  {
    const size_t k = arma::randi<size_t>(
        arma::distr_param((int) firstUpperElement, (int) lastUpperElement - 1));
    const eT v = work[k];

    // Now perform a half-quicksort such that all elements greater than v are in
    // the first part of the array.
    size_t left = firstUpperElement;
    size_t right = lastUpperElement - 1;
    while (left <= right)
    {
      while ((left < lastUpperElement) && (work[left] >= v))
        ++left;
      while ((right > firstUpperElement) && (work[right] < v))
        --right;

      if (left >= right)
        break;

      // work[left] is less than v, and work[right] is not.  Since we want all
      // elements greater than or equal to v on the left, swap.
      const eT tmp = work[left];
      work[left] = work[right];
      work[right] = tmp;
    }

    // Now, work[0] through work[left - 1] are in the greater set G.
    const eT sDelta = accu(work.subvec(firstUpperElement, left - 1));
    const size_t rhoDelta = (left - firstUpperElement);

    if ((s + sDelta) - ((eT) (rho + rhoDelta)) * v < eT(lambda))
    {
      s += sDelta;
      rho += rhoDelta;
      firstUpperElement = left;
    }
    else
    {
      // v was an element that was less than rho, so, shrink the array and try
      // again with larger elements.  We actually want to shrink the array so
      // that it does not include v, so we need to find the first element that
      // is v (since there may be duplicates).
      size_t firstVIndex = left - 1;
      while ((work[firstVIndex] == v) && (firstVIndex >= firstUpperElement))
        --firstVIndex;
      lastUpperElement = firstVIndex + 1;
    }
  }

  const eT theta = (s - eT(lambda)) / rho;
  coordinates.transform(
      [theta](eT val)
      {
        if (val > 0)
          return std::max(val - theta, eT(0));
        else
          return std::min(val + theta, eT(0));
      });

  // Sanity check: ensure we actually ended up inside the L1 ball.  This might
  // not happen due to floating-point inaccuracies.  If so, try again.
  const eT newNorm = norm(coordinates, 1);
  if (newNorm > eT(lambda) && eT(lambda) > eT(0))
  {
    // Shrink the L1 ball by the amount of the error.
    eT newLambda = (eT(lambda) - 2 * (newNorm - eT(lambda)));
    if (newLambda == eT(lambda))
    {
      // Make sure we at least remove a few ULPs.
      newLambda = eT(lambda) -
          5 * (eT(lambda) - eT(std::nexttoward(lambda, 0.0)));
    }

    L1Constraint newConstraint(newLambda);
    newConstraint.ProximalStep(coordinates, 0.0 /* ignored */);
  }
}

// Helper function: extract only nonzero elements from sparse objects, or
// extract the entire dense object.
template<typename MatType>
inline arma::Col<typename MatType::elem_type> L1Constraint::ExtractNonzeros(
    const MatType& coordinates) const
{
  return arma::Col<typename MatType::elem_type>(vectorise(abs(coordinates)));
}

template<typename eT>
inline arma::Col<eT> L1Constraint::ExtractNonzeros(
    const arma::SpMat<eT>& coordinates) const
{
  arma::Col<eT> result(coordinates.n_nonzero);
  typename arma::SpMat<eT>::const_iterator it = coordinates.begin();
  size_t i = 0;
  while (it != coordinates.end())
  {
    // Extract only nonzero values.  Note we use the absolute value because that
    // is what the algorithm requires (not the original value).
    result[i] = std::abs(*it);
    ++it;
    ++i;
  }

  return result;
}

} // namespace ens

#endif
