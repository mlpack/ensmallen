/**
 * @file himmelblau_function_impl.hpp
 * @author Suryoday Basak
 *
 * Implementation of the Beale function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_HIMMELBLAU_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_HIMMELBLAU_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "himmelblau_function.hpp"
using namespace std;

namespace ens {
namespace test {

inline HimmelblauFunction::HimmelblauFunction() { /* Nothing to do here */ }

inline void HimmelblauFunction::Shuffle() { /* Nothing to do here */ }

inline double HimmelblauFunction::Evaluate(const arma::mat& coordinates,
                                           const size_t /* begin */,
                                           const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  const double objective = pow(x1 * x1 + x2  - 11 , 2) +
                           pow(x1 + x2 * x2 - 7, 2);
  return objective;
}

inline double HimmelblauFunction::Evaluate(const arma::mat& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

inline void HimmelblauFunction::Gradient(const arma::mat& coordinates,
                                         const size_t /* begin */,
                                         arma::mat& gradient,
                                         const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  // Aliases for different terms in the expression of the gradient
  const double x1Sq = x1 * x1;
  const double x2Sq = x2 * x2;

  gradient.set_size(2, 1);
  gradient(0) = (4 * x1 * (x1Sq + x2 - 11)) + (2 * (x1 + x2Sq - 7));
  gradient(1) = (2 * (x1Sq + x2 - 11)) + (4 * x2 * (x1 + x2Sq - 7));
}

inline void HimmelblauFunction::Gradient(const arma::mat& coordinates,
                                         arma::mat& gradient)
{
  Gradient(coordinates, 0, gradient, 1);
}

} // namespace test
} // namespace ens

#endif
