/**
 * @file schaffer_function_n2_impl.hpp
 * @author Suryoday Basak
 *
 * Implementation of Schaffer function N.2.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_SCHAFFER_FUNCTION_N2_IMPL_HPP
#define ENSMALLEN_PROBLEMS_SCHAFFER_FUNCTION_N2_IMPL_HPP

// In case it hasn't been included yet.
#include "schaffer_function_n2.hpp"
using namespace std;

namespace ens {
namespace test {

inline SchafferFunctionN2::SchafferFunctionN2() { /* Nothing to do here */ }

inline void SchafferFunctionN2::Shuffle() { /* Nothing to do here */ }

inline double SchafferFunctionN2::Evaluate(const arma::mat& coordinates,
                                           const size_t /* begin */,
                                           const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  const double objective = 0.5 + (pow(sin(pow(x1, 2) - pow(x2, 2)), 2) - 0.5) /
      pow(1 + 0.001 * (pow(x1, 2) + pow(x2, 2)), 2);

  return objective;
}

inline double SchafferFunctionN2::Evaluate(const arma::mat& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

inline void SchafferFunctionN2::Gradient(const arma::mat& coordinates,
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
  const double sum1 = x1Sq - x2Sq;
  const double sinSum1 = sin(sum1);
  const double sum2 = 0.001 * (x1Sq + x2Sq) + 1;
  const double trigExpression = 4 * sinSum1 * cos(sum1);
  const double numerator1 = - 0.004 * (pow(sinSum1, 2) - 0.5);
  const double expr1 = numerator1 / pow(sum2, 3);
  const double expr2 = trigExpression / pow(sum2, 2);

  gradient.set_size(2, 1);
  gradient(0) = x1 * (expr1 + expr2);
  gradient(1) = x2 * (expr1 - expr2);
}

inline void SchafferFunctionN2::Gradient(const arma::mat& coordinates,
                                         arma::mat& gradient)
{
  Gradient(coordinates, 0, gradient, 1);
}

} // namespace test
} // namespace ens

#endif
