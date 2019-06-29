/**
 * @file three_hump_camel_function_impl.hpp
 * @author Suryoday Basak
 *
 * Implementation of the Three-hump camel function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_THREE_HUMP_CAMEL_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_THREE_HUMP_CAMEL_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "three_hump_camel_function.hpp"
using namespace std;

namespace ens {
namespace test {

inline ThreeHumpCamelFunction::ThreeHumpCamelFunction()
{ /* Nothing to do here */ }

inline void ThreeHumpCamelFunction::Shuffle() { /* Nothing to do here */ }

inline double ThreeHumpCamelFunction::Evaluate(
    const arma::mat& coordinates,
    const size_t /* begin */,
    const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  const double objective = (2 * pow(x1, 2)) - (1.05 * pow(x1, 4)) +
      (pow(x1, 6) / 6) + (x1 * x2) + pow(x2, 2);
  return objective;
}

inline double ThreeHumpCamelFunction::Evaluate(
    const arma::mat& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

inline void ThreeHumpCamelFunction::Gradient(const arma::mat& coordinates,
                                             const size_t /* begin */,
                                             arma::mat& gradient,
                                             const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  gradient.set_size(2, 1);
  gradient(0) = pow(x1, 5) - (4.2 * pow(x1, 3)) + (4 * x1) + x2;
  gradient(1) = x1 + (2 * x2);
}

inline void ThreeHumpCamelFunction::Gradient(const arma::mat& coordinates,
                                             arma::mat& gradient)
{
  Gradient(coordinates, 0, gradient, 1);
}

} // namespace test
} // namespace ens

#endif
