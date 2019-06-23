/**
 * @file schaffer_function_n4_impl.hpp
 * @author Suryoday Basak
 *
 * Implementation of Schaffer function N.4.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_SCHAFFER_FUNCTION_N4_IMPL_HPP
#define ENSMALLEN_PROBLEMS_SCHAFFER_FUNCTION_N4_IMPL_HPP

// In case it hasn't been included yet.
#include "schaffer_function_n4.hpp"
using namespace std;

namespace ens {
namespace test {

inline SchafferFunctionN4::SchafferFunctionN4() { /* Nothing to do here */ }

inline void SchafferFunctionN4::Shuffle() { /* Nothing to do here */ }

inline double SchafferFunctionN4::Evaluate(const arma::mat& coordinates,
                                           const size_t /* begin */,
                                           const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  const double objective = 0.5 + (pow(cos(sin(abs(pow(x1, 2) -
      pow(x2, 2)))), 2) - 0.5) / pow(1 + 0.001 * (pow(x1, 2) +
      pow(x2, 2)), 2);

  return objective;
}

inline double SchafferFunctionN4::Evaluate(const arma::mat& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

} // namespace test
} // namespace ens

#endif
