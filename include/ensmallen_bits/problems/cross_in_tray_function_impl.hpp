/**
 * @file cross_in_tray_function_impl.hpp
 * @author Suryoday Basak
 *
 * Implementation of the Cross-in-Tray function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_CROSS_IN_TRAY_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_CROSS_IN_TRAY_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "cross_in_tray_function.hpp"
using namespace std;

namespace ens {
namespace test {

inline CrossInTrayFunction::CrossInTrayFunction() { /* Nothing to do here */ }

inline void CrossInTrayFunction::Shuffle() { /* Nothing to do here */ }

inline double CrossInTrayFunction::Evaluate(const arma::mat& coordinates,
                                            const size_t /* begin */,
                                            const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  const double objective = -0.0001 * pow(abs(sin(x1) * sin(x2) *
      exp(abs(100 - (sqrt(pow(x1, 2) + pow(x2, 2)) /
      arma::datum::pi))) + 1), 0.1);
  return objective;
}

inline double CrossInTrayFunction::Evaluate(const arma::mat& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

} // namespace test
} // namespace ens

#endif
