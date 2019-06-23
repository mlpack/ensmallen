/**
 * @file holder_table_function_impl.hpp
 * @author Suryoday Basak
 *
 * Implementation of the Holder table function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_HOLDER_TABLE_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_HOLDER_TABLE_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "holder_table_function.hpp"
using namespace std;

namespace ens {
namespace test {

inline HolderTableFunction::HolderTableFunction() { /* Nothing to do here */ }

inline void HolderTableFunction::Shuffle() { /* Nothing to do here */ }

inline double HolderTableFunction::Evaluate(const arma::mat& coordinates,
                                            const size_t /* begin */,
                                            const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  const double objective = - abs(sin(x1) * cos(x2) * exp(abs(1 -
      (sqrt(x1 * x1 + x2 * x2) / arma::datum::pi))));

  return objective;
}

inline double HolderTableFunction::Evaluate(const arma::mat& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

} // namespace test
} // namespace ens

#endif
