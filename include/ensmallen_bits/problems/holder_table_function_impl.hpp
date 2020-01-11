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

namespace ens {
namespace test {

inline HolderTableFunction::HolderTableFunction() { /* Nothing to do here */ }

inline void HolderTableFunction::Shuffle() { /* Nothing to do here */ }

template<typename MatType>
typename MatType::elem_type HolderTableFunction::Evaluate(
    const MatType& coordinates,
    const size_t /* begin */,
    const size_t /* batchSize */) const
{
  // Convenience typedef.
  typedef typename MatType::elem_type ElemType;

  // For convenience; we assume these temporaries will be optimized out.
  const ElemType x1 = coordinates(0);
  const ElemType x2 = coordinates(1);

  const ElemType objective = -std::abs(std::sin(x1) * std::cos(x2) *
      std::exp(std::abs(1 - (std::sqrt(x1 * x1 + x2 * x2) / arma::datum::pi))));

  return objective;
}

template<typename MatType>
typename MatType::elem_type HolderTableFunction::Evaluate(
    const MatType& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

} // namespace test
} // namespace ens

#endif
