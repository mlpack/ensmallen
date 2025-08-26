/**
 * @file sparse_test_function_impl.hpp
 * @author Shikhar Bhardwaj
 *
 * Sparse test function for Parallel SGD.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_SPARSE_TEST_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_SPARSE_TEST_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "sparse_test_function.hpp"

namespace ens {
namespace test {

inline SparseTestFunction::SparseTestFunction()
{
  intercepts = arma::vec("20 12 15 100");
  bi = arma::vec("-4 -2 -3 -8");
}

//! Evaluate a function.
template<typename MatType>
inline typename MatType::elem_type SparseTestFunction::Evaluate(
    const MatType& coordinates,
    const size_t i,
    const size_t batchSize) const
{
  typename MatType::elem_type result = 0.0;
  for (size_t j = i; j < i + batchSize; ++j)
  {
    result += coordinates[j] * coordinates[j] + bi[j] * coordinates[j] +
        intercepts[j];
  }

  return result;
}

//! Evaluate all the functions.
template<typename MatType>
inline typename MatType::elem_type SparseTestFunction::Evaluate(
    const MatType& coordinates) const
{
  typename MatType::elem_type objective = 0.0;
  for (size_t i = 0; i < NumFunctions(); ++i)
  {
    objective += coordinates[i] * coordinates[i] + bi[i] * coordinates[i] +
      intercepts[i];
  }

  return objective;
}

//! Evaluate the gradient of a function.
template<typename MatType, typename GradType>
inline void SparseTestFunction::Gradient(const MatType& coordinates,
                                         const size_t i,
                                         GradType& gradient,
                                         const size_t batchSize) const
{
  gradient.zeros(arma::size(coordinates));
  for (size_t j = i; j < i + batchSize; ++j)
    gradient[j] = 2 * coordinates[j] + bi[j];
}

//! Evaluate the gradient of a feature function.
template<typename MatType, typename GradType>
inline void SparseTestFunction::PartialGradient(const MatType& coordinates,
                                                const size_t j,
                                                GradType& gradient) const
{
  gradient.zeros(arma::size(coordinates));
  gradient[j] = 2 * coordinates[j] + bi[j];
}

} // namespace test
} // namespace ens

#endif
