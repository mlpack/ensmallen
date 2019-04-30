/**
 * @file gradient_descent_test_function_impl.hpp
 * @author Sumedh Ghaisas
 *
 * Implementation of very simple test function for gradient descent.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_GRADIENT_DESCENT_TEST_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_GRADIENT_DESCENT_TEST_FUNCTION_IMPL_HPP

#include "gradient_descent_test_function.hpp"

namespace ens {
namespace test {

template<typename MatType>
inline typename MatType::elem_type GDTestFunction::Evaluate(
    const MatType& coordinates) const
{
  MatType temp = arma::trans(coordinates) * coordinates;
  return temp(0, 0);
}

template<typename MatType, typename GradType>
inline void GDTestFunction::Gradient(const MatType& coordinates,
                                     GradType& gradient) const
{
  gradient = 2 * coordinates;
}

} // namespace test
} // namespace ens

#endif
