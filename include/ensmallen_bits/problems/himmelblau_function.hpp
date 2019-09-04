/**
 * @file himmelblau_function.hpp
 * @author Suryoday Basak
 *
 * Definition of the Himmelblau function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_HIMMELBLAU_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_HIMMELBLAU_FUNCTION_HPP

namespace ens {
namespace test {

/**
 * The Himmelblau function, defined by
 *
 * \f[
 * f(x_1,x_2) = (x_1^2 + y - 11)^2 + (x_1 + x_2^2 - 7)^2
 * \f]
 *
 * This should optimize to f(x) = 0, at x = [3.0,  2.0], or
 *          x = [-2.805118, 3.131312], or
 *          x = [-3.779310, -3.283186], or
 *          x = [3.584428, -1.848126].
 *
 * For more information, please refer to:
 *
 * @code
 * @book{davidmautnerhimmelblau1972,
 *   Author      = {David Mautner Himmelblau},
 *   title       = {Applied Nonlinear Programming},
 *   description = {Applied Nonlinear Programming (Book, 1972)},
 *   publisher   = {McGraw-Hill},
 *   year        = {1972},
 *   month       = {jun},
 *   isbn        = {0070289212},
 * }
 * @endcode
 */
class HimmelblauFunction
{
 public:
  //! Initialize the HimmelblauFunction.
  HimmelblauFunction();

  /**
   * Shuffle the order of function visitation. This may be called by the
   * optimizer.
   */
  void Shuffle();

  //! Return 1 (the number of functions).
  size_t NumFunctions() const { return 1; }

  //! Get the starting point.
  template<typename MatType = arma::mat>
  MatType GetInitialPoint() const { return MatType("5; -5"); }

  /**
   * Evaluate a function for a particular batch-size.
   *
   * @param coordinates The function coordinates.
   * @param begin The first function.
   * @param batchSize Number of points to process.
   */
  template<typename MatType>
  typename MatType::elem_type Evaluate(const MatType& coordinates,
                                       const size_t begin,
                                       const size_t batchSize) const;

  /**
   * Evaluate a function with the given coordinates.
   *
   * @param coordinates The function coordinates.
   */
  template<typename MatType>
  typename MatType::elem_type Evaluate(const MatType& coordinates) const;

  /**
   * Evaluate the gradient of a function for a particular batch-size.
   *
   * @param coordinates The function coordinates.
   * @param begin The first function.
   * @param gradient The function gradient.
   * @param batchSize Number of points to process.
   */
  template<typename MatType, typename GradType>
  void Gradient(const MatType& coordinates,
                const size_t begin,
                GradType& gradient,
                const size_t batchSize) const;

  /**
   * Evaluate the gradient of a function with the given coordinates.
   *
   * @param coordinates The function coordinates.
   * @param gradient The function gradient.
   */
  template<typename MatType, typename GradType>
  void Gradient(const MatType& coordinates, GradType& gradient);
};

} // namespace test
} // namespace ens

// Include implementation.
#include "himmelblau_function_impl.hpp"

#endif // ENSMALLEN_PROBLEMS_HIMMELBLAU_FUNCTION_HPP
