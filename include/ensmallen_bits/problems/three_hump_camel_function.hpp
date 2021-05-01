/**
 * @file three_hump_camel_function.hpp
 * @author Suryoday Basak
 *
 * Definition of the Three-hump camel function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_THREE_HUMP_CAMEL_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_THREE_HUMP_CAMEL_FUNCTION_HPP

namespace ens {
namespace test {

/**
 * The Three-hump camel function, defined by
 *
 * \f[
 * f(x_1,x_2) = 2 * x_1^2 - 1.05 * x_1^4 + (x_1^6)/6 + x_1 * x_2 + x_2^2
 * \f]
 *
 * This should optimize to f(x) = 0, at x = [0, 0].
 *
 * For more information, please refer to:
 *
 * @code
 * @article{Ali2005,
 *   doi       = {10.1007/s10898-004-9972-2},
 *   year      = {2005},
 *   month     = apr,
 *   publisher = {Springer Nature},
 *   volume    = {31},
 *   number    = {4},
 *   pages     = {635--672},
 *   author    = {M. Montaz Ali and Charoenchai Khompatraporn and
 *                Zelda B. Zabinsky},
 *   title     = {A Numerical Evaluation of Several Stochastic Algorithms
 *                on Selected Continuous Global Optimization Test Problems},
 *   journal   = {Journal of Global Optimization}
 * }
 * @endcode
 */
class ThreeHumpCamelFunction
{
 public:
  //! Initialize the ThreeHumpCamelFunction.
  ThreeHumpCamelFunction();

  /**
   * Shuffle the order of function visitation. This may be called by the
   * optimizer.
   */
  void Shuffle();

  //! Return 1 (the number of functions).
  size_t NumFunctions() const { return 1; }

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

  // Note: GetInitialPoint(), GetFinalPoint(), and GetFinalObjective() are not
  // required for using ensmallen to optimize this function!  They are
  // specifically used as a convenience just for ensmallen's testing
  // infrastructure.

  //! Get the starting point.
  template<typename MatType = arma::mat>
  MatType GetInitialPoint() const { return MatType("1; 1"); }

  //! Get the final point.
  template<typename MatType = arma::mat>
  MatType GetFinalPoint() const { return MatType("0.0; 0.0"); }

  //! Get the final objective.
  double GetFinalObjective() const { return 0.0; }
};

} // namespace test
} // namespace ens

// Include implementation.
#include "three_hump_camel_function_impl.hpp"

#endif // ENSMALLEN_PROBLEMS_THREE_HUMP_CAMEL_FUNCTION_HPP
