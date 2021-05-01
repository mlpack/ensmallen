/**
 * @file schaffer_function_n2.hpp
 * @author Suryoday Basak
 *
 * Definition of Schaffer function N.2.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_SCHAFFER_FUNCTION_N2_HPP
#define ENSMALLEN_PROBLEMS_SCHAFFER_FUNCTION_N2_HPP

namespace ens {
namespace test {

/**
 * The Schaffer function N.2, defined by
 *
 * \f[
 * f(x1, x2) = 0.5 + ((sin^2(x1^2 - x2^2) - 0.5) /
 *             (1 + 0.001 * (x1^2 + x2^2))^2)
 * \f]
 *
 * This should optimize to f(x1, x2) = 0, at (x1, x2) = [0, 0].
 *
 * For more information, please refer to:
 *
 * @code
 * @article{1308.4008,
 *   Author = {Momin Jamil and Xin-She Yang},
 *   Title  = {A Literature Survey of Benchmark Functions For Global
 *             Optimization Problems},
 *   Year   = {2013},
 *   Eprint = {arXiv:1308.4008},
 *   Doi    = {10.1504/IJMMNO.2013.055204},
 * }
 * @endcode
 */
class SchafferFunctionN2
{
 public:
  //! Initialize the SchafferFunctionN2.
  SchafferFunctionN2();

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
  MatType GetInitialPoint() const { return MatType("-100; 100"); }

  //! Get the final point.
  template<typename MatType = arma::mat>
  MatType GetFinalPoint() const { return MatType("0.0; 0.0"); }

  //! Get the final objective.
  double GetFinalObjective() const { return 0.0; }
};

} // namespace test
} // namespace ens

// Include implementation.
#include "schaffer_function_n2_impl.hpp"

#endif // ENSMALLEN_PROBLEMS_SCHAFFER_FUNCTION_N2_HPP
