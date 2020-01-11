/**
 * @file schaffer_function_n4.hpp
 * @author Suryoday Basak
 *
 * Definition of Schaffer function N.4.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_SCHAFFER_FUNCTION_N4_HPP
#define ENSMALLEN_PROBLEMS_SCHAFFER_FUNCTION_N4_HPP

namespace ens {
namespace test {

/**
 * The Schaffer function N.4, defined by
 *
 * \f[
 * f(x1, x2) = 0.5 + (cos^2(sin(|x1^2 - x2^2|)) - 0.5) /
 *     (1 + 0.001 * (x1^2 + x2^2))^2
 * \f]
 *
 * This should optimize to f(x1, x2) = 0.292579, at (x1, x2) = [0, 1.25313], or
 *                                                  (x1, x2) = [0, -1.25313].
 *
 * For more information, please refer to:
 *
 * @code
 * @misc{LevyFunction,
 *   URL = {http://benchmarkfcns.xyz/benchmarkfcns/schaffern4fcn.html}
 * }
 * @endcode
 */
class SchafferFunctionN4
{
 public:
  //! Initialize the SchafferFunctionN4.
  SchafferFunctionN4();

  /**
   * Shuffle the order of function visitation. This may be called by the
   * optimizer.
   */
  void Shuffle();

  //! Return 1 (the number of functions).
  size_t NumFunctions() const { return 1; }

  //! Get the starting point.
  template<typename MatType = arma::mat>
  MatType GetInitialPoint() const { return MatType("-5; 5"); }

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
};

} // namespace test
} // namespace ens

// Include implementation.
#include "schaffer_function_n4_impl.hpp"

#endif // ENSMALLEN_PROBLEMS_SCHAFFER_FUNCTION_N4_HPP
