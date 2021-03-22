/**
 * @file cross_in_tray_function.hpp
 * @author Suryoday Basak
 *
 * Definition of the Cross-in-Tray function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_CROSS_IN_TRAY_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_CROSS_IN_TRAY_FUNCTION_HPP

namespace ens {
namespace test {

/**
 * The Cross-in-Tray function, defined by
 *
 * \f[
 * f(x1, x2) = - 0.0001 * (|sin(x1) * sin(x2) *
 *               exp(|100 - (sqrt(x1^2 + x2^2) / pi)|)| + 1)^0.1
 * \f]
 *
 * This should optimize to f(x1, x2) = -2.06261, at
 *                                (x1, x2) = [1.34941, -1.34941], or
 *                                (x1, x2) = [1.34941, 1.34941],  or
 *                                (x1, x2) = [-1.34941, 1.34941], or
 *                                (x1, x2) = [-1.34941, -1.34941]
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
class CrossInTrayFunction
{
 public:
  //! Initialize the CrossInTrayFunction.
  CrossInTrayFunction();

  /**
   * Shuffle the order of function visitation. This may be called by the
   * optimizer.
   */
  void Shuffle();

  //! Return 1 (the number of functions).
  size_t NumFunctions() const { return 1; }

  /*
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

  /*
   * Evaluate a function with the given coordinates.
   *
   * @param coordinates The function coordinates.
   */
  template<typename MatType>
  typename MatType::elem_type Evaluate(const MatType& coordinates) const;

  // Note: GetInitialPoint() is not required for using ensmallen to optimize
  // this function!  It is specifically used as a convenience just for
  // ensmallen's testing infrastructure.

  //! Get the starting point.
  template<typename MatType = arma::mat>
  MatType GetInitialPoint() const { return MatType("0; 0"); }
};

} // namespace test
} // namespace ens

// Include implementation.
#include "cross_in_tray_function_impl.hpp"

#endif // ENSMALLEN_PROBLEMS_CROSS_IN_TRAY_FUNCTION_HPP
