/**
 * @file goldstein_price_function.hpp
 * @author Suryoday Basak
 *
 * Definition of the Goldstein-Price function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_GOLDSTEIN_PRICE_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_GOLDSTEIN_PRICE_FUNCTION_HPP

namespace ens {
namespace test {

/**
 * The Goldstein-Price function, defined by
 * \f[
 * f(x_1, x_2) = (1 + (x_1 + x_2 + 1)^2 * (19 - 14 * x_1 + 3 * x_1^2 - 14 *
 *               x_2 + 6 * x_1 * x_2 + 3 * x_2^2)) *
 *               (30 + (2 * x_1 - 3 * x_2)^2 * (18 - 32 * x_1 + 12 * x^2 +
 *               48 * x_2 - 36 * x_1 * x_2 + 27 * x_2^2))
 * \f]
 *
 * This should optimize to f(x) = 3, at x = [0, -1].
 *
 * For more information, please refer to:
 *
 * @code
 * @article{Picheny:2013:BKI:2579829.2579986,
 *   author     = {Picheny, Victor and Wagner, Tobias and Ginsbourger, David},
 *   title      = {A Benchmark of Kriging-based Infill Criteria for Noisy
 *                 Optimization},
 *   journal    = {Struct. Multidiscip. Optim.},
 *   issue_date = {September 2013},
 *   volume     = {48},
 *   number     = {3},
 *   month      = sep,
 *   year       = {2013},
 *   issn       = {1615-147X},
 *   pages      = {607--626},
 *   numpages   = {20},
 *   doi        = {10.1007/s00158-013-0919-4},
 * }
 * @endcode
 */
class GoldsteinPriceFunction
{
 public:
  //! Initialize the GoldsteinPriceFunction.
  GoldsteinPriceFunction();

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
  MatType GetInitialPoint() const { return MatType("0.2; -0.5"); }

  //! Get the final point.
  template<typename MatType = arma::mat>
  MatType GetFinalPoint() const { return MatType("0.0; -1.0"); }

  //! Get the final objective.
  double GetFinalObjective() const { return 3.0; }
};

} // namespace test
} // namespace ens

// Include implementation.
#include "goldstein_price_function_impl.hpp"

#endif // ENSMALLEN_PROBLEMS_GOLDSTEIN_PRICE_FUNCTION_HPP
