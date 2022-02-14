/**
 * @file beale_function.hpp
 * @author Suryoday Basak
 *
 * Definition of the Beale function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_BEALE_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_BEALE_FUNCTION_HPP

namespace ens {
namespace test {

/**
 * The Beale function, defined by
 *
 * \f[
 * f(x_1,x_2) = (1.5 - x_1 + x_1 * x_2)^2 +
 *              (2.25 - x_1 + x_1 * x_2^2)^2 +
 *              (2.625 - x_1 + x_1 * x_2^3)^2
 * \f]
 *
 * This should optimize to f(x) = 0, at x = [3, 0.5].
 *
 * For more information, please refer to:
 *
 * @code
 * @misc{1307.5838,
 *   Author = {Masoumeh Vali},
 *   Title  = {Rotational Mutation Genetic Algorithm on optimizationProblems},
 *   Year   = {2013},
 *   Eprint = {arXiv:1307.5838},
 * }
 * @endcode
 */
class BealeFunction
{
 public:
  //! Initialize the BealeFunction.
  BealeFunction();

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
  MatType GetInitialPoint() const { return MatType("2.8; 0.35"); }

  //! Get the final point.
  template<typename MatType = arma::mat>
  MatType GetFinalPoint() const { return MatType("3.0; 0.5"); }

  //! Get the final objective.
  double GetFinalObjective() const { return 0.0; }
};

} // namespace test
} // namespace ens

// Include implementation.
#include "beale_function_impl.hpp"

#endif // ENSMALLEN_PROBLEMS_BEALE_FUNCTION_HPP
