/**
 * @file rastrigin_function.hpp
 * @author Marcus Edel
 *
 * Definition of the Rastrigin function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_RASTRIGIN_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_RASTRIGIN_FUNCTION_HPP

namespace ens {
namespace test {

/**
 * The Rastrigin function, defined by
 *
 * \f[
 * f(x) = 10 * d * \sum_{i=1}^{d} x_i^2 - 10 * \cos(2 * \pi * x_i)
 * \f]
 *
 * This should optimize to f(x) = 0
 * at x = [0, ..., 0].
 *
 * For more information, please refer to:
 *
 * @code
 * @article{Jamil2013,
 *   title   = {Systems of extremal control},
 *   author  = {Rastrigin, L. A.},
 *   journal = {Mir},
 *   year    = {1974}
 * }
 * @endcode
 */
class RastriginFunction
{
 public:
  /*
   * Initialize the RastriginFunction.
   *
   * @param n Number of dimensions for the function.
   */
  RastriginFunction(const size_t n = 2);

  /**
   * Shuffle the order of function visitation. This may be called by the
   * optimizer.
   */
  void Shuffle();

  //! Return 1 (the number of functions).
  size_t NumFunctions() const { return n; }

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
  MatType GetInitialPoint() const
  {
    return arma::conv_to<MatType>::from(initialPoint);
  }

  //! Get the final point.
  template<typename MatType = arma::mat>
  MatType GetFinalPoint() const
  {
    return arma::zeros<MatType>(initialPoint.n_rows, initialPoint.n_cols);
  }

  //! Get the final objective.
  double GetFinalObjective() const { return 0.0; }

 private:
  //! Number of dimensions for the function.
  size_t n;

  //! For shuffling.
  arma::Row<size_t> visitationOrder;

  //! Initial starting point.
  arma::mat initialPoint;
};

} // namespace test
} // namespace ens

// Include implementation.
#include "rastrigin_function_impl.hpp"

#endif // ENSMALLEN_PROBLEMS_RASTRIGIN_FUNCTION_HPP
