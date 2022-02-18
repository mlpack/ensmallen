/**
 * @file styblinski_tang_function.hpp
 * @author Marcus Edel
 *
 * Definition of the Styblinski-Tang function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_STYBLINSKI_TANG_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_STYBLINSKI_TANG_FUNCTION_HPP

namespace ens {
namespace test {

/**
 * The Styblinski-Tang function, defined by
 *
 * \f[
 * f(x) = 0.5 * \sum_{i=1}^{d} x_i^4 - 16_i^2+5x_i
 * \f]
 *
 * This should optimize to f(x) = -39.16599 * d
 * at x = [-2.903534, ..., -2.903534].
 *
 * For more information, please refer to:
 *
 * @code
 * @article{Jamil2013,
 *   title   = {A Literature Survey of Benchmark Functions For Global
 *              Optimization Problems},
 *   author  = {Momin Jamil and Xin{-}She Yang},
 *   journal = {CoRR},
 *   year    = {2013},
 *   url     = {http://arxiv.org/abs/1308.4008}
 * }
 * @endcode
 */
class StyblinskiTangFunction
{
 public:
  /*
   * Initialize the StyblinskiTangFunction.
   *
   * @param n Number of dimensions for the function.
   */
  StyblinskiTangFunction(const size_t n = 2);

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
    MatType result(initialPoint.n_rows, initialPoint.n_cols);
    for (size_t i = 0; i < result.n_elem; ++i)
      result[i] = -2.903534;
    return result;
  }

  //! Get the final objective.
  double GetFinalObjective() const { return -39.16599 * n; }

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
#include "styblinski_tang_function_impl.hpp"

#endif // ENSMALLEN_PROBLEMS_STYBLINSKI_TANG_FUNCTION_HPP
