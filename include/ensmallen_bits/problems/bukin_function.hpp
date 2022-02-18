/**
 * @file bukin_function.hpp
 * @author Marcus Edel
 *
 * Definition of the Booth function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_BUKIN_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_BUKIN_FUNCTION_HPP

namespace ens {
namespace test {

/**
 * The Bukin function, defined by
 *
 * \f[
 * f(x) = 100 * \sqrt(\left|x_2 - 0.01 * x_1^2 \right|) +
 *    0.01 * \left|x_1 + 10 \right|
 * \f]
 *
 * This should optimize to f(x) = 0, at x = [-10, 1].
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
class BukinFunction
{
 public:
  /**
   * Initialize the BukinFunction.
   *
   * @param epsilon Coefficient to avoid division by zero (numerical stability).
   */
  BukinFunction(const double epsilon = 1e-8);

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

  //! Get the value used for numerical stability.
  double Epsilon() const { return epsilon; }
  //! Modify the value used for numerical stability.
  double& Epsilon() { return epsilon; }

  // Note: GetInitialPoint(), GetFinalPoint(), and GetFinalObjective() are not
  // required for using ensmallen to optimize this function!  They are
  // specifically used as a convenience just for ensmallen's testing
  // infrastructure.

  //! Get the starting point.
  template<typename MatType = arma::mat>
  MatType GetInitialPoint() const { return MatType("-10; -2.0"); }

  //! Get the final point.
  template<typename MatType = arma::mat>
  MatType GetFinalPoint() const { return MatType("-10.0; 1.0"); }

  //! Get the final objective.
  double GetFinalObjective() const { return 0.0; }

 private:
  //! The value used for numerical stability.
  double epsilon;
};

} // namespace test
} // namespace ens

// Include implementation.
#include "bukin_function_impl.hpp"

#endif // ENSMALLEN_PROBLEMS_BUKIN_FUNCTION_HPP
