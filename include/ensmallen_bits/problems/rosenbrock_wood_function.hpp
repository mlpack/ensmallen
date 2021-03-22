/**
 * @file rosenbrock_wood_function.hpp
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Definition of the Rosenbrock-Wood function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_ROSENBROCK_WOOD_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_ROSENBROCK_WOOD_FUNCTION_HPP

#include "generalized_rosenbrock_function.hpp"
#include "wood_function.hpp"

namespace ens {
namespace test {

/**
 * The Generalized Rosenbrock function in 4 dimensions with the Wood Function in
 * four dimensions.  In this function we are actually optimizing a 2x4 matrix of
 * coordinates, not a vector.
 */
class RosenbrockWoodFunction
{
 public:
  //! Initialize the RosenbrockWoodFunction.
  RosenbrockWoodFunction();

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
  void Gradient(const MatType& coordinates, GradType& gradient) const;

  // Note: GetInitialPoint(), GetFinalPoint(), and GetFinalObjective() are not
  // required for using ensmallen to optimize this function!  They are
  // specifically used as a convenience just for ensmallen's testing
  // infrastructure.

  //! Get the starting point.
  template<typename MatType = arma::mat>
  const MatType GetInitialPoint() const
  {
    return arma::conv_to<MatType>::from(initialPoint);
  }

  //! Get the final point.
  template<typename MatType = arma::mat>
  MatType GetFinalPoint() const
  {
    return arma::ones<MatType>(initialPoint.n_rows, initialPoint.n_cols);
  }

  //! Get the final objective.
  double GetFinalObjective() const { return 0.0; }

 private:
  //! Locally-stored initial point.
  arma::mat initialPoint;

  //! Locally-stored Generalized-Rosenbrock function.
  GeneralizedRosenbrockFunction rf;

  //! Locally-stored Wood function.
  WoodFunction wf;
};

} // namespace test
} // namespace ens

// Include implementation.
#include "rosenbrock_wood_function_impl.hpp"

#endif // ENSMALLEN_PROBLEMS_ROSENBROCK_WOOD_FUNCTION_HPP
