/**
 * @file generalized_rosenbrock_function.hpp
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Definition of the Generalized Rosenbrock function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_GENERALIZED_ROSENBROCK_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_GENERALIZED_ROSENBROCK_FUNCTION_HPP

namespace ens {
namespace test {

/**
 * The Generalized Rosenbrock function in n dimensions, defined by
 *  f(x) = sum_i^{n - 1} (f(i)(x))
 *  f_i(x) = 100 * (x_i^2 - x_{i + 1})^2 + (1 - x_i)^2
 *  x_0 = [-1.2, 1, -1.2, 1, ...]
 *
 * This should optimize to f(x) = 0, at x = [1, 1, 1, 1, ...].
 *
 * This function can also be used for stochastic gradient descent (SGD) as a
 * decomposable function (SeparableFunctionType), so there are other
 * overloads of Evaluate() and Gradient() implemented, as well as
 * NumFunctions().
 *
 * For more information, please refer to:
 *
 * @code
 * @phdthesis{Jong1975,
 *   title  = {Analysis of the behavior of a class of genetic adaptive
 *             systems},
 *   author = {De Jong, Kenneth Alan},
 *   school = {Queensland University of Technology},
 *   year   = {1975},
 *   type   = {{PhD} dissertation},
 * }
 * @endcode
 */
template<
    typename MatType = arma::mat,
    typename LabelsType = typename ForwardType<MatType>::urowvec>
class GeneralizedRosenbrockFunctionType
{
 public:
  /*
   * Initialize the GeneralizedRosenbrockFunction.
   *
   * @param n Number of dimensions for the function.
   */
  GeneralizedRosenbrockFunctionType(const size_t n);

  /**
   * Shuffle the order of function visitation. This may be called by the
   * optimizer.
   */
 void Shuffle();

  //! Return 1 (the number of functions).
  size_t NumFunctions() const { return n - 1; }

  /**
   * Evaluate a function for a particular batch-size.
   *
   * @param coordinates The function coordinates.
   * @param begin The first function.
   * @param batchSize Number of points to process.
   */
  typename MatType::elem_type Evaluate(const MatType& coordinates,
                                       const size_t begin,
                                       const size_t batchSize = 1) const;

  /**
   * Evaluate a function with the given coordinates.
   *
   * @param coordinates The function coordinates.
   */
  typename MatType::elem_type Evaluate(const MatType& coordinates) const;

  /**
   * Evaluate the gradient of a function for a particular batch-size.
   *
   * @param coordinates The function coordinates.
   * @param begin The first function.
   * @param gradient The function gradient.
   * @param batchSize Number of points to process.
   */
  template<typename GradType>
  void Gradient(const MatType& coordinates,
                const size_t begin,
                GradType& gradient,
                const size_t batchSize = 1) const;

  /**
   * Evaluate the gradient of a function with the given coordinates.
   *
   * @param coordinates The function coordinates.
   * @param gradient The function gradient.
   */
  template<typename GradType>
  void Gradient(const MatType& coordinates, GradType& gradient) const;

  // Note: GetInitialPoint(), GetFinalPoint(), and GetFinalObjective() are not
  // required for using ensmallen to optimize this function!  They are
  // specifically used as a convenience just for ensmallen's testing
  // infrastructure.

  //! Get the starting point.
  template<typename InputMatType = MatType>
  const InputMatType GetInitialPoint() const
  {
    return conv_to<InputMatType>::from(initialPoint);
  }

  //! Get the final point.
  template<typename InputMatType = MatType>
  const InputMatType GetFinalPoint() const
  {
    InputMatType finalPoint(initialPoint.n_rows, initialPoint.n_cols);
    finalPoint.ones();
    return finalPoint;
  }

  //! Get the final objective.
  double GetFinalObjective() const { return 0.0; }

 private:
  //! Locally-stored Initial point.
  MatType initialPoint;

  //! Number of dimensions for the function.
  size_t n;

  //! For shuffling.
  LabelsType visitationOrder;
};

using GeneralizedRosenbrockFunction = GeneralizedRosenbrockFunctionType<
    arma::mat>;

} // namespace test
} // namespace ens

// Include implementation.
#include "generalized_rosenbrock_function_impl.hpp"

#endif // ENSMALLEN_PROBLEMS_GENERALIZED_ROSENBROCK_FUNCTION_HPP
