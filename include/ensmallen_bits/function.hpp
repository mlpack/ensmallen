/**
 * @file function.hpp
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * The Function class is a wrapper class for any objective function that
 * provides any of the functions that an optimizer might use.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_FUNCTION_HPP
#define ENSMALLEN_FUNCTION_HPP

namespace ens {

template<typename FunctionType,
         typename MatType,
         typename GradType,
         typename OptimizerType>
class Function;

} // namespace ens

#include "function/traits.hpp"
#include "function/static_checks.hpp"
#include "function/add_evaluate.hpp"
#include "function/add_gradient.hpp"
#include "function/add_evaluate_with_gradient.hpp"
#include "function/add_decomposable_evaluate.hpp"
#include "function/add_decomposable_gradient.hpp"
#include "function/add_decomposable_evaluate_with_gradient.hpp"

#include "callbacks/callbacks.hpp"

namespace ens {

struct EmptyOptimizer { }; // Empty optimizer class.

/**
 * The Function class is a wrapper class for any FunctionType that will add any
 * possible derived methods.  For instance, if the given FunctionType has
 * Evaluate() and Gradient(), then Function<FunctionType> will have
 * EvaluateWithGradient().  This infrastructure allows two things:
 *
 *   1. Optimizers can expect FunctionTypes to have a wider array of functions
 *      than those FunctionTypes may actually implement.
 *
 *   2. FunctionTypes don't need to implement every single method that an
 *      optimizer might require, just those from which every method can be
 *      inferred.
 *
 * This class works by inheriting from a large set of "mixin" classes that
 * provide missing functions, if needed.  For instance, the AddGradient<> mixin
 * will provide a Gradient() method if the given FunctionType implements an
 * EvaluateWithGradient() method.
 *
 * Since all of the casting is static and each of the mixin classes is an empty
 * class, there should be no runtime overhead at all for this functionality.  In
 * addition, this class does not (to the best of my knowledge) rely on any
 * undefined behavior.
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         typename OptimizerType = EmptyOptimizer>
class Function :
    public AddDecomposableEvaluateWithGradientStatic<FunctionType, MatType,
        GradType, OptimizerType>,
    public AddDecomposableEvaluateWithGradientConst<FunctionType, MatType,
        GradType, OptimizerType>,
    public AddDecomposableEvaluateWithGradient<FunctionType, MatType,
        GradType, OptimizerType>,
    public AddDecomposableGradientStatic<FunctionType, MatType,
        GradType, OptimizerType>,
    public AddDecomposableGradientConst<FunctionType, MatType,
        GradType, OptimizerType>,
    public AddDecomposableGradient<FunctionType, MatType,
        GradType, OptimizerType>,
    public AddDecomposableEvaluateStatic<FunctionType, MatType,
        GradType, OptimizerType>,
    public AddDecomposableEvaluateConst<FunctionType, MatType,
        GradType, OptimizerType>,
    public AddDecomposableEvaluate<FunctionType, MatType,
        GradType, OptimizerType>,
    public AddEvaluateWithGradientStatic<FunctionType, MatType,
        GradType, OptimizerType>,
    public AddEvaluateWithGradientConst<FunctionType, MatType,
        GradType, OptimizerType>,
    public AddEvaluateWithGradient<FunctionType, MatType,
        GradType, OptimizerType>,
    public AddGradientStatic<FunctionType, MatType, GradType, OptimizerType>,
    public AddGradientConst<FunctionType, MatType, GradType, OptimizerType>,
    public AddGradient<FunctionType, MatType, GradType, OptimizerType>,
    public AddEvaluateStatic<FunctionType, MatType, GradType, OptimizerType>,
    public AddEvaluateConst<FunctionType, MatType, GradType, OptimizerType>,
    public AddEvaluate<FunctionType, MatType, GradType, OptimizerType>,
    public FunctionType
{
 public:
  // All of the mixin classes either reflect existing functionality or provide
  // an unconstructable overload with the same name, so we can use using
  // declarations here to ensure that they are all accessible.  Since we don't
  // know what FunctionType has, we can't use any using declarations there.
  using AddDecomposableEvaluateWithGradientStatic<
      FunctionType, MatType, GradType, OptimizerType>::EvaluateWithGradient;
  using AddDecomposableEvaluateWithGradientConst<
      FunctionType, MatType, GradType, OptimizerType>::EvaluateWithGradient;
  using AddDecomposableEvaluateWithGradient<
      FunctionType, MatType, GradType, OptimizerType>::EvaluateWithGradient;
  using AddDecomposableGradientStatic<
      FunctionType, MatType, GradType, OptimizerType>::Gradient;
  using AddDecomposableGradientConst<FunctionType, MatType,
      GradType, OptimizerType>::Gradient;
  using AddDecomposableGradient<FunctionType, MatType,
      GradType, OptimizerType>::Gradient;
  using AddDecomposableEvaluateStatic<
      FunctionType, MatType, GradType, OptimizerType>::Evaluate;
  using AddDecomposableEvaluateConst<FunctionType, MatType,
      GradType, OptimizerType>::Evaluate;
  using AddDecomposableEvaluate<FunctionType, MatType,
      GradType, OptimizerType>::Evaluate;
  using AddEvaluateWithGradientStatic<FunctionType, MatType,
      GradType, OptimizerType>::EvaluateWithGradient;
  using AddEvaluateWithGradientConst<FunctionType, MatType,
      GradType, OptimizerType>::EvaluateWithGradient;
  using AddEvaluateWithGradient<FunctionType, MatType,
      GradType, OptimizerType>::EvaluateWithGradient;
  using AddGradientStatic<FunctionType, MatType,
      GradType, OptimizerType>::Gradient;
  using AddGradientConst<FunctionType, MatType,
      GradType, OptimizerType>::Gradient;
  using AddGradient<FunctionType, MatType, GradType, OptimizerType>::Gradient;
  using AddEvaluateStatic<FunctionType, MatType,
      GradType, OptimizerType>::Evaluate;
  using AddEvaluateConst<FunctionType, MatType,
      GradType, OptimizerType>::Evaluate;
  using AddEvaluate<FunctionType, MatType, GradType, OptimizerType>::Evaluate;

  /* Callbacks entry points. */

  /**
   * Register the optimizer and the given callbacks.
   *
   * @param optimizer reference to the optimizer.
   * @param callbacks The functions to call.
   */
  void Register(OptimizerType optimizer)
  {
    this->optimizer = optimizer;
  }

  /* Callbacks entry point for differentiable separable functions. */

  /**
   * Return the objective function for the given coordinates, starting at the
   * given decomposable function using the given batch size and call the
   * callback function at the end.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param begin Index of first function to evaluate.
   * @param batchSize Number of functions to evaluate.
   * @param callbacks The functions to call.
   * @return The calculated objective.
   */
  template<std::size_t I = 0, typename... CType>
  typename std::enable_if<sizeof...(CType) != 0, double>::type
  Evaluate(const MatType& coordinates,
           size_t begin,
           size_t batchSize,
           CType&... callbacks)
  {
    const double objective = Evaluate(coordinates, begin, batchSize);
    (void)std::initializer_list<int>{
        (Callback::Evaluate(callbacks, optimizer, *this, coordinates,
        objective), 0)...};
    return objective;
  }

  /**
   * Calculate the gradient and store it in the given matrix, starting at the
   * given decomposable function index and using the given batch size and call
   * the callback function at the end.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param begin Index of decomposable function to start at.
   * @param gradient Matrix to store the gradient into.
   * @param batchSize Number of decomposable functions to calculate for.
   * @param callbacks The functions to call.
   * @return The calculated objective.
   */
  template<std::size_t I = 0, typename... CType>
  typename std::enable_if<sizeof...(CType) != 0, void>::type
  Gradient(const MatType& coordinates,
           size_t begin,
           MatType& gradient,
           const size_t batchSize,
           CType&... /* callbacks */)
  {
    Gradient(coordinates, begin, gradient, batchSize);
  }

  /**
   * Return both the evaluated objective function and its gradient, storing the
   * gradient in the given matrix, starting at the given decomposable function,
   * using the given batch size and call the callback functions at the end.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param begin Index of decomposable function to begin with.
   * @param gradient Matrix to store the gradient into.
   * @param batchSize Number of decomposable functions to evaluate.
   * @param callbacks The functions to call.
   * @return The calculated objective.
   */
  template<std::size_t I = 0, typename... CType>
  typename std::enable_if<sizeof...(CType) != 0, double>::type
  EvaluateWithGradient(const MatType& coordinates,
                       size_t begin,
                       GradType& gradient,
                       size_t batchSize,
                       CType&... callbacks)
  {
    const double objective = EvaluateWithGradient(coordinates, begin, gradient,
        batchSize);

    (void)std::initializer_list<int>{
        (Callback::Evaluate(callbacks, optimizer, *this, coordinates,
        objective), 0)...};

    (void)std::initializer_list<int>{
        (Callback::Gradient(callbacks, optimizer, *this, coordinates,
        gradient), 0)...};

    return objective;
  }

  /* Callbacks entry point for all functions. */

  /**
   * Callback function called at the beginning of a pass over the data. The
   * objective may be exact or an estimate depending on exactObjective's value.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param epoch The index of the current epoch.
   * @param callbacks The functions to call.
   */
  template<std::size_t I = 0, typename... CType>
  typename std::enable_if<sizeof...(CType) != 0, void>::type
  BeginEpoch(const MatType& coordinates,
             const size_t epoch,
             const double objective,
             CType&... callbacks)
  {
    (void)std::initializer_list<int>{
        (Callback::BeginEpoch(callbacks, optimizer, *this, coordinates, epoch,
        objective), 0)...};
  }

  // Provide a dummy overload so the name 'BeginEpoch' exists for this object.
  template<typename... CType>
  void BeginEpoch(const MatType& /* coordinates */,
                  const size_t /* epoch */,
                  const double /* objective */)
  { /* Nothing to do here. */ }

  /**
   * Callback function called at the end of a pass over the data. The
   * objective may be exact or an estimate depending on exactObjective's value.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param epoch The index of the current epoch.
   * @param callbacks The functions to call.
   */
  template<std::size_t I = 0, typename... CType>
  typename std::enable_if<sizeof...(CType) != 0, void>::type
  EndEpoch(const MatType& coordinates,
           const size_t epoch,
           const double objective,
           CType&... callbacks)
  {
    (void)std::initializer_list<int>{
        (Callback::EndEpoch(callbacks, optimizer, *this, coordinates, epoch,
        objective), 0)...};
  }

  // Provide a dummy overload so the name 'EndEpoch' exists for this object.
  template<typename... CType>
  void EndEpoch(const MatType& /* coordinates */,
                const size_t /* epoch */,
                const double /* objective */)
  { /* Nothing to do here. */ }

 private:
  //! The optimizer used to update the function.
  OptimizerType optimizer;
};

} // namespace ens

#endif
