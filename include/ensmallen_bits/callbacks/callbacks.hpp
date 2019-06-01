/**
 * @file callbacks.hpp
 * @author Marcus Edel
 *
 * The Callback class will invoke the specified callbacks.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CALLBACKS_CALLBACKS_HPP
#define ENSMALLEN_CALLBACKS_CALLBACKS_HPP

#include <ensmallen_bits/callbacks/traits.hpp>

namespace ens {

class Callback
{
 public:
  /**
   * Invoke the Evaluate callback if it exists.
   *
   * @param callback The callback to call.
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param objective Objective value of the current point.
   */
  template<typename CallbackFunctionType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType>
  static typename std::enable_if<
      callbacks::traits::HasEvaluate<
      CallbackFunctionType, callbacks::traits::TypedForms<OptimizerType,
      FunctionType, MatType>::template EvaluateForm>::value, void>::type
  Evaluate(CallbackFunctionType& callback,
           OptimizerType& optimizer,
           FunctionType& function,
           const MatType& coordinates,
           const double objective)
  {
    const_cast<CallbackFunctionType&>(callback).Evaluate(
        optimizer, function, coordinates, objective);
  }

  template<typename CallbackFunctionType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType>
  static typename std::enable_if<
      !callbacks::traits::HasEvaluate<
      CallbackFunctionType, callbacks::traits::TypedForms<OptimizerType,
      FunctionType, MatType>::template EvaluateForm>::value, void>::type
  Evaluate(CallbackFunctionType& /* callback */,
           OptimizerType& /* optimizer */,
           FunctionType& /* function */,
           const MatType& /* coordinates */,
           const double /* objective */)
  { /* Skip, nothing to do here. */ }

  /**
   * Invoke the Gradient callback if it exists.
   *
   * @param callback The callback to call.
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param gradient Matrix that holds the gradient.
   */
  template<typename CallbackFunctionType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType,
           typename GradType>
  static typename std::enable_if<
      callbacks::traits::HasGradient<
      CallbackFunctionType, callbacks::traits::TypedForms<OptimizerType,
      FunctionType, MatType, GradType>::template GradientForm>::value,
      void>::type
  Gradient(CallbackFunctionType& callback,
           OptimizerType& optimizer,
           FunctionType& function,
           const MatType& coordinates,
           GradType& gradient)
  {
    const_cast<CallbackFunctionType&>(callback).Gradient(
        optimizer, function, coordinates, gradient);
  }

  template<typename CallbackFunctionType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType,
           typename GradType>
  static typename std::enable_if<
      !callbacks::traits::HasGradient<
      CallbackFunctionType, callbacks::traits::TypedForms<OptimizerType,
      FunctionType, MatType, GradType>::template GradientForm>::value,
      void>::type
  Gradient(CallbackFunctionType& /* callback */,
           OptimizerType& /* optimizer */,
           FunctionType& /* function */,
           const MatType& /* coordinates */,
           GradType& /* gradient */)
  { /* Skip, nothing to do here. */ }

  /**
   * Invoke the BeginEpoch callback if it exists in the callbacks container.
   *
   * @param callback The callback to call.
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param epoch The index of the current epoch.
   * @param objective Objective value of the current point.
   */
  template<typename CallbackFunctionType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType>
  static typename std::enable_if<
      callbacks::traits::HasBeginEpoch<
      CallbackFunctionType, callbacks::traits::TypedForms<OptimizerType,
      FunctionType, MatType>::template BeginEpochForm>::value, void>::type
  BeginEpoch(CallbackFunctionType& callback,
             OptimizerType& optimizer,
             FunctionType& function,
             const MatType& coordinates,
             const size_t epoch,
             const double objective)
  {
    const_cast<CallbackFunctionType&>(callback).BeginEpoch(
        optimizer, function, coordinates, epoch, objective);
  }

  template<typename CallbackFunctionType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType>
  static typename std::enable_if<
      !callbacks::traits::HasBeginEpoch<
      CallbackFunctionType, callbacks::traits::TypedForms<OptimizerType,
      FunctionType, MatType>::template BeginEpochForm>::value, void>::type
  BeginEpoch(CallbackFunctionType& /* callback */,
             OptimizerType& /* optimizer */,
             FunctionType& /* function */,
             const MatType& /* coordinates */,
             const size_t /* epoch */,
             const double /* objective */)
  { /* Skip, nothing to do here. */ }

  /**
   * Invoke the EndEpoch callback if it exists.
   *
   * @param callback The callback to call.
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param epoch The index of the current epoch.
   * @param objective Objective value of the current point.
   */
  template<typename CallbackFunctionType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType>
  static typename std::enable_if<
      callbacks::traits::HasEndEpoch<
      CallbackFunctionType, callbacks::traits::TypedForms<OptimizerType,
      FunctionType, MatType>::template EndEpochForm>::value, void>::type
  EndEpoch(CallbackFunctionType& callback,
           OptimizerType& optimizer,
           FunctionType& function,
           const MatType& coordinates,
           const size_t epoch,
           const double objective)
  {
    const_cast<CallbackFunctionType&>(callback).EndEpoch(
        optimizer, function, coordinates, epoch, objective);
  }

  template<typename CallbackFunctionType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType>
  static typename std::enable_if<
      !callbacks::traits::HasEndEpoch<
      CallbackFunctionType, callbacks::traits::TypedForms<OptimizerType,
      FunctionType, MatType>::template EndEpochForm>::value, void>::type
  EndEpoch(CallbackFunctionType& /* callback */,
           OptimizerType& /* optimizer */,
           FunctionType& /* function */,
           const MatType& /* coordinates */,
           const size_t /* epoch */,
           const double /* objective */)
  { /* Skip, nothing to do here. */ }
};

} // namespace ens

#endif
