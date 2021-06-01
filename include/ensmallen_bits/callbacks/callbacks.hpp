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

/**
 * Callbacks are a set of functions that can be applied at given stages of the
 * optimization process. The following callbacks are available:
 *
 * - Evaluate(optimizer, function, coordinates, objective):
 *   called after any call to Evaluate().
 *
 * - StepTaken(optimizer, function, coordinates):
 *   called after any step is taken.
 *
 * - Gradient(optimizer, function, coordinates, gradient):
 *   called whenever the gradient is computed.
 *
 * - BeginEpoch(optimizer, function, coordinates, epoch, objective):
 *   called at the beginning of a pass over the data. The objective may be
 *   exact or an estimate depending on exactObjective's value.
 *
 * - EvaluateConstraint(optimizer, function, coordinates, constraint,
 *                      constraintValue):
 *   called after any call to EvaluateConstraint().
 *
 * - GradientConstraint(optimizer, function, coordinates, constraint,
 *                      constraintGradient):
 *   called after any call to GradientConstraint().
 *
 * - BeginOptimization(optimizer, function, coordinates):
 *   called at the beginning of the optimization.
 *
 * - EndOptimization(optimizer, function, coordinates):
 *   called at the end of the optimization.
 */
class Callback
{
 public:
  /**
   * Invoke the BeginOptimization() callback if it exists.
   *
   * @param callback The callback to call.
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   */
  template<typename CallbackType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType>
  static typename std::enable_if<
      callbacks::traits::HasBeginOptimizationSignature<
      CallbackType, OptimizerType, FunctionType, MatType>::hasBool,
      bool>::type
  BeginOptimizationFunction(CallbackType& callback,
                            OptimizerType& optimizer,
                            FunctionType& function,
                            MatType& coordinates)
  {
    return const_cast<CallbackType&>(callback).BeginOptimization(optimizer,
        function, coordinates);
  }

  template<typename CallbackType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType>
  static typename std::enable_if<
      callbacks::traits::HasBeginOptimizationSignature<
      CallbackType, OptimizerType, FunctionType, MatType>::hasVoid,
      bool>::type
  BeginOptimizationFunction(CallbackType& callback,
                            OptimizerType& optimizer,
                            FunctionType& function,
                            MatType& coordinates)
  {
    const_cast<CallbackType&>(callback).BeginOptimization(optimizer, function,
        coordinates);
    return false;
  }

  template<typename CallbackType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType>
  static typename std::enable_if<
      callbacks::traits::HasBeginOptimizationSignature<
      CallbackType, OptimizerType, FunctionType, MatType>::hasNone,
      bool>::type
  BeginOptimizationFunction(CallbackType& /* callback */,
                            OptimizerType& /* optimizer */,
                            FunctionType& /* function */,
                            MatType& /* coordinates */)
  { return false; }

  /**
   * Iterate over the callbacks and invoke the BeginOptimization() callback if
   * it exists.
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param callbacks The callbacks container.
   */
  template<typename OptimizerType,
           typename FunctionType,
           typename MatType,
           typename... CallbackTypes>
  static bool BeginOptimization(OptimizerType& optimizer,
                                FunctionType& function,
                                MatType& coordinates,
                                CallbackTypes&... callbacks)
  {
    // This will return immediately once a callback returns true.
    bool result = false;
    (void)std::initializer_list<bool>{ result =
        result || Callback::BeginOptimizationFunction(callbacks, optimizer,
            function, coordinates)... };
     return result;
  }

  /**
   * Invoke the EndOptimization() callback if it exists.
   *
   * @param callback The callback to call.
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   */
  template<typename CallbackType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType>
  static typename std::enable_if<callbacks::traits::HasEndOptimizationSignature<
      CallbackType, OptimizerType, FunctionType, MatType>::value,
      bool>::type
  EndOptimizationFunction(CallbackType& callback,
                          OptimizerType& optimizer,
                          FunctionType& function,
                          MatType& coordinates)
  {
    return (const_cast<CallbackType&>(callback).EndOptimization(
        optimizer, function, coordinates), false);
  }

  template<typename CallbackType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType>
  static typename std::enable_if<!callbacks::traits::HasEndOptimizationSignature<
      CallbackType, OptimizerType, FunctionType, MatType>::value,
      bool>::type
  EndOptimizationFunction(CallbackType& /* callback */,
                          OptimizerType& /* optimizer */,
                          FunctionType& /* function */,
                          MatType& /* coordinates */)
  { return false; }

  /**
   * Iterate over the callbacks and invoke the EndOptimization() callback if it
   * exists.
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param callbacks The callbacks container.
   */
  template<typename OptimizerType,
           typename FunctionType,
           typename MatType,
           typename... CallbackTypes>
  static bool EndOptimization(OptimizerType& optimizer,
                              FunctionType& function,
                              MatType& coordinates,
                              CallbackTypes&... callbacks)
  {
    // This will return immediately once a callback returns true.
    bool result = false;
    (void)std::initializer_list<bool>{ result =
        result || Callback::EndOptimizationFunction(callbacks, optimizer,
            function, coordinates)... };
     return result;
  }

  /**
   * Invoke the Evaluate() callback if it exists.
   *
   * @param callback The callback to call.
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param objective Objective value of the current point.
   */
  template<typename CallbackType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType>
  static typename std::enable_if<callbacks::traits::HasEvaluateSignature<
      CallbackType, OptimizerType, FunctionType, MatType>::value,
      bool>::type
  EvaluateFunction(CallbackType& callback,
                   OptimizerType& optimizer,
                   FunctionType& function,
                   const MatType& coordinates,
                   const double objective)
  {
    return (const_cast<CallbackType&>(callback).Evaluate(
        optimizer, function, coordinates, objective), false);
  }

  template<typename CallbackType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType>
  static typename std::enable_if<!callbacks::traits::HasEvaluateSignature<
      CallbackType, OptimizerType, FunctionType, MatType>::value,
      bool>::type
  EvaluateFunction(CallbackType& /* callback */,
                   OptimizerType& /* optimizer */,
                   FunctionType& /* function */,
                   const MatType& /* coordinates */,
                   const double /* objective */)
  { return false; }

  /**
   * Iterate over the callbacks and invoke the Evaluate() callback if it exists.
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param objective Objective value of the current point.
   * @param callbacks The callbacks container.
   */
  template<typename OptimizerType,
           typename FunctionType,
           typename MatType,
           typename... CallbackTypes>
  static bool Evaluate(OptimizerType& optimizer,
                       FunctionType& function,
                       const MatType& coordinates,
                       const double objective,
                       CallbackTypes&... callbacks)
  {
    // This will return immediately once a callback returns true.
    bool result = false;
    (void)(objective);  // prevent spurious compiler warnings
    (void)std::initializer_list<bool>{ result =
        result || Callback::EvaluateFunction(callbacks, optimizer, function,
        coordinates, objective)... };
     return result;
  }

  /**
   * Invoke the EvaluateConstraint() callback if it exists.
   *
   * @param callback The callback to call.
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param constraint The index of the constraint.
   * @param constraintValue Constraint value of the current point.
   */
  template<typename CallbackType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType>
  static typename std::enable_if<
      callbacks::traits::HasEvaluateConstraintSignature<
      CallbackType, OptimizerType, FunctionType, MatType>::value,
      bool>::type
  EvaluateConstraintFunction(CallbackType& callback,
                             OptimizerType& optimizer,
                             FunctionType& function,
                             const MatType& coordinates,
                             const size_t constraint,
                             const double constraintValue)
  {
    return (const_cast<CallbackType&>(callback).EvaluateConstraint(
        optimizer, function, coordinates, constraint, constraintValue), false);
  }

  template<typename CallbackType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType>
  static typename std::enable_if<
      !callbacks::traits::HasEvaluateConstraintSignature<
      CallbackType, OptimizerType, FunctionType, MatType>::value,
      bool>::type
  EvaluateConstraintFunction(CallbackType& /* callback */,
                             OptimizerType& /* optimizer */,
                             FunctionType& /* function */,
                             const MatType& /* coordinates */,
                             const size_t /* constraint */,
                             const double /* constraintValue */)
  { return false; }

  /**
   * Iterate over the callbacks and invoke the EvaluateConstraint() callback if
   * it exists.
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param constraint The index of the constraint.
   * @param constraintValue Constraint value of the current point.
   * @param callbacks The callbacks container.
   */
  template<typename OptimizerType,
           typename FunctionType,
           typename MatType,
           typename... CallbackTypes>
  static bool EvaluateConstraint(OptimizerType& optimizer,
                                 FunctionType& function,
                                 const MatType& coordinates,
                                 const size_t constraint,
                                 const double constraintValue,
                                 CallbackTypes&... callbacks)
  {
    // This will return immediately once a callback returns true.
    bool result = false;
    (void)(constraint);  // prevent spurious compiler warnings
    (void)(constraintValue);
    (void)std::initializer_list<bool>{ result =
        result || Callback::EvaluateConstraintFunction(callbacks, optimizer,
            function, coordinates, constraint, constraintValue)... };
     return result;
  }

  /**
   * Invoke the Gradient() callback if it exists.
   *
   * @param callback The callback to call.
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param gradient Matrix that holds the gradient.
   */
  template<typename CallbackType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType,
           typename GradType>
  static typename std::enable_if<callbacks::traits::HasGradientSignature<
      CallbackType, OptimizerType, FunctionType, MatType, GradType>::value,
      bool>::type
  GradientFunction(CallbackType& callback,
                   OptimizerType& optimizer,
                   FunctionType& function,
                   const MatType& coordinates,
                   GradType& gradient)
  {
    return (const_cast<CallbackType&>(callback).Gradient(
        optimizer, function, coordinates, gradient), false);
  }

  template<typename CallbackType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType,
           typename GradType>
  static typename std::enable_if<!callbacks::traits::HasGradientSignature<
      CallbackType, OptimizerType, FunctionType, MatType, GradType>::value,
      bool>::type
  GradientFunction(CallbackType& /* callback */,
                   OptimizerType& /* optimizer */,
                   FunctionType& /* function */,
                   const MatType& /* coordinates */,
                   GradType& /* gradient */)
  { return false; }

  /**
   * Iterate over the callbacks and invoke the Gradient() callback if it exists.
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param gradient Matrix that holds the gradient.
   * @param callbacks The callbacks container.
   */
  template<typename OptimizerType,
           typename FunctionType,
           typename MatType,
           typename GradType,
           typename... CallbackTypes>
  static bool Gradient(OptimizerType& optimizer,
                       FunctionType& function,
                       const MatType& coordinates,
                       GradType& gradient,
                       CallbackTypes&... callbacks)
  {
    // This will return immediately once a callback returns true.
    bool result = false;
    (void)std::initializer_list<bool>{ result =
        result || Callback::GradientFunction(callbacks, optimizer, function,
        coordinates, gradient)... };
     return result;
  }

  /**
   * Invoke the GradientConstraint() callback if it exists.
   *
   * @param callback The callback to call.
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param gradient Matrix that holds the gradient.
   */
  template<typename CallbackType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType,
           typename GradType>
  static typename std::enable_if<
      callbacks::traits::HasGradientConstraintSignature<
      CallbackType, OptimizerType, FunctionType, MatType, GradType>::value,
      bool>::type
  GradientConstraintFunction(CallbackType& callback,
                             OptimizerType& optimizer,
                             FunctionType& function,
                             const MatType& coordinates,
                             const size_t constraint,
                             GradType& gradient)
  {
    return (const_cast<CallbackType&>(callback).GradientConstraint(
        optimizer, function, coordinates, constraint, gradient), false);
  }

  template<typename CallbackType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType,
           typename GradType>
  static typename std::enable_if<
      !callbacks::traits::HasGradientConstraintSignature<
      CallbackType, OptimizerType, FunctionType, MatType, GradType>::value,
      bool>::type
  GradientConstraintFunction(CallbackType& /* callback */,
                             OptimizerType& /* optimizer */,
                             FunctionType& /* function */,
                             const MatType& /* coordinates */,
                             const size_t /* constraint */,
                             GradType& /* gradient */)
  { return false; }

  /**
   * Iterate over the callbacks and invoke the GradientConstraint() callback if
   * it exists.
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param gradient Matrix that holds the gradient.
   * @param callbacks The callbacks container.
   */
  template<typename OptimizerType,
           typename FunctionType,
           typename MatType,
           typename GradType,
           typename... CallbackTypes>
  static bool Gradient(OptimizerType& optimizer,
                       FunctionType& function,
                       const MatType& coordinates,
                       const size_t constraint,
                       GradType& gradient,
                       CallbackTypes&... callbacks)
  {
    // This will return immediately once a callback returns true.
    bool result = false;
    (void)(constraint);  // prevent spurious compiler warnings
    (void)std::initializer_list<bool>{ result =
        result || Callback::GradientConstraintFunction(callbacks, optimizer,
        function, coordinates, constraint, gradient)... };
     return result;
  }

  /**
   * Iterate over the callbacks and invoke the Evaluate() and Gradient()
   * callback if it exists.
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param objective Objective value of the current point.
   * @param gradient Matrix that holds the gradient.
   * @param callbacks The callbacks container.
   */
  template<typename OptimizerType,
           typename FunctionType,
           typename MatType,
           typename GradType,
           typename... CallbackTypes>
  static bool EvaluateWithGradient(OptimizerType& optimizer,
                                   FunctionType& function,
                                   const MatType& coordinates,
                                   const double objective,
                                   GradType& gradient,
                                   CallbackTypes&... callbacks)
  {
    // This will return immediately once a callback returns true.
    bool result = false;
    (void)(objective);  // prevent spurious compiler warnings
    (void)std::initializer_list<bool>{ result =
        result || Callback::EvaluateFunction(callbacks, optimizer, function,
        coordinates, objective)... };

    (void)std::initializer_list<bool>{ result =
        result || Callback::GradientFunction(callbacks, optimizer, function,
        coordinates, gradient)... };
     return result;
  }

  /**
   * Invoke the BeginEpoch() callback if it exists.
   *
   * @param callback The callback to call.
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param epoch The index of the current epoch.
   * @param objective Objective value of the current point.
   */
  template<typename CallbackType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType>
  static typename std::enable_if<callbacks::traits::HasBeginEpochSignature<
      CallbackType, OptimizerType, FunctionType, MatType>::value, bool>::type
  BeginEpochFunction(CallbackType& callback,
                     OptimizerType& optimizer,
                     FunctionType& function,
                     const MatType& coordinates,
                     const size_t epoch,
                     const double objective)
  {
    return (const_cast<CallbackType&>(callback).BeginEpoch(
        optimizer, function, coordinates, epoch, objective), false);
  }

  template<typename CallbackType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType>
  static typename std::enable_if<!callbacks::traits::HasBeginEpochSignature<
      CallbackType, OptimizerType, FunctionType, MatType>::value, bool>::type
  BeginEpochFunction(CallbackType& /* callback  */,
                     OptimizerType& /* optimizer */,
                     FunctionType& /* function  */,
                     const MatType& /* coordinates  */,
                     const size_t /* epoch  */,
                     const double /* objective */)
  { return false; }

  /**
   * Iterate over all callbacks and invoke the BeginEpoch() callback if it
   * exists.
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param epoch The index of the current epoch.
   * @param objective Objective value of the current point.
   * @param callbacks The callbacks container.
   */
  template<typename OptimizerType,
           typename FunctionType,
           typename MatType,
           typename... CallbackTypes>
  static bool BeginEpoch(OptimizerType& optimizer,
                         FunctionType& function,
                         const MatType& coordinates,
                         const size_t epoch,
                         const double objective,
                         CallbackTypes&... callbacks)
  {
    // This will return immediately once a callback returns true.
    bool result = false;
    (void)(epoch);  // prevent spurious compiler warnings
    (void)(objective);
    (void)std::initializer_list<bool>{ result =
        result || Callback::BeginEpochFunction(callbacks, optimizer, function,
        coordinates, epoch, objective)... };
     return result;
  }

  /**
   * Invoke the EndEpoch() callback if it exists.
   *
   * @param callback The callback to call.
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param epoch The index of the current epoch.
   * @param objective Objective value of the current point.
   */
  template<typename CallbackType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType>
  static typename std::enable_if<callbacks::traits::HasEndEpochSignature<
      CallbackType, OptimizerType, FunctionType, MatType>::hasBool, bool>::type
  EndEpochFunction(CallbackType& callback,
                   OptimizerType& optimizer,
                   FunctionType& function,
                   const MatType& coordinates,
                   const size_t epoch,
                   const double objective)
  {
    return const_cast<CallbackType&>(callback).EndEpoch(
        optimizer, function, coordinates, epoch, objective);
  }

  template<typename CallbackType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType>
  static typename std::enable_if<callbacks::traits::HasEndEpochSignature<
      CallbackType, OptimizerType, FunctionType, MatType>::hasVoid, bool>::type
  EndEpochFunction(CallbackType& callback,
                   OptimizerType& optimizer,
                   FunctionType& function,
                   const MatType& coordinates,
                   const size_t epoch,
                   const double objective)
  {
    const_cast<CallbackType&>(callback).EndEpoch(
        optimizer, function, coordinates, epoch, objective);
    return false;
  }

  template<typename CallbackType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType>
  static typename std::enable_if<callbacks::traits::HasEndEpochSignature<
      CallbackType, OptimizerType, FunctionType, MatType>::hasNone, bool>::type
  EndEpochFunction(CallbackType& /* callback */,
                   OptimizerType& /* optimizer */,
                   FunctionType& /* function */,
                   const MatType& /* coordinates */,
                   const size_t /* epoch */,
                   const double /* objective */)
  { return false; }

  /**
   * Iterate over all callbacks and invoke the EndEpoch() callback if it exists.
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param epoch The index of the current epoch.
   * @param objective Objective value of the current point.
   * @param callbacks The callbacks container.
   */
  template<typename OptimizerType,
           typename FunctionType,
           typename MatType,
           typename... CallbackTypes>
  static bool EndEpoch(OptimizerType& optimizer,
                       FunctionType& function,
                       const MatType& coordinates,
                       const size_t epoch,
                       const double objective,
                       CallbackTypes&... callbacks)
  {
    // This will return immediately once a callback returns true.
    bool result = false;
    (void)(epoch);  // prevent spurious compiler warnings
    (void)(objective);
    (void)std::initializer_list<bool>{ result =
        result || Callback::EndEpochFunction(callbacks, optimizer, function,
        coordinates, epoch, objective)... };
     return result;
  }

  /**
   * Invoke the StepTaken() callback if it exists.
   *
   * @param callback The callback to call.
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   */
  template<typename CallbackType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType>
  static typename std::enable_if<
      callbacks::traits::HasStepTakenSignature<
      CallbackType, OptimizerType, FunctionType, MatType>::hasBool,
      bool>::type
  StepTakenFunction(CallbackType& callback,
                    OptimizerType& optimizer,
                    FunctionType& function,
                    MatType& coordinates)
  {
    return const_cast<CallbackType&>(callback).StepTaken(optimizer,
        function, coordinates);
  }

  template<typename CallbackType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType>
  static typename std::enable_if<
      callbacks::traits::HasStepTakenSignature<
      CallbackType, OptimizerType, FunctionType, MatType>::hasVoid,
      bool>::type
  StepTakenFunction(CallbackType& callback,
                    OptimizerType& optimizer,
                    FunctionType& function,
                    MatType& coordinates)
  {
    const_cast<CallbackType&>(callback).StepTaken(optimizer, function,
        coordinates);
    return false;
  }

  template<typename CallbackType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType>
  static typename std::enable_if<
      callbacks::traits::HasStepTakenSignature<
      CallbackType, OptimizerType, FunctionType, MatType>::hasNone,
      bool>::type
  StepTakenFunction(CallbackType& /* callback */,
                    OptimizerType& /* optimizer */,
                    FunctionType& /* function */,
                    MatType& /* coordinates */)
  { return false; }

 /**
  * Invoke the GenerationalStepTaken() callback if it exists.
  * Specialization for MultiObjective case.
  *
  * @param callback The callback to call.
  * @param optimizer The optimizer used to update the function.
  * @param function Function to optimize.
  * @param coordinates Starting point.
  * @param objectives The set of calculated objectives so far.
  * @param frontIndices The indices of the members belonging to Pareto Front.
  */
  template<typename CallbackType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType,
           typename ObjectivesVecType,
           typename IndicesType>
  static typename std::enable_if<
      callbacks::traits::HasGenerationalStepTakenSignature<
      CallbackType, OptimizerType, FunctionType, MatType, ObjectivesVecType,
      IndicesType>::hasBool, bool>::type
  GenerationalStepTakenFunction(CallbackType& callback,
                                OptimizerType& optimizer,
                                FunctionType& function,
                                MatType& coordinates,
                                ObjectivesVecType& objectives,
                                IndicesType& frontIndices)
  {
    return const_cast<CallbackType&>(callback).GenerationalStepTaken(
        optimizer, function, coordinates, objectives, frontIndices);
  }

  template<typename CallbackType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType,
           typename ObjectivesVecType,
           typename IndicesType>
  static typename std::enable_if<
      callbacks::traits::HasGenerationalStepTakenSignature<
      CallbackType, OptimizerType, FunctionType, MatType, ObjectivesVecType,
      IndicesType>::hasVoid, bool>::type
  GenerationalStepTakenFunction(CallbackType& callback,
                                OptimizerType& optimizer,
                                FunctionType& function,
                                MatType& coordinates,
                                ObjectivesVecType& objectives,
                                IndicesType& frontIndices)
  {
    const_cast<CallbackType&>(callback).GenerationalStepTaken(
        optimizer, function, coordinates, objectives, frontIndices);

    return false;
  }

  template<typename CallbackType,
           typename OptimizerType,
           typename FunctionType,
           typename MatType,
           typename ObjectivesVecType,
           typename IndicesType>
  static typename std::enable_if<
      callbacks::traits::HasGenerationalStepTakenSignature<
      CallbackType, OptimizerType, FunctionType, MatType, ObjectivesVecType,
      IndicesType>::hasNone, bool>::type
  GenerationalStepTakenFunction(CallbackType& /* callback */,
                                OptimizerType& /* optimizer */,
                                FunctionType& /* function */,
                                MatType& /* coordinates */,
                                ObjectivesVecType& /* objectives */,
                                IndicesType& /* frontIndices */)
  { return false; }

  /**
   * Iterate over the callbacks and invoke the StepTaken() callback if it
   * exists.
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param callbacks The callbacks container.
   */
  template<typename OptimizerType,
           typename FunctionType,
           typename MatType,
           typename... CallbackTypes>
  static bool StepTaken(OptimizerType& optimizer,
                        FunctionType& function,
                        MatType& coordinates,
                        CallbackTypes&... callbacks)
  {
    // This will return immediately once a callback returns true.
    bool result = false;
    (void)std::initializer_list<bool>{ result =
        result || Callback::StepTakenFunction(callbacks, optimizer,
            function, coordinates)... };
     return result;
  }

/**
 * Iterate over the callbacks and invoke the GenerationalStepTaken() callback if it
 * exists.
 *
 * Specialization for MultiObjective case.
 *
 * @param optimizer The optimizer used to update the function.
 * @param function Function to optimize.
 * @param coordinates Starting point.
 * @param objectives The set of calculated objectives so far.
 * @param frontIndices The indices of the members belonging to Pareto Front.
 * @param callbacks The callbacks container.
 */
template<typename OptimizerType,
         typename FunctionType,
         typename ObjectivesVecType,
         typename IndicesType,
         typename MatType,
         typename ...CallbackTypes>
  static bool GenerationalStepTaken(OptimizerType& optimizer,
                                    FunctionType& functions,
                                    MatType& coordinates,
                                    ObjectivesVecType& objectives,
                                    IndicesType& frontIndices,
                                    CallbackTypes&... callbacks)
{
  // This will return immediately once a callback returns true.
  bool result = false;
  (void)std::initializer_list<bool>{ result =
    result || Callback::GenerationalStepTakenFunction(callbacks, optimizer,
      functions, coordinates, objectives, frontIndices)... };
  return result;
}
};
} // namespace ens

#endif
