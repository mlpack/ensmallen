/**
 * @file traits.hpp
 * @author Marcus Edel
 *
 * This file provides metaprogramming utilities for detecting certain members of
 * CallbackType classes.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CALLBACKS_TRAITS_HPP
#define ENSMALLEN_CALLBACKS_TRAITS_HPP

#include <ensmallen_bits/function/sfinae_utility.hpp>

namespace ens {
namespace callbacks {
namespace traits {

//! Detect an Evaluate() method.
ENS_HAS_EXACT_METHOD_FORM(Evaluate, HasEvaluate)
//! Detect an EvaluateConstraint() method.
ENS_HAS_EXACT_METHOD_FORM(EvaluateConstraint, HasEvaluateConstraint)
//! Detect an Gradient() method.
ENS_HAS_EXACT_METHOD_FORM(Gradient, HasGradient)
//! Detect an GradientConstraint() method.
ENS_HAS_EXACT_METHOD_FORM(GradientConstraint, HasGradientConstraint)
//! Detect an BeginOptimization() method.
ENS_HAS_EXACT_METHOD_FORM(BeginOptimization, HasBeginOptimization)
//! Detect an EndOptimization() method.
ENS_HAS_EXACT_METHOD_FORM(EndOptimization, HasEndOptimization)
//! Detect an BeginEpoch() method.
ENS_HAS_EXACT_METHOD_FORM(BeginEpoch, HasBeginEpoch)
//! Detect an EndEpoch() method.
ENS_HAS_EXACT_METHOD_FORM(EndEpoch, HasEndEpoch)
//! Detect an StepTaken() method.
ENS_HAS_EXACT_METHOD_FORM(StepTaken, HasStepTaken)
//! Detect an BatchSize() method.
ENS_HAS_EXACT_METHOD_FORM(BatchSize, HasBatchSize)
//! Detect an MaxIterations() method.
ENS_HAS_EXACT_METHOD_FORM(MaxIterations, HasMaxIterations)
//! Detect an NumFunctions() method.
ENS_HAS_EXACT_METHOD_FORM(NumFunctions, HasNumFunctions)

template<typename OptimizerType,
         typename FunctionType,
         typename MatType,
         typename GradType = MatType>
struct TypedForms
{
  //! This is the form of a bool Evaluate() callback method.
  template<typename CallbackType>
  using EvaluateBoolForm =
      void(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            const MatType&,
                            const double);

  //! This is the form of a void Evaluate() callback method.
  template<typename CallbackType>
  using EvaluateVoidForm =
      void(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            const MatType&,
                            const double);

  //! This is the form of a bool EvaluateConstraint() callback method.
  template<typename CallbackType>
  using EvaluateConstraintBoolForm =
      void(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            const MatType&,
                            const size_t,
                            const double);

  //! This is the form of a void EvaluateConstraint() callback method.
  template<typename CallbackType>
  using EvaluateConstraintVoidForm =
      void(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            const MatType&,
                            const size_t,
                            const double);

  //! This is the form of a bool Gradient() callback method.
  template<typename CallbackType>
  using GradientBoolForm =
      void(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            const MatType&,
                            const MatType&);

  //! This is the form of a void Gradient() callback method.
  template<typename CallbackType>
  using GradientVoidForm =
      void(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            const MatType&,
                            const MatType&);

  //! This is the form of a bool GradientConstraint() callback method.
  template<typename CallbackType>
  using GradientConstraintBoolForm =
      void(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            const MatType&,
                            const size_t,
                            const MatType&);

  //! This is the form of a void GradientConstraint() callback method.
  template<typename CallbackType>
  using GradientConstraintVoidForm =
      void(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            const MatType&,
                            const size_t,
                            const MatType&);

  //! This is the form of a bool BeginOptimization() callback method.
  template<typename CallbackType>
  using BeginOptimizationBoolForm =
      bool(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            MatType&);

  //! This is the form of a void BeginOptimization() callback method.
  template<typename CallbackType>
  using BeginOptimizationVoidForm =
      void(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            MatType&);

  //! This is the form of a bool EndOptimization() callback method.
  template<typename CallbackType>
  using EndOptimizationBoolForm =
      bool(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            MatType&);

  //! This is the form of a void EndOptimization() callback method.
  template<typename CallbackType>
  using EndOptimizationVoidForm =
      void(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            MatType&);

  //! This is the form of a bool BeginEpoch() callback method.
  template<typename CallbackType>
  using BeginEpochBoolForm =
      bool(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            const MatType&,
                            const size_t,
                            const double);

  //! This is the form of a void BeginEpoch() callback method.
  template<typename CallbackType>
  using BeginEpochVoidForm =
      void(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            const MatType&,
                            const size_t,
                            const double);

  //! This is the form of a bool EndEpoch() callback method.
  template<typename CallbackType>
  using EndEpochBoolForm =
      bool(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            const MatType&,
                            const size_t,
                            const double);

  //! This is the form of a void EndEpoch() callback method.
  template<typename CallbackType>
  using EndEpochVoidForm =
      void(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            const MatType&,
                            const size_t,
                            const double);

  //! This is the form of a bool StepTaken() callback method.
  template<typename CallbackType>
  using StepTakenBoolForm =
      bool(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            const MatType&);

  //! This is the form of a void StepTaken() callback method.
  template<typename CallbackType>
  using StepTakenVoidForm =
      void(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            const MatType&);
};

//! Utility struct, check if either void BeginOptimization() or
//! bool BeginOptimization() exists.
template<typename CallbackType,
         typename OptimizerType,
         typename FunctionType,
         typename MatType>
struct HasBeginOptimizationSignature
{
  const static bool hasBool =
      HasBeginOptimization<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template BeginOptimizationBoolForm>::value &&
      !HasBeginOptimization<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template BeginOptimizationVoidForm>::value;

  const static bool hasVoid =
      !HasBeginOptimization<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template BeginOptimizationBoolForm>::value &&
      HasBeginOptimization<CallbackType, TypedForms<OptimizerType,
         FunctionType, MatType>::template BeginOptimizationVoidForm>::value;

  const static bool hasNone =
      !HasBeginOptimization<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template BeginOptimizationBoolForm>::value &&
      !HasBeginOptimization<CallbackType, TypedForms<OptimizerType,
         FunctionType, MatType>::template BeginOptimizationVoidForm>::value;
};

//! Utility struct, check if either void Evaluate() or bool Evaluate()
//! exists.
template<typename CallbackType,
         typename OptimizerType,
         typename FunctionType,
         typename MatType>
struct HasEvaluateSignature
{
  const static bool value =
      HasEvaluate<CallbackType, TypedForms<OptimizerType,
      FunctionType, MatType>::template EvaluateBoolForm>::value ||
      HasEvaluate<CallbackType, TypedForms<OptimizerType,
      FunctionType, MatType>::template EvaluateVoidForm>::value;
};

//! Utility struct, check if either void EvaluateConstraint() or
//! bool EvaluateConstraint() exists.
template<typename CallbackType,
         typename OptimizerType,
         typename FunctionType,
         typename MatType>
struct HasEvaluateConstraintSignature
{
  const static bool value =
      HasEvaluateConstraint<CallbackType, TypedForms<OptimizerType,
      FunctionType, MatType>::template EvaluateConstraintBoolForm>::value ||
      HasEvaluateConstraint<CallbackType, TypedForms<OptimizerType,
      FunctionType, MatType>::template EvaluateConstraintVoidForm>::value;
};

//! Utility struct, check if either void Gradient() or bool Gradient()
//! exists.
template<typename CallbackType,
         typename OptimizerType,
         typename FunctionType,
         typename MatType,
         typename Gradient>
struct HasGradientSignature
{
  const static bool value =
      HasGradient<CallbackType, TypedForms<OptimizerType,
      FunctionType, MatType, Gradient>::template GradientBoolForm>::value ||
      HasGradient<CallbackType, TypedForms<OptimizerType,
      FunctionType, MatType, Gradient>::template GradientVoidForm>::value;
};

//! Utility struct, check if either void GradientConstraint() or
//! bool GradientConstraint() exists.
template<typename CallbackType,
         typename OptimizerType,
         typename FunctionType,
         typename MatType,
         typename Gradient>
struct HasGradientConstraintSignature
{
  const static bool value =
      HasGradientConstraint<CallbackType, TypedForms<OptimizerType,
      FunctionType, MatType,
          Gradient>::template GradientConstraintBoolForm>::value ||
      HasGradientConstraint<CallbackType, TypedForms<OptimizerType,
      FunctionType, MatType,
          Gradient>::template GradientConstraintVoidForm>::value;
};

//! Utility struct, check if either void EndOptimization() or
//! bool EndOptimization() exists.
template<typename CallbackType,
         typename OptimizerType,
         typename FunctionType,
         typename MatType>
struct HasEndOptimizationSignature
{
  const static bool value =
      HasEndOptimization<CallbackType, TypedForms<OptimizerType,
      FunctionType, MatType>::template EndOptimizationBoolForm>::value ||
      HasEndOptimization<CallbackType, TypedForms<OptimizerType,
      FunctionType, MatType>::template EndOptimizationVoidForm>::value;
};

//! Utility struct, check if either void BeginEpoch() or bool BeginEpoch()
//! exists.
template<typename CallbackType,
         typename OptimizerType,
         typename FunctionType,
         typename MatType>
struct HasBeginEpochSignature
{
  const static bool value =
      HasBeginEpoch<CallbackType, TypedForms<OptimizerType,
      FunctionType, MatType>::template BeginEpochBoolForm>::value ||
      HasBeginEpoch<CallbackType, TypedForms<OptimizerType,
      FunctionType, MatType>::template BeginEpochVoidForm>::value;
};

//! Utility struct, check if either void EndEpoch() or bool EndEpoch()
//! exists.
template<typename CallbackType,
         typename OptimizerType,
         typename FunctionType,
         typename MatType>
struct HasEndEpochSignature
{
  const static bool hasBool =
      HasEndEpoch<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template EndEpochBoolForm>::value &&
      !HasEndEpoch<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template EndEpochVoidForm>::value;

  const static bool hasVoid =
      !HasEndEpoch<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template EndEpochBoolForm>::value &&
      HasEndEpoch<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template EndEpochVoidForm>::value;

  const static bool hasNone =
      !HasEndEpoch<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template EndEpochBoolForm>::value &&
      !HasEndEpoch<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template EndEpochVoidForm>::value;
};

//! Utility struct, check if either void StepTaken() or bool StepTaken() exists.
template<typename CallbackType,
         typename OptimizerType,
         typename FunctionType,
         typename MatType>
struct HasStepTakenSignature
{
  const static bool hasBool =
      HasStepTaken<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template StepTakenBoolForm>::value &&
      !HasStepTaken<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template StepTakenVoidForm>::value;

  const static bool hasVoid =
      !HasStepTaken<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template StepTakenBoolForm>::value &&
      HasStepTaken<CallbackType, TypedForms<OptimizerType,
         FunctionType, MatType>::template StepTakenVoidForm>::value;

  const static bool hasNone =
      !HasStepTaken<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template StepTakenBoolForm>::value &&
      !HasStepTaken<CallbackType, TypedForms<OptimizerType,
         FunctionType, MatType>::template StepTakenVoidForm>::value;
};

//! Utility struct, check if size_t BatchSize() const exists.
template<typename OptimizerType>
struct HasBatchSizeSignature
{
  template<typename C>
  using BatchSizeForm = size_t(C::*)(void) const;

  const static bool value =
      HasBatchSize<OptimizerType, BatchSizeForm>::value;
};

//! Utility struct, check if size_t MaxIterations() const exists.
template<typename OptimizerType>
struct HasMaxIterationsSignature
{
  template<typename C>
  using HasMaxIterationsForm = size_t(C::*)(void) const;

  const static bool value =
      HasMaxIterations<OptimizerType, HasMaxIterationsForm>::value;
};

//! Utility struct, check if size_t NumFunctions() const or
//! size_t NumFunctions() exists.
template<typename OptimizerType>
struct HasNumFunctionsSignature
{
  template<typename C>
  using NumFunctionsConstForm = size_t(C::*)(void) const;

  template<typename C>
  using NumFunctionsForm = size_t(C::*)(void);

  const static bool value =
      HasNumFunctions<OptimizerType, NumFunctionsForm>::value ||
      HasNumFunctions<OptimizerType, NumFunctionsConstForm>::value;
};

} // namespace traits
} // namespace callbacks
} // namespace ens

#endif
