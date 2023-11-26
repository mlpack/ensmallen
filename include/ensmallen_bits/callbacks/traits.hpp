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
//! Detect an GenerationalStepTaken() method.
ENS_HAS_EXACT_METHOD_FORM(GenerationalStepTaken, HasGenerationalStepTaken)

template<typename OptimizerType,
         typename FunctionType,
         typename MatType,
         typename GradType = MatType>
struct TypedForms
{
  //! This is the form of a bool Evaluate() callback method.
  template<typename CallbackType>
  using EvaluateBoolForm =
      bool(CallbackType::*)(OptimizerType&,
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
      bool(CallbackType::*)(OptimizerType&,
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
      bool(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            const MatType&,
                            const MatType&);

  //! This is the form of a bool Gradient() callback method where the gradient
  //! is modifiable.
  template<typename CallbackType>
  using GradientBoolModifiableForm =
      bool(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            const MatType&,
                            MatType&);

  //! This is the form of a void Gradient() callback method.
  template<typename CallbackType>
  using GradientVoidForm =
      void(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            const MatType&,
                            const MatType&);

  //! This is the form of a void Gradient() callback method where the gradient
  //! is modifiable.
  template<typename CallbackType>
  using GradientVoidModifiableForm =
      void(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            const MatType&,
                            MatType&);

  //! This is the form of a bool GradientConstraint() callback method.
  template<typename CallbackType>
  using GradientConstraintBoolForm =
      bool(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            const MatType&,
                            const size_t,
                            const MatType&);

  //! This is the form of a bool GradientConstraint() callback method where the
  //! gradient is modifiable.
  template<typename CallbackType>
  using GradientConstraintBoolModifiableForm =
      bool(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            const MatType&,
                            const size_t,
                            MatType&);

  //! This is the form of a void GradientConstraint() callback method.
  template<typename CallbackType>
  using GradientConstraintVoidForm =
      void(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            const MatType&,
                            const size_t,
                            const MatType&);

  //! This is the form of a void GradientConstraint() callback method where the
  //! gradient is modifiable.
  template<typename CallbackType>
  using GradientConstraintVoidModifiableForm =
      void(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            const MatType&,
                            const size_t,
                            MatType&);

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
  constexpr static bool value =
      HasBeginOptimization<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template BeginOptimizationBoolForm>::value ||
      HasBeginOptimization<CallbackType, TypedForms<OptimizerType,
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
  constexpr static bool hasBool =
      HasEvaluate<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template EvaluateBoolForm>::value &&
      !HasEvaluate<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template EvaluateVoidForm>::value;

  constexpr static bool hasVoid =
      !HasEvaluate<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template EvaluateBoolForm>::value &&
      HasEvaluate<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template EvaluateVoidForm>::value;

  constexpr static bool hasNone =
      !HasEvaluate<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template EvaluateBoolForm>::value &&
      !HasEvaluate<CallbackType, TypedForms<OptimizerType,
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
  constexpr static bool hasBool =
      HasEvaluateConstraint<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template EvaluateConstraintBoolForm>::value &&
      !HasEvaluateConstraint<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template EvaluateConstraintVoidForm>::value;

  constexpr static bool hasVoid =
      !HasEvaluateConstraint<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template EvaluateConstraintBoolForm>::value &&
      HasEvaluateConstraint<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template EvaluateConstraintVoidForm>::value;

  constexpr static bool hasNone =
      !HasEvaluateConstraint<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template EvaluateConstraintBoolForm>::value &&
      !HasEvaluateConstraint<CallbackType, TypedForms<OptimizerType,
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
  constexpr static bool hasBool =
      (HasGradient<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType, Gradient>::template GradientBoolForm>::value ||
      HasGradient<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType, Gradient>::template
          GradientBoolModifiableForm>::value) &&
      (!HasGradient<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType, Gradient>::template GradientVoidForm>::value ||
      !HasGradient<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType, Gradient>::template
          GradientVoidModifiableForm>::value);

  constexpr static bool hasVoid =
      (!HasGradient<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType, Gradient>::template GradientBoolForm>::value ||
      !HasGradient<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType, Gradient>::template
          GradientBoolModifiableForm>::value) &&
      (HasGradient<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType, Gradient>::template GradientVoidForm>::value ||
      HasGradient<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType, Gradient>::template
          GradientVoidModifiableForm>::value);

  constexpr static bool hasNone =
      !HasGradient<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType, Gradient>::template GradientBoolForm>::value &&
      !HasGradient<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType, Gradient>::template
          GradientBoolModifiableForm>::value &&
      !HasGradient<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType, Gradient>::template GradientVoidForm>::value &&
      !HasGradient<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType, Gradient>::template
          GradientVoidModifiableForm>::value;
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
  constexpr static bool hasBool =
      HasGradientConstraint<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType, Gradient>::template
          GradientConstraintBoolForm>::value &&
      !HasGradientConstraint<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType, Gradient>::template
          GradientConstraintVoidForm>::value;

  constexpr static bool hasVoid =
      !HasGradientConstraint<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType, Gradient>::template
          GradientConstraintBoolForm>::value &&
      HasGradientConstraint<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType, Gradient>::template
          GradientConstraintVoidForm>::value;

  constexpr static bool hasNone =
      !HasGradientConstraint<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType, Gradient>::template
          GradientConstraintBoolForm>::value &&
      !HasGradientConstraint<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType, Gradient>::template
          GradientConstraintVoidForm>::value;
};

//! Utility struct, check if either void EndOptimization() or
//! bool EndOptimization() exists.
template<typename CallbackType,
         typename OptimizerType,
         typename FunctionType,
         typename MatType>
struct HasEndOptimizationSignature
{
  constexpr static bool value =
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
  constexpr static bool hasBool =
      HasBeginEpoch<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template BeginEpochBoolForm>::value &&
      !HasBeginEpoch<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template BeginEpochVoidForm>::value;

  constexpr static bool hasVoid =
      !HasBeginEpoch<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template BeginEpochBoolForm>::value &&
      HasBeginEpoch<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template BeginEpochVoidForm>::value;

  constexpr static bool hasNone =
      !HasBeginEpoch<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template BeginEpochBoolForm>::value &&
      !HasBeginEpoch<CallbackType, TypedForms<OptimizerType,
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
  constexpr static bool hasBool =
      HasEndEpoch<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template EndEpochBoolForm>::value &&
      !HasEndEpoch<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template EndEpochVoidForm>::value;

  constexpr static bool hasVoid =
      !HasEndEpoch<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template EndEpochBoolForm>::value &&
      HasEndEpoch<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template EndEpochVoidForm>::value;

  constexpr static bool hasNone =
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
  constexpr static bool hasBool =
      HasStepTaken<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template StepTakenBoolForm>::value &&
      !HasStepTaken<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template StepTakenVoidForm>::value;

  constexpr static bool hasVoid =
      !HasStepTaken<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template StepTakenBoolForm>::value &&
      HasStepTaken<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template StepTakenVoidForm>::value;

  constexpr static bool hasNone =
      !HasStepTaken<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template StepTakenBoolForm>::value &&
      !HasStepTaken<CallbackType, TypedForms<OptimizerType,
          FunctionType, MatType>::template StepTakenVoidForm>::value;
};

//! A utility struct for Typed Forms required in
//! callbacks for MultiObjective Optimizers.
template<typename OptimizerType,
         typename FunctionType,
         typename MatType,
         typename ObjectivesVecType,
         typename IndicesType,
         typename GradType = MatType>
struct MOOTypedForms
{
  //! This is the form of a bool GenerationalStepTaken() for MOO callback method.
  template<typename CallbackType>
  using GenerationalStepTakenBoolForm =
      bool(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            const MatType&,
                            const ObjectivesVecType&,
                            const IndicesType&);

  //! This is the form of a void StepTaken() for MOO callback method.
  template<typename CallbackType>
  using GenerationalStepTakenVoidForm =
      void(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            const MatType&,
                            const ObjectivesVecType&,
                            const IndicesType&);
};

//! Utility struct, check if either void StepTaken() or bool StepTaken() exists.
//! Specialization for Multiobjective case.
template<typename CallbackType,
         typename OptimizerType,
         typename FunctionType,
         typename ObjectivesVecType,
         typename IndicesType,
         typename MatType>
 struct HasGenerationalStepTakenSignature
{
  constexpr static bool hasBool =
    HasGenerationalStepTaken<CallbackType, MOOTypedForms<OptimizerType,
        FunctionType, MatType, ObjectivesVecType, IndicesType>::
        template GenerationalStepTakenBoolForm>::value &&
    !HasGenerationalStepTaken<CallbackType, MOOTypedForms<OptimizerType,
        FunctionType, MatType, ObjectivesVecType, IndicesType>::
        template GenerationalStepTakenVoidForm>::value;

  constexpr static bool hasVoid =
    !HasGenerationalStepTaken<CallbackType, MOOTypedForms<OptimizerType,
        FunctionType, MatType, ObjectivesVecType, IndicesType>::
        template GenerationalStepTakenBoolForm>::value &&
    HasGenerationalStepTaken<CallbackType, MOOTypedForms<OptimizerType,
        FunctionType, MatType, ObjectivesVecType, IndicesType>::
        template GenerationalStepTakenVoidForm>::value;

  constexpr static bool hasNone =
    !HasGenerationalStepTaken<CallbackType, MOOTypedForms<OptimizerType,
        FunctionType, MatType, ObjectivesVecType, IndicesType>::
        template GenerationalStepTakenBoolForm>::value &&
    !HasGenerationalStepTaken<CallbackType, MOOTypedForms<OptimizerType,
        FunctionType, MatType, ObjectivesVecType, IndicesType>::
        template GenerationalStepTakenVoidForm>::value;
};
} // namespace traits
} // namespace callbacks
} // namespace ens

#endif
