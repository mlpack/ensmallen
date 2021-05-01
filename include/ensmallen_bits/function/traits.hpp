/**
 * @file traits.hpp
 * @author Ryan Curtin
 *
 * This file provides metaprogramming utilities for detecting certain members of
 * FunctionType classes.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_FUNCTION_TRAITS_HPP
#define ENSMALLEN_FUNCTION_TRAITS_HPP

#include "sfinae_utility.hpp"
#include "arma_traits.hpp"

namespace ens {
namespace traits {

//! Detect an Evaluate() method.
ENS_HAS_EXACT_METHOD_FORM(Evaluate, HasEvaluate)
//! Detect a Gradient() method.
ENS_HAS_EXACT_METHOD_FORM(Gradient, HasGradient)
//! Detect an EvaluateWithGradient() method.
ENS_HAS_EXACT_METHOD_FORM(EvaluateWithGradient, HasEvaluateWithGradient)
//! Detect a NumFunctions() method.
ENS_HAS_EXACT_METHOD_FORM(NumFunctions, HasNumFunctions)
//! Detect a Shuffle() method.
ENS_HAS_EXACT_METHOD_FORM(Shuffle, HasShuffle)
//! Detect a NumConstraints() method.
ENS_HAS_EXACT_METHOD_FORM(NumConstraints, HasNumConstraints)
//! Detect an EvaluateConstraint() method.
ENS_HAS_EXACT_METHOD_FORM(EvaluateConstraint, HasEvaluateConstraint)
//! Detect a GradientConstraint() method.
ENS_HAS_EXACT_METHOD_FORM(GradientConstraint, HasGradientConstraint)
//! Detect a NumFeatures() method.
ENS_HAS_EXACT_METHOD_FORM(NumFeatures, HasNumFeatures)
//! Detect a PartialGradient() method.
ENS_HAS_EXACT_METHOD_FORM(PartialGradient, HasPartialGradient)
//! Detect an MaxIterations() method.
ENS_HAS_EXACT_METHOD_FORM(MaxIterations, HasMaxIterations)
//! Detect an ResetPolicy() method.
ENS_HAS_EXACT_METHOD_FORM(ResetPolicy, HasResetPolicy)
//! Detect an BatchSize() method.
ENS_HAS_EXACT_METHOD_FORM(BatchSize, HasBatchSize)
//! Detect an StepSize() method.
ENS_HAS_EXACT_METHOD_FORM(StepSize, HasStepSize)

template<typename MatType, typename GradType>
struct TypedForms
{
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;
  typedef typename MatTypeTraits<GradType>::BaseMatType BaseGradType;

  //! This is the form of a non-const Evaluate() method.
  template<typename FunctionType>
  using EvaluateForm =
      typename BaseMatType::elem_type(FunctionType::*)(const BaseMatType&);

  //! This is the form of a const Evaluate() method.
  template<typename FunctionType>
  using EvaluateConstForm = typename BaseMatType::elem_type(FunctionType::*)(
      const BaseMatType&) const;

  //! This is the form of a static Evaluate() method.
  template<typename FunctionType>
  using EvaluateStaticForm = typename BaseMatType::elem_type(*)(
      const BaseMatType&);

  //! This is the form of a non-const Gradient() method.
  template<typename FunctionType>
  using GradientForm = void(FunctionType::*)(const BaseMatType&, BaseGradType&);

  //! This is the form of a const Gradient() method.
  template<typename FunctionType>
  using GradientConstForm =
      void(FunctionType::*)(const BaseMatType&, BaseGradType&) const;

  //! This is the form of a static Gradient() method.
  template<typename FunctionType>
  using GradientStaticForm = void(*)(const BaseMatType&, BaseGradType&);

  //! This is the form of a non-const EvaluateWithGradient() method.
  template<typename FunctionType>
  using EvaluateWithGradientForm =
      typename BaseMatType::elem_type(FunctionType::*)(const BaseMatType&,
                                                       BaseGradType&);

  //! This is the form of a const EvaluateWithGradient() method.
  template<typename FunctionType>
  using EvaluateWithGradientConstForm =
      typename BaseMatType::elem_type(FunctionType::*)(const BaseMatType&,
                                                       BaseGradType&) const;

  //! This is the form of a static EvaluateWithGradient() method.
  template<typename FunctionType>
  using EvaluateWithGradientStaticForm = typename BaseMatType::elem_type(*)(
      const BaseMatType&, BaseGradType&);

  //! This is the form of a non-const NumFunctions() method.
  template <typename FunctionType>
  using NumFunctionsForm = size_t(FunctionType::*)();

  //! This is the form of a const NumFunctions() method.
  template <typename FunctionType>
  using NumFunctionsConstForm = size_t(FunctionType::*)() const;

  //! This is the form of a static NumFunctions() method.
  template<typename FunctionType>
  using NumFunctionsStaticForm = size_t(*)();

  //! This is the form of a non-const Shuffle() method.
  template<typename FunctionType>
  using ShuffleForm = void(FunctionType::*)();

  //! This is the form of a const Shuffle() method.
  template<typename FunctionType>
  using ShuffleConstForm = void(FunctionType::*)() const;

  //! This is the form of a static Shuffle() method.
  template<typename FunctionType>
  using ShuffleStaticForm = void(*)();

  //! This is the form of a separable Evaluate() method.
  template<typename FunctionType>
  using SeparableEvaluateForm =
      typename BaseMatType::elem_type(FunctionType::*)(const BaseMatType&,
                                                       const size_t,
                                                       const size_t);

  //! This is the form of a separable const Evaluate() method.
  template<typename FunctionType>
  using SeparableEvaluateConstForm =
      typename BaseMatType::elem_type(FunctionType::*)(const BaseMatType&,
                                                       const size_t,
                                                       const size_t) const;

  //! This is the form of a separable static Evaluate() method.
  template<typename FunctionType>
  using SeparableEvaluateStaticForm = typename BaseMatType::elem_type(*)(
        const BaseMatType&, const size_t, const size_t);

  //! This is the form of a separable non-const Gradient() method.
  template<typename FunctionType>
  using SeparableGradientForm = void(FunctionType::*)(
      const BaseMatType&, const size_t, BaseGradType&, const size_t);

  //! This the form of a separable const Gradient() method.
  template<typename FunctionType>
  using SeparableGradientConstForm = void(FunctionType::*)(
      const BaseMatType&, const size_t, BaseGradType&, const size_t) const;

  //! This is the form of a separable static Gradient() method.
  template<typename FunctionType>
  using SeparableGradientStaticForm = void(*)(
      const BaseMatType&, const size_t, BaseGradType&, const size_t);

  //! This is the form of a separable non-const EvaluateWithGradient()
  //! method.
  template<typename FunctionType>
  using SeparableEvaluateWithGradientForm =
      typename BaseMatType::elem_type(FunctionType::*)(const BaseMatType&,
                                                       const size_t,
                                                       BaseGradType&,
                                                       const size_t);

  //! This is the form of a separable const EvaluateWithGradient() method.
  template<typename FunctionType>
  using SeparableEvaluateWithGradientConstForm =
      typename BaseMatType::elem_type(FunctionType::*)(const BaseMatType&,
                                                       const size_t,
                                                       BaseGradType&,
                                                       const size_t) const;

  //! This is the form of a separable static EvaluateWithGradient() method.
  template<typename FunctionType>
  using SeparableEvaluateWithGradientStaticForm =
      typename BaseMatType::elem_type(*)(const BaseMatType&,
                                         const size_t,
                                         BaseGradType&,
                                         const size_t);

  //! This is the form of a non-const NumConstraints() method.
  template<typename FunctionType>
  using NumConstraintsForm = size_t(FunctionType::*)();

  //! This is the form of a const NumConstraints() method.
  template<typename FunctionType>
  using NumConstraintsConstForm = size_t(FunctionType::*)() const;

  //! This is the form of a static NumConstraints() method.
  template<typename FunctionType>
  using NumConstraintsStaticForm = size_t(*)();

  //! This is the form of a non-const EvaluateConstraint() method.
  template <typename FunctionType>
  using EvaluateConstraintForm =
      typename BaseMatType::elem_type(FunctionType::*)(const size_t,
                                                       const BaseMatType&);

  //! This is the form of a const EvaluateConstraint() method.
  template<typename FunctionType>
  using EvaluateConstraintConstForm =
      typename BaseMatType::elem_type(FunctionType::*)(const size_t,
                                                       const BaseMatType&)
          const;

  //! This is the form of a static EvaluateConstraint() method.
  template<typename FunctionType>
  using EvaluateConstraintStaticForm = typename BaseMatType::elem_type(*)(
      const size_t, const BaseMatType&);

  //! This is the form of a non-const GradientConstraint() method.
  template <typename FunctionType>
  using GradientConstraintForm = void(FunctionType::*)(
      const size_t, const BaseMatType&, BaseGradType&);

  //! This is the form of a const GradientConstraint() method.
  template<typename FunctionType>
  using GradientConstraintConstForm = void(FunctionType::*)(
      const size_t, const BaseMatType&, BaseGradType&) const;

  //! This is the form of a static GradientConstraint() method.
  template<typename Class, typename... Ts>
  using GradientConstraintStaticForm = void(*)(
      const size_t, const BaseMatType&, BaseGradType&);

  //! This is the form of a non-const sparse Gradient() method.
  //! This check isn't particularly useful---the user needs to specify a sparse
  //! gradient type...
  template<typename FunctionType>
  using SparseGradientForm = void(FunctionType::*)(
      const BaseMatType&, const size_t, BaseGradType&, const size_t);

  //! This is the form of a const sparse Gradient() method.
  //! This check isn't particularly useful---the user needs to specify a sparse
  //! gradient type...
  template<typename FunctionType>
  using SparseGradientConstForm = void(FunctionType::*)(
      const BaseMatType&, const size_t, BaseGradType&, const size_t) const;

  //! This is the form of a static sparse Gradient() method.
  //! This check isn't particularly useful---the user needs to specify a sparse
  //! gradient type...
  template<typename FunctionType>
  using SparseGradientStaticForm = void(*)(
      const BaseMatType&, const size_t, BaseGradType&, const size_t);

  //! This is the form of a non-const NumFeatures() method.
  template<typename FunctionType>
  using NumFeaturesForm = size_t(FunctionType::*)();

  //! This is the form of a const NumFeatures() method.
  template<typename FunctionType>
  using NumFeaturesConstForm = size_t(FunctionType::*)() const;

  //! This is the form of a static NumFeatures() method.
  template<typename FunctionType>
  using NumFeaturesStaticForm = size_t(*)();

  //! This is the form of a non-const PartialGradient() method.
  template<typename FunctionType>
  using PartialGradientForm = void(FunctionType::*)(
      const BaseMatType&, const size_t, BaseGradType&);

  //! This is the form of a const PartialGradient() method.
  template<typename FunctionType>
  using PartialGradientConstForm = void(FunctionType::*)(
      const BaseMatType&, const size_t, BaseGradType&) const;

  //! This is the form of a static PartialGradient() method.
  template<typename FunctionType>
  using PartialGradientStaticForm = void(*)(
      const BaseMatType&, const size_t, BaseGradType&);

  //! This is a utility struct that will match any non-const form.
  template<typename FunctionType, typename... Ts>
  using OtherForm = typename BaseMatType::elem_type(FunctionType::*)(Ts...);

  //! This is a utility struct that will match any const form.
  template<typename FunctionType, typename... Ts>
  using OtherConstForm = typename BaseMatType::elem_type(FunctionType::*)(Ts...)
      const;

  //! This is a utility struct that will match any static form.
  template<typename FunctionType, typename... Ts>
  using OtherStaticForm = typename BaseMatType::elem_type(*)(Ts...);
};

/**
 * This is a utility type used to provide unusable overloads from each of the
 * mixin classes.  If you are seeing an error mentioning this class, the most
 * likely issue is that you have not implemented the right methods for your
 * FunctionType class.
 */
struct UnconstructableType
{
 private:
  UnconstructableType() { }
};

/**
 * Utility struct: sometimes we want to know if we have two functions available,
 * and that at least one of them is non-const and non-static.  If the
 * corresponding checkers (from ENS_HAS_METHOD_FORM()) are given as CheckerA and
 * CheckerB, and the corresponding non-const, const, and static function
 * signatures are given as SignatureA, ConstSignatureA, StaticSignatureA,
 * SignatureB, ConstSignatureB, and StaticSignatureB, then 'value' will be true
 * if methods with the correct names exist in the given ClassType and at least
 * one of those two methods is non-const and non-static.
 */
template<typename ClassType,
         template<typename, template<typename...> class, size_t> class CheckerA,
         template<typename...> class SignatureA,
         template<typename...> class ConstSignatureA,
         template<typename...> class StaticSignatureA,
         template<typename, template<typename...> class, size_t> class CheckerB,
         template<typename...> class SignatureB,
         template<typename...> class ConstSignatureB,
         template<typename...> class StaticSignatureB>
struct HasNonConstSignatures
{
  // Check if any const or static version of method A exists.
  const static bool HasAnyFormA =
      CheckerA<ClassType, SignatureA, 0>::value ||
      CheckerA<ClassType, ConstSignatureA, 0>::value ||
      CheckerA<ClassType, StaticSignatureA, 0>::value;
  // Check if any const or static versino of method B exists.
  const static bool HasAnyFormB =
      CheckerB<ClassType, SignatureB, 0>::value ||
      CheckerB<ClassType, ConstSignatureB, 0>::value ||
      CheckerB<ClassType, StaticSignatureB, 0>::value;

  // Make sure at least one const version exists.
  const static bool HasEitherNonConstForm =
      CheckerA<ClassType, SignatureA, 0>::value ||
      CheckerB<ClassType, SignatureB, 0>::value;

  const static bool value = HasEitherNonConstForm && HasAnyFormA && HasAnyFormB;
};

/**
 * Utility struct: sometimes we want to know if we have two functions available,
 * and that at least one of them is const and both of them are not non-const and
 * non-static.  If the corresponding checkers (from ENS_HAS_METHOD_FORM()) are
 * given as CheckerA and CheckerB, and the corresponding const and static
 * function signatures are given as ConstSignatureA, StaticSignatureA,
 * ConstSignatureB, and StaticSignatureB, then 'value' will be true if methods
 * with the correct names exist in the given ClassType and at least one of those
 * two methods is const, and neither method is non-const and non-static.
 */
template<typename ClassType,
         template<typename, template<typename...> class, size_t> class CheckerA,
         template<typename...> class ConstSignatureA,
         template<typename...> class StaticSignatureA,
         template<typename, template<typename...> class, size_t> class CheckerB,
         template<typename...> class ConstSignatureB,
         template<typename...> class StaticSignatureB>
struct HasConstSignatures
{
  // Check if any const or static version of method A exists.
  const static bool HasAnyFormA =
      CheckerA<ClassType, ConstSignatureA, 0>::value ||
      CheckerA<ClassType, StaticSignatureA, 0>::value;
  // Check if any const or static version of method B exists.
  const static bool HasAnyFormB =
      CheckerB<ClassType, ConstSignatureB, 0>::value ||
      CheckerB<ClassType, StaticSignatureB, 0>::value;

  // Make sure at least one const version exists.
  const static bool HasEitherConstForm =
      CheckerA<ClassType, ConstSignatureA, 0>::value ||
      CheckerB<ClassType, ConstSignatureB, 0>::value;

  const static bool value = HasEitherConstForm && HasAnyFormA && HasAnyFormB;
};

//! Utility struct, check if size_t BatchSize() const or size_t BatchSize()
//! exists.
template<typename OptimizerType>
struct HasBatchSizeSignature
{
  template<typename C>
  using BatchSizeConstForm = size_t(C::*)(void) const;

  template<typename C>
  using BatchSizeForm = size_t(C::*)(void);

  const static bool value =
      HasBatchSize<OptimizerType, BatchSizeForm>::value ||
      HasBatchSize<OptimizerType, BatchSizeConstForm>::value;
};

//! Utility struct, check if size_t StepSize() const or size_t StepSize()
//! exists.
template<typename OptimizerType>
struct HasStepSizeSignature
{
  template<typename C>
  using StepSizeConstForm = double(C::*)(void) const;

  template<typename C>
  using StepSizeForm = double(C::*)(void);

  const static bool value =
      HasStepSize<OptimizerType, StepSizeForm>::value ||
      HasStepSize<OptimizerType, StepSizeConstForm>::value;
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

//! Utility struct, check if bool ResetPolicy() exists.
template<typename OptimizerType>
struct HasResetPolicySignature
{
  template<typename C>
  using HasResetPolicyForm = bool&(C::*)(void);

  const static bool value =
      HasResetPolicy<OptimizerType, HasResetPolicyForm>::value;
};

} // namespace traits
} // namespace ens

#endif
