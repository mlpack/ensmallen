/**
 * @file static_checks.hpp
 * @author Shikhar Bhardwaj
 *
 * This file contains the definitions of the method forms required by the
 * FunctionType API used by the optimizers. These method forms can be used to
 * check the compliance of a user provided FunctionType with the required
 * interface from the optimizer at compile time.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_STATIC_CHECKS_HPP
#define ENSMALLEN_STATIC_CHECKS_HPP

#include "sfinae_utility.hpp"

namespace ens {
namespace traits {

/**
 * Check if a suitable overload of Evaluate() is available.
 *
 * This is required by the FunctionType API.
 */
template<typename FunctionType, typename MatType, typename GradType>
struct CheckEvaluate
{
  const static bool value =
      HasEvaluate<FunctionType,
          TypedForms<MatType, GradType>::template EvaluateForm>::value ||
      HasEvaluate<FunctionType,
          TypedForms<MatType, GradType>::template EvaluateConstForm>::value ||
      HasEvaluate<FunctionType,
          TypedForms<MatType, GradType>::template EvaluateStaticForm>::value;
};

/**
 * Check if a suitable overload of Gradient() is available.
 *
 * This is required by the FunctionType API.
 */
template<typename FunctionType, typename MatType, typename GradType>
struct CheckGradient
{
  const static bool value =
      HasGradient<FunctionType,
          TypedForms<MatType, GradType>::template GradientForm>::value ||
      HasGradient<FunctionType,
          TypedForms<MatType, GradType>::template GradientConstForm>::value ||
      HasGradient<FunctionType,
          TypedForms<MatType, GradType>::template GradientStaticForm>::value;
};

/**
 * Check if a suitable overload of NumFunctions() is available.
 *
 * This is required by the SeparableFunctionType API.
 */
template<typename FunctionType, typename MatType, typename GradType>
struct CheckNumFunctions
{
  const static bool value =
      HasNumFunctions<FunctionType, TypedForms<MatType, GradType>::template
          NumFunctionsForm>::value ||
      HasNumFunctions<FunctionType, TypedForms<MatType, GradType>::template
          NumFunctionsConstForm>::value ||
      HasNumFunctions<FunctionType, TypedForms<MatType, GradType>::template
          NumFunctionsStaticForm>::value;
};

/**
 * Check if a suitable overload of Shuffle() is available.
 *
 * This is required by the SeparableFunctionType API.
 */
template<typename FunctionType, typename MatType, typename GradType>
struct CheckShuffle
{
  const static bool value =
      HasShuffle<FunctionType, TypedForms<MatType, GradType>::template
          ShuffleForm>::value ||
      HasShuffle<FunctionType, TypedForms<MatType, GradType>::template
          ShuffleConstForm>::value ||
      HasShuffle<FunctionType, TypedForms<MatType, GradType>::template
          ShuffleStaticForm>::value;
};

/**
 * Check if a suitable separable overload of Evaluate() is available.
 *
 * This is required by the SeparableFunctionType API.
 */
template<typename FunctionType, typename MatType, typename GradType>
struct CheckSeparableEvaluate
{
  const static bool value =
      HasEvaluate<FunctionType, TypedForms<MatType, GradType>::template
          SeparableEvaluateForm>::value ||
      HasEvaluate<FunctionType, TypedForms<MatType, GradType>::template
          SeparableEvaluateConstForm>::value ||
      HasEvaluate<FunctionType, TypedForms<MatType, GradType>::template
          SeparableEvaluateStaticForm>::value;
};

/**
 * Check if a suitable separable overload of Gradient() is available.
 *
 * This is required by the SeparableFunctionType API.
 */
template<typename FunctionType, typename MatType, typename GradType>
struct CheckSeparableGradient
{
  const static bool value =
      HasGradient<FunctionType, TypedForms<MatType, GradType>::template
          SeparableGradientForm>::value ||
      HasGradient<FunctionType, TypedForms<MatType, GradType>::template
          SeparableGradientConstForm>::value ||
      HasGradient<FunctionType, TypedForms<MatType, GradType>::template
          SeparableGradientStaticForm>::value;
};

/**
 * Check if a suitable overload of NumConstraints() is available.
 *
 * This is required by the ConstrainedFunctionType API.
 */
template<typename FunctionType, typename MatType, typename GradType>
struct CheckNumConstraints
{
  const static bool value =
      HasNumConstraints<FunctionType, TypedForms<MatType, GradType>::template
          NumConstraintsForm>::value ||
      HasNumConstraints<FunctionType, TypedForms<MatType, GradType>::template
          NumConstraintsConstForm>::value ||
      HasNumConstraints<FunctionType, TypedForms<MatType, GradType>::template
          NumConstraintsStaticForm>::value;
};

/**
 * Check if a suitable overload of EvaluateConstraint() is available.
 *
 * This is required by the ConstrainedFunctionType API.
 */
template<typename FunctionType, typename MatType, typename GradType>
struct CheckEvaluateConstraint
{
  const static bool value =
      HasEvaluateConstraint<FunctionType,
          TypedForms<MatType, GradType>::template
              EvaluateConstraintForm>::value ||
      HasEvaluateConstraint<FunctionType,
          TypedForms<MatType, GradType>::template
              EvaluateConstraintConstForm>::value ||
      HasEvaluateConstraint<FunctionType,
          TypedForms<MatType, GradType>::template
              EvaluateConstraintStaticForm>::value;
};

/**
 * Check if a suitable overload of GradientConstraint() is available.
 *
 * This is required by the ConstrainedFunctionType API.
 */
template<typename FunctionType, typename MatType, typename GradType>
struct CheckGradientConstraint
{
  const static bool value =
      HasGradientConstraint<FunctionType,
          TypedForms<MatType, GradType>::template
              GradientConstraintForm>::value ||
      HasGradientConstraint<FunctionType,
          TypedForms<MatType, GradType>::template
              GradientConstraintConstForm>::value ||
      HasGradientConstraint<FunctionType,
          TypedForms<MatType, GradType>::template
              GradientConstraintStaticForm>::value;
};

/**
 * Check if a suitable overload of Gradient() that supports sparse gradients is
 * available.
 *
 * This is required by the SparseFunctionType API.
 */
template<typename FunctionType, typename MatType, typename GradType>
struct CheckSparseGradient
{
  const static bool value =
      HasGradient<FunctionType, TypedForms<MatType, GradType>::template
          SparseGradientForm>::value ||
      HasGradient<FunctionType, TypedForms<MatType, GradType>::template
          SparseGradientConstForm>::value ||
      HasGradient<FunctionType, TypedForms<MatType, GradType>::template
          SparseGradientStaticForm>::value;
};

/**
 * Check if a suitable overload of NumFeatures() is available.
 *
 * This is required by the ResolvableFunctionType API.
 */
template<typename FunctionType, typename MatType, typename GradType>
struct CheckNumFeatures
{
  const static bool value =
      HasNumFeatures<FunctionType, TypedForms<MatType, GradType>::template
          NumFeaturesForm>::value ||
      HasNumFeatures<FunctionType, TypedForms<MatType, GradType>::template
          NumFeaturesConstForm>::value ||
      HasNumFeatures<FunctionType, TypedForms<MatType, GradType>::template
          NumFeaturesStaticForm>::value;
};

/**
 * Check if a suitable overload of PartialGradient() is available.
 *
 * This is required by the ResolvableFunctionType API.
 */
template<typename FunctionType, typename MatType, typename GradType>
struct CheckPartialGradient
{
  const static bool value =
      HasPartialGradient<FunctionType, TypedForms<MatType, GradType>::template
          PartialGradientForm>::value ||
      HasPartialGradient<FunctionType, TypedForms<MatType, GradType>::template
          PartialGradientConstForm>::value ||
      HasPartialGradient<FunctionType, TypedForms<MatType, GradType>::template
          PartialGradientStaticForm>::value;
};

/**
 * Check if a suitable overload of EvaluateWithGradient() is available.
 *
 * This is required by the FunctionType API.
 */
template<typename FunctionType, typename MatType, typename GradType>
struct CheckEvaluateWithGradient
{
  const static bool value =
      HasEvaluateWithGradient<FunctionType,
          TypedForms<MatType, GradType>::template
              EvaluateWithGradientForm>::value ||
      HasEvaluateWithGradient<FunctionType,
          TypedForms<MatType, GradType>::template
              EvaluateWithGradientConstForm>::value ||
      HasEvaluateWithGradient<FunctionType,
          TypedForms<MatType, GradType>::template
              EvaluateWithGradientStaticForm>::value;
};

/**
 * Check if a suitable separable overload of EvaluateWithGradient() is
 * available.
 *
 * This is required by the FunctionType API.
 */
template<typename FunctionType, typename MatType, typename GradType>
struct CheckSeparableEvaluateWithGradient
{
  const static bool value =
      HasEvaluateWithGradient<FunctionType,
          TypedForms<MatType, GradType>::template
              SeparableEvaluateWithGradientForm>::value ||
      HasEvaluateWithGradient<FunctionType,
          TypedForms<MatType, GradType>::template
              SeparableEvaluateWithGradientConstForm>::value ||
      HasEvaluateWithGradient<FunctionType,
          TypedForms<MatType, GradType>::template
              SeparableEvaluateWithGradientStaticForm>::value;
};

/**
 * Perform checks for the regular FunctionType API.
 */
template<typename FunctionType, typename MatType, typename GradType>
inline void CheckFunctionTypeAPI()
{
#ifndef ENS_DISABLE_TYPE_CHECKS
  static_assert(CheckEvaluate<FunctionType, MatType, GradType>::value,
      "The FunctionType does not have a correct definition of Evaluate(). "
      "Please check that the FunctionType fully satisfies the requirements of "
      "the FunctionType API; see the optimizer tutorial for details.");

  static_assert(CheckGradient<FunctionType, MatType, GradType>::value,
      "The FunctionType does not have a correct definition of Gradient(). "
      "Please check that the FunctionType fully satisfies the requirements of "
      "the FunctionType API; see the optimizer tutorial for details.");

  static_assert(
      CheckEvaluateWithGradient<FunctionType, MatType, GradType>::value,
      "The FunctionType does not have a correct definition of "
      "EvaluateWithGradient().  Please check that the FunctionType fully "
      "satisfies the requirements of the FunctionType API; see the optimizer "
      "tutorial for more details.");
#endif
}

/**
 * Perform checks for the SeparableFunctionType API.
 */
template<typename FunctionType, typename MatType, typename GradType>
inline void CheckSeparableFunctionTypeAPI()
{
#ifndef ENS_DISABLE_TYPE_CHECKS
  static_assert(CheckSeparableEvaluate<FunctionType,
                                       MatType,
                                       GradType>::value,
      "The FunctionType does not have a correct definition of a separable "
      "Evaluate() method.  Please check that the FunctionType fully satisfies"
      " the requirements of the SeparableFunctionType API; see the optimizer"
      " tutorial for more details.");

  static_assert(CheckSeparableGradient<FunctionType,
                                       MatType,
                                       GradType>::value,
      "The FunctionType does not have a correct definition of a separable "
      "Gradient() method.  Please check that the FunctionType fully satisfies"
      " the requirements of the SeparableFunctionType API; see the optimizer"
      " tutorial for more details.");

  static_assert(CheckSeparableEvaluateWithGradient<FunctionType,
                                                   MatType,
                                                   GradType>::value,
      "The FunctionType does not have a correct definition of a separable "
      "EvaluateWithGradient() method.  Please check that the FunctionType "
      "fully satisfies the requirements of the SeparableFunctionType API; "
      "see the optimizer tutorial for more details.");

  static_assert(CheckNumFunctions<FunctionType, MatType, GradType>::value,
      "The FunctionType does not have a correct definition of NumFunctions(). "
      "Please check that the FunctionType fully satisfies the requirements of "
      "the SeparableFunctionType API; see the optimizer tutorial for more "
      "details.");

  static_assert(CheckShuffle<FunctionType, MatType, GradType>::value,
      "The FunctionType does not have a correct definition of Shuffle(). "
      "Please check that the FunctionType fully satisfies the requirements of "
      "the SeparableFunctionType API; see the optimizer tutorial for more "
      "details.");
#endif
}

/**
 * Perform checks for the SparseFunctionType API.
 */
template<typename FunctionType, typename MatType, typename GradType>
inline void CheckSparseFunctionTypeAPI()
{
#ifndef ENS_DISABLE_TYPE_CHECKS
  static_assert(CheckNumFunctions<FunctionType, MatType, GradType>::value,
      "The FunctionType does not have a correct definition of NumFunctions(). "
      "Please check that the FunctionType fully satisfies the requirements of "
      "the SparseFunctionType API; see the optimizer tutorial for more "
      "details.");

  static_assert(CheckSeparableEvaluate<FunctionType,
                                          MatType,
                                          GradType>::value,
      "The FunctionType does not have a correct definition of Evaluate(). "
      "Please check that the FunctionType fully satisfies the requirements of "
      "the SparseFunctionType API; see the optimizer tutorial for more "
      "details.");

  static_assert(CheckSparseGradient<FunctionType, MatType, GradType>::value,
      "The FunctionType does not have a correct definition of a sparse "
      "Gradient() method. Please check that the FunctionType fully satisfies "
      "the requirements of the SparseFunctionType API; see the optimizer "
      "tutorial for more details.");
#endif
}

/**
 * Perform checks for the ArbitraryFunctionType API.
 */
template<typename FunctionType, typename MatType>
inline void CheckArbitraryFunctionTypeAPI()
{
#ifndef ENS_DISABLE_TYPE_CHECKS
  static_assert(CheckEvaluate<FunctionType, MatType, MatType>::value,
      "The FunctionType does not have a correct definition of Evaluate(). "
      "Please check that the FunctionType fully satisfies the requirements of "
      "the ArbitraryFunctionType API; see the optimizer tutorial for "
      "more details.");
#endif
}

template<typename FunctionType, typename... RemainingTypes>
typename std::enable_if<(sizeof...(RemainingTypes) > 1), void>::type
CheckArbitraryFunctionTypeAPI()
{
#ifndef ENS_DISABLE_TYPE_CHECKS
  constexpr size_t size = sizeof...(RemainingTypes);
  using TupleType = typename std::tuple<RemainingTypes...>;
  using MatType = typename std::tuple_element<size - 1, TupleType>::type;

  static_assert(CheckEvaluate<FunctionType, MatType, MatType>::value,
      "One of the provided FunctionType does not have a correct definition of Evaluate(). "
      "Please check that the corresponding FunctionType fully satisfies the requirements "
      "of the ArbitraryFunctionType API; see the optimizer tutorial for "
      "more details.");

  CheckArbitraryFunctionTypeAPI<RemainingTypes...>();
#endif
}

/**
 * Perform checks for the ResolvableFunctionType API.
 */
template<typename FunctionType, typename MatType, typename GradType>
inline void CheckResolvableFunctionTypeAPI()
{
#ifndef ENS_DISABLE_TYPE_CHECKS
  static_assert(CheckNumFeatures<FunctionType, MatType, GradType>::value,
      "The FunctionType does not have a correct definition of NumFeatures(). "
      "Please check that the FunctionType fully satisfies the requirements of "
      "the ResolvableFunctionType API; see the optimizer tutorial for more "
      "details.");

  static_assert(CheckEvaluate<FunctionType, MatType, GradType>::value,
      "The FunctionType does not have a correct definition of Evaluate(). "
      "Please check that the FunctionType fully satisfies the requirements of "
      "the ResolvableFunctionType API; see the optimizer tutorial for more "
      "details.");

  static_assert(CheckPartialGradient<FunctionType, MatType, GradType>::value,
      "The FunctionType does not have a correct definition of a partial "
      "Gradient() function. Please check that the FunctionType fully satisfies "
      "the requirements of the ResolvableFunctionType API; see the optimizer "
      "tutorial for more details.");
#endif
}

/**
 * Perform checks for the ConstrainedFunctionType API.
 */
template<typename FunctionType, typename MatType, typename GradType>
inline void CheckConstrainedFunctionTypeAPI()
{
#ifndef ENS_DISABLE_TYPE_CHECKS
  static_assert(CheckEvaluate<FunctionType, MatType, GradType>::value,
      "The FunctionType does not have a correct definition of Evaluate(). "
      "Please check that the FunctionType fully satisfies the requirements of "
      "the ConstrainedFunctionType API; see the optimizer tutorial for more "
      "details.");

  static_assert(CheckGradient<FunctionType, MatType, GradType>::value,
      "The FunctionType does not have a correct definition of Gradient(). "
      "Please check that the FunctionType fully satisfies the requirements of "
      "the ConstrainedFunctionType API; see the optimizer tutorial for more "
      "details.");

  static_assert(CheckNumConstraints<FunctionType, MatType, GradType>::value,
      "The FunctionType does not have a correct definition of NumConstraints()."
      " Please check that the FunctionType fully satisfies the requirements of "
      "the ConstrainedFunctionType API; see the optimizer tutorial for more "
      "details.");

  static_assert(CheckEvaluateConstraint<FunctionType, MatType, GradType>::value,
      "The FunctionType does not have a correct definition of "
      "EvaluateConstraint(). Please check that the FunctionType fully satisfies"
      " the ConstrainedFunctionType API; see the optimizer tutorial for more "
      "details.");

  static_assert(CheckGradientConstraint<FunctionType, MatType, GradType>::value,
      "The FunctionType does not have a correct definition of "
      "GradientConstraint(). Please check that the FunctionType fully satisfies"
      " the ConstrainedFunctionType API; see the optimizer tutorial for more "
      "details.");
#endif
}

/**
 * Perform checks for the ArbitrarySeparableFunctionType API.  (I
 * know, it is a long name...)
 */
template<typename FunctionType, typename MatType>
inline void CheckArbitrarySeparableFunctionTypeAPI()
{
#ifndef ENS_DISABLE_TYPE_CHECKS
  static_assert(CheckSeparableEvaluate<FunctionType,
                                          MatType,
                                          MatType>::value,
      "The FunctionType does not have a correct definition of Evaluate(). "
      "Please check that the FunctionType fully satisfies the requirements of "
      "the ArbitrarySeparableFunctionType API; see the optimizer "
      "tutorial for more details.");
#endif
}

} // namespace traits
} // namespace ens

#endif
