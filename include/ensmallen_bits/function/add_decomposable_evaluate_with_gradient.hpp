/**
 * @file add_decomposable_evaluate_with_gradient.hpp
 * @author Ryan Curtin
 *
 * Adds a decomposable EvaluateWithGradient() function if both a decomposable
 * Evaluate() and a decomposable Gradient() function exist.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_FUNCTION_ADD_DECOMPOSABLE_EVALUATE_W_GRADIENT_HPP
#define ENSMALLEN_FUNCTION_ADD_DECOMPOSABLE_EVALUATE_W_GRADIENT_HPP

#include "traits.hpp"

namespace ens {

/**
 * The AddDecomposableEvaluateWithGradient mixin class will add a decomposable
 * EvaluateWithGradient() method if a decomposable Evaluate() method and a
 * decomposable Gradient() method exists, or nothing otherwise.
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         // Check if there is at least one non-const Evaluate() or Gradient().
         bool HasDecomposableEvaluateGradient = traits::HasNonConstSignatures<
             FunctionType,
             traits::HasEvaluate,
             traits::TypedForms<MatType, GradType>::template
                 DecomposableEvaluateForm,
             traits::TypedForms<MatType, GradType>::template
                 DecomposableEvaluateConstForm,
             traits::TypedForms<MatType, GradType>::template
                 DecomposableEvaluateStaticForm,
             traits::HasGradient,
             traits::TypedForms<MatType, GradType>::template
                 DecomposableGradientForm,
             traits::TypedForms<MatType, GradType>::template
                 DecomposableGradientConstForm,
             traits::TypedForms<MatType, GradType>::template
                 DecomposableGradientStaticForm>::value,
         bool HasDecomposableEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     DecomposableEvaluateWithGradientForm>::value>
class AddDecomposableEvaluateWithGradient
{
 public:
  // Provide a dummy overload so the name 'EvaluateWithGradient' exists for this
  // object.
  typename MatType::elem_type EvaluateWithGradient(
      traits::UnconstructableType&,
      const size_t,
      const size_t);
};

/**
 * Reflect the existing EvaluateWithGradient().
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasDecomposableEvaluateGradient>
class AddDecomposableEvaluateWithGradient<FunctionType, MatType, GradType,
    HasDecomposableEvaluateGradient, true>
{
 public:
  // Reflect the existing EvaluateWithGradient().
  typename MatType::elem_type EvaluateWithGradient(const MatType& coordinates,
                                                   const size_t begin,
                                                   GradType& gradient,
                                                   const size_t batchSize)
  {
    return static_cast<FunctionType*>(
        static_cast<Function<FunctionType,
                             MatType,
                             GradType>*>(this))->EvaluateWithGradient(
        coordinates, begin, gradient, batchSize);
  }
};

/**
 * If we have a both decomposable Evaluate() and a decomposable Gradient() but
 * not a decomposable EvaluateWithGradient(), add a decomposable
 * EvaluateWithGradient() method.
 */
template<typename FunctionType, typename MatType, typename GradType>
class AddDecomposableEvaluateWithGradient<FunctionType, MatType, GradType, true,
    false>
{
 public:
  /**
   * Return both the evaluated objective function and its gradient, storing the
   * gradient in the given matrix, starting at the given decomposable function
   * and using the given batch size.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param begin Index of decomposable function to begin with.
   * @param gradient Matrix to store the gradient into.
   * @param batchSize Number of decomposable functions to evaluate.
   */
  typename MatType::elem_type EvaluateWithGradient(const MatType& coordinates,
                                                   const size_t begin,
                                                   GradType& gradient,
                                                   const size_t batchSize)
  {
    const typename MatType::elem_type objective =
        static_cast<Function<FunctionType, MatType, GradType>*>(this)->Evaluate(
        coordinates, begin, batchSize);
    static_cast<Function<FunctionType, MatType, GradType>*>(this)->Gradient(
        coordinates, begin, gradient, batchSize);
    return objective;
  }
};

/**
 * The AddDecomposableEvaluateWithGradientConst mixin class will add a
 * decomposable const EvaluateWithGradient() method if both a decomposable const
 * Evaluate() and a decomposable const Gradient() function exist, or nothing
 * otherwise.
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         // Check if there is at least one const Evaluate() or Gradient().
         bool HasDecomposableEvaluateGradient = traits::HasConstSignatures<
             FunctionType,
             traits::HasEvaluate,
             traits::TypedForms<MatType, GradType>::template
                 DecomposableEvaluateConstForm,
             traits::TypedForms<MatType, GradType>::template
                 DecomposableEvaluateStaticForm,
             traits::HasGradient,
             traits::TypedForms<MatType, GradType>::template
                 DecomposableGradientConstForm,
             traits::TypedForms<MatType, GradType>::template
                 DecomposableGradientStaticForm>::value,
         bool HasDecomposableEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     DecomposableEvaluateWithGradientConstForm>::value>
class AddDecomposableEvaluateWithGradientConst
{
 public:
  // Provide a dummy overload so the name 'EvaluateWithGradient' exists for this
  // object.
  typename MatType::elem_type EvaluateWithGradient(
      traits::UnconstructableType&,
      const size_t,
      const size_t) const;
};

/**
 * Reflect the existing EvaluateWithGradient().
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasDecomposableEvaluateGradient>
class AddDecomposableEvaluateWithGradientConst<FunctionType, MatType, GradType,
    HasDecomposableEvaluateGradient, true>
{
 public:
  // Reflect the existing Evaluate().
  typename MatType::elem_type EvaluateWithGradient(const MatType& coordinates,
                                                   const size_t begin,
                                                   GradType& gradient,
                                                   const size_t batchSize) const
  {
    return static_cast<const FunctionType*>(
        static_cast<const Function<FunctionType,
                                   MatType,
                                   GradType>*>(this))->EvaluateWithGradient(
        coordinates, begin, gradient, batchSize);
  }
};

/**
 * If we have both a decomposable const Evaluate() and a decomposable const
 * Gradient() but not a decomposable const EvaluateWithGradient(), add a
 * decomposable const EvaluateWithGradient() method.
 */
template<typename FunctionType, typename MatType, typename GradType>
class AddDecomposableEvaluateWithGradientConst<FunctionType, MatType, GradType,
    true, false>
{
 public:
  /**
   * Return both the evaluated objective function and its gradient, storing the
   * gradient in the given matrix, starting at the given decomposable function
   * and using the given batch size.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param begin Index of decomposable function to begin with.
   * @param gradient Matrix to store the gradient into.
   * @param batchSize Number of decomposable functions to evaluate.
   */
  typename MatType::elem_type EvaluateWithGradient(const MatType& coordinates,
                                                   const size_t begin,
                                                   GradType& gradient,
                                                   const size_t batchSize) const
  {
    const typename MatType::elem_type objective =
        static_cast<const Function<FunctionType,
                                   MatType,
                                   GradType>*>(this)->Evaluate(coordinates,
        begin, batchSize);
    static_cast<const Function<FunctionType,
                               MatType,
                               GradType>*>(this)->Gradient(coordinates,
        begin, gradient, batchSize);
    return objective;
  }
};

/**
 * The AddDecomposableEvaluateWithGradientStatic mixin class will add a
 * decomposable static EvaluateWithGradient() method if both a decomposable
 * static Evaluate() and a decomposable static gradient() function exist, or
 * nothing otherwise.
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasDecomposableEvaluateGradient =
             traits::HasEvaluate<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     DecomposableEvaluateStaticForm>::value &&
             traits::HasGradient<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     DecomposableGradientStaticForm>::value,
         bool HasDecomposableEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     DecomposableEvaluateWithGradientStaticForm>::value>
class AddDecomposableEvaluateWithGradientStatic
{
 public:
  // Provide a dummy overload so the name 'EvaluateWithGradient' exists for this
  // object.
  static typename MatType::elem_type EvaluateWithGradient(
      traits::UnconstructableType&,
      const size_t,
      const size_t);
};

/**
 * Reflect the existing EvaluateWithGradient().
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasDecomposableEvaluateGradient>
class AddDecomposableEvaluateWithGradientStatic<FunctionType, MatType, GradType,
    HasDecomposableEvaluateGradient, true>
{
 public:
  // Reflect the existing Evaluate().
  static typename MatType::elem_type EvaluateWithGradient(
      const MatType& coordinates,
      const size_t begin,
      GradType& gradient,
      const size_t batchSize)
  {
    return FunctionType::EvaluateWithGradient(coordinates, begin, gradient,
        batchSize);
  }
};

/**
 * If we have a decomposable static Evaluate() and a decomposable static
 * Gradient() but not a decomposable static EvaluateWithGradient(), add a
 * decomposable static Gradient() method.
 */
template<typename FunctionType, typename MatType, typename GradType>
class AddDecomposableEvaluateWithGradientStatic<FunctionType, MatType, GradType,
    true, false>
{
 public:
  /**
   * Return both the evaluated objective function and its gradient, storing the
   * gradient in the given matrix, starting at the given decomposable function
   * and using the given batch size.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param begin Index of decomposable function to begin with.
   * @param gradient Matrix to store the gradient into.
   * @param batchSize Number of decomposable functions to evaluate.
   */
  typename MatType::elem_type EvaluateWithGradient(
      const MatType& coordinates,
      const size_t begin,
      GradType& gradient,
      const size_t batchSize) const
  {
    const typename MatType::elem_type objective = FunctionType::Evaluate(
        coordinates, begin, batchSize);
    FunctionType::Gradient(coordinates, begin, gradient, batchSize);
    return objective;
  }
};

} // namespace ens

#endif
