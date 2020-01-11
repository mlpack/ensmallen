/**
 * @file add_separable_evaluate_with_gradient.hpp
 * @author Ryan Curtin
 *
 * Adds a separable EvaluateWithGradient() function if both a separable
 * Evaluate() and a separable Gradient() function exist.
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
 * The AddSeparableEvaluateWithGradient mixin class will add a separable
 * EvaluateWithGradient() method if a separable Evaluate() method and a
 * separable Gradient() method exists, or nothing otherwise.
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         // Check if there is at least one non-const Evaluate() or Gradient().
         bool HasSeparableEvaluateGradient = traits::HasNonConstSignatures<
             FunctionType,
             traits::HasEvaluate,
             traits::TypedForms<MatType, GradType>::template
                 SeparableEvaluateForm,
             traits::TypedForms<MatType, GradType>::template
                 SeparableEvaluateConstForm,
             traits::TypedForms<MatType, GradType>::template
                 SeparableEvaluateStaticForm,
             traits::HasGradient,
             traits::TypedForms<MatType, GradType>::template
                 SeparableGradientForm,
             traits::TypedForms<MatType, GradType>::template
                 SeparableGradientConstForm,
             traits::TypedForms<MatType, GradType>::template
                 SeparableGradientStaticForm>::value,
         bool HasSeparableEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     SeparableEvaluateWithGradientForm>::value>
class AddSeparableEvaluateWithGradient
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
         bool HasSeparableEvaluateGradient>
class AddSeparableEvaluateWithGradient<FunctionType, MatType, GradType,
    HasSeparableEvaluateGradient, true>
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
 * If we have a both separable Evaluate() and a separable Gradient() but
 * not a separable EvaluateWithGradient(), add a separable
 * EvaluateWithGradient() method.
 */
template<typename FunctionType, typename MatType, typename GradType>
class AddSeparableEvaluateWithGradient<FunctionType, MatType, GradType, true,
    false>
{
 public:
  /**
   * Return both the evaluated objective function and its gradient, storing the
   * gradient in the given matrix, starting at the given separable function
   * and using the given batch size.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param begin Index of separable function to begin with.
   * @param gradient Matrix to store the gradient into.
   * @param batchSize Number of separable functions to evaluate.
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
 * The AddSeparableEvaluateWithGradientConst mixin class will add a
 * separable const EvaluateWithGradient() method if both a separable const
 * Evaluate() and a separable const Gradient() function exist, or nothing
 * otherwise.
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         // Check if there is at least one const Evaluate() or Gradient().
         bool HasSeparableEvaluateGradient = traits::HasConstSignatures<
             FunctionType,
             traits::HasEvaluate,
             traits::TypedForms<MatType, GradType>::template
                 SeparableEvaluateConstForm,
             traits::TypedForms<MatType, GradType>::template
                 SeparableEvaluateStaticForm,
             traits::HasGradient,
             traits::TypedForms<MatType, GradType>::template
                 SeparableGradientConstForm,
             traits::TypedForms<MatType, GradType>::template
                 SeparableGradientStaticForm>::value,
         bool HasSeparableEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     SeparableEvaluateWithGradientConstForm>::value>
class AddSeparableEvaluateWithGradientConst
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
         bool HasSeparableEvaluateGradient>
class AddSeparableEvaluateWithGradientConst<FunctionType, MatType, GradType,
    HasSeparableEvaluateGradient, true>
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
 * If we have both a separable const Evaluate() and a separable const
 * Gradient() but not a separable const EvaluateWithGradient(), add a
 * separable const EvaluateWithGradient() method.
 */
template<typename FunctionType, typename MatType, typename GradType>
class AddSeparableEvaluateWithGradientConst<FunctionType, MatType, GradType,
    true, false>
{
 public:
  /**
   * Return both the evaluated objective function and its gradient, storing the
   * gradient in the given matrix, starting at the given separable function
   * and using the given batch size.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param begin Index of separable function to begin with.
   * @param gradient Matrix to store the gradient into.
   * @param batchSize Number of separable functions to evaluate.
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
 * The AddSeparableEvaluateWithGradientStatic mixin class will add a
 * separable static EvaluateWithGradient() method if both a separable
 * static Evaluate() and a separable static gradient() function exist, or
 * nothing otherwise.
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasSeparableEvaluateGradient =
             traits::HasEvaluate<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     SeparableEvaluateStaticForm>::value &&
             traits::HasGradient<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     SeparableGradientStaticForm>::value,
         bool HasSeparableEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     SeparableEvaluateWithGradientStaticForm>::value>
class AddSeparableEvaluateWithGradientStatic
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
         bool HasSeparableEvaluateGradient>
class AddSeparableEvaluateWithGradientStatic<FunctionType, MatType, GradType,
    HasSeparableEvaluateGradient, true>
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
 * If we have a separable static Evaluate() and a separable static
 * Gradient() but not a separable static EvaluateWithGradient(), add a
 * separable static Gradient() method.
 */
template<typename FunctionType, typename MatType, typename GradType>
class AddSeparableEvaluateWithGradientStatic<FunctionType, MatType, GradType,
    true, false>
{
 public:
  /**
   * Return both the evaluated objective function and its gradient, storing the
   * gradient in the given matrix, starting at the given separable function
   * and using the given batch size.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param begin Index of separable function to begin with.
   * @param gradient Matrix to store the gradient into.
   * @param batchSize Number of separable functions to evaluate.
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
