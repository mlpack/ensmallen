/**
 * @file add_evaluate_with_gradient.hpp
 * @author Ryan Curtin
 *
 * This file defines a mixin for the Function class that will ensure that the
 * EvaluateWithGradient() function is available if possible.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_FUNCTION_ADD_EVALUATE_WITH_GRADIENT_HPP
#define ENSMALLEN_FUNCTION_ADD_EVALUATE_WITH_GRADIENT_HPP

#include "sfinae_utility.hpp"
#include "traits.hpp"

namespace ens {

/**
 * The AddEvaluateWithGradient mixin class will provide an
 * EvaluateWithGradient() method if the given FunctionType has both Evaluate()
 * and Gradient(), or it will provide nothing otherwise.
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         // Check if there is at least one non-const Evaluate() or Gradient().
         bool HasEvaluateGradient = traits::HasNonConstSignatures<
             FunctionType,
             traits::HasEvaluate,
             traits::TypedForms<MatType, GradType>::template EvaluateForm,
             traits::TypedForms<MatType, GradType>::template EvaluateConstForm,
             traits::TypedForms<MatType, GradType>::template EvaluateStaticForm,
             traits::HasGradient,
             traits::TypedForms<MatType, GradType>::template GradientForm,
             traits::TypedForms<MatType, GradType>::template GradientConstForm,
             traits::TypedForms<MatType, GradType>::template GradientStaticForm
         >::value,
         bool HasEvaluateWithGradient = traits::HasEvaluateWithGradient<
             FunctionType,
             traits::TypedForms<MatType, GradType>::template
                 EvaluateWithGradientForm>::value>
class AddEvaluateWithGradient
{
 public:
  // Provide a dummy overload so the name 'EvaluateWithGradient' exists for this
  // object.
  typename MatType::elem_type EvaluateWithGradient(
      traits::UnconstructableType&);
};

/**
 * Reflect the existing EvaluateWithGradient().
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasEvaluateGradient>
class AddEvaluateWithGradient<FunctionType,
                              MatType,
                              GradType,
                              HasEvaluateGradient,
                              true>
{
 public:
  // Reflect the existing EvaluateWithGradient().
  typename MatType::elem_type EvaluateWithGradient(
      const MatType& coordinates, GradType& gradient)
  {
    return static_cast<FunctionType*>(
        static_cast<Function<FunctionType,
                             MatType,
                             GradType>*>(this))->EvaluateWithGradient(
        coordinates, gradient);
  }
};

/**
 * If the FunctionType has Evaluate() and Gradient(), provide
 * EvaluateWithGradient().
 */
template<typename FunctionType, typename MatType, typename GradType>
class AddEvaluateWithGradient<FunctionType, MatType, GradType, true, false>
{
 public:
  /**
   * Return both the evaluated objective function and its gradient, storing the
   * gradient in the given matrix.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param gradient Matrix to store the gradient into.
   */
  typename MatType::elem_type EvaluateWithGradient(const MatType& coordinates,
                                                   GradType& gradient)
  {
    const typename MatType::elem_type objective =
        static_cast<Function<FunctionType,
                             MatType, GradType>*>(this)->Evaluate(coordinates);
    static_cast<Function<FunctionType,
                         MatType,
                         GradType>*>(this)->Gradient(coordinates, gradient);
    return objective;
  }
};

/**
 * The AddEvaluateWithGradient mixin class will provide an
 * EvaluateWithGradient() const method if the given FunctionType has both
 * Evaluate() const and Gradient() const, or it will provide nothing otherwise.
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         // Check if there is at least one const Evaluate() or Gradient().
         bool HasEvaluateGradient = traits::HasConstSignatures<
             FunctionType,
             traits::HasEvaluate,
             traits::TypedForms<MatType, GradType>::template EvaluateConstForm,
             traits::TypedForms<MatType, GradType>::template EvaluateStaticForm,
             traits::HasGradient,
             traits::TypedForms<MatType, GradType>::template GradientConstForm,
             traits::TypedForms<MatType, GradType>::template GradientStaticForm
         >::value,
         bool HasEvaluateWithGradient = traits::HasEvaluateWithGradient<
             FunctionType,
             traits::TypedForms<
                 MatType, GradType
             >::template EvaluateWithGradientConstForm>::value>
class AddEvaluateWithGradientConst
{
 public:
  // Provide a dummy overload so the name 'EvaluateWithGradient' exists for this
  // object.
  typename MatType::elem_type EvaluateWithGradient(
      traits::UnconstructableType&) const;
};

/**
 * Reflect the existing EvaluateWithGradient().
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasEvaluateGradient>
class AddEvaluateWithGradientConst<FunctionType,
                                   MatType,
                                   GradType,
                                   HasEvaluateGradient,
                                   true>
{
 public:
  // Reflect the existing EvaluateWithGradient().
  typename MatType::elem_type EvaluateWithGradient(
      const MatType& coordinates, GradType& gradient) const
  {
    return static_cast<const FunctionType*>(
        static_cast<const Function<FunctionType,
                                   MatType,
                                   GradType>*>(this))->EvaluateWithGradient(
        coordinates, gradient);
  }
};

/**
 * If the FunctionType has Evaluate() const and Gradient() const, provide
 * EvaluateWithGradient() const.
 */
template<typename FunctionType, typename MatType, typename GradType>
class AddEvaluateWithGradientConst<FunctionType, MatType, GradType, true, false>
{
 public:
  /**
   * Return both the evaluated objective function and its gradient, storing the
   * gradient in the given matrix.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param gradient Matrix to store the gradient into.
   */
  typename MatType::elem_type EvaluateWithGradient(const MatType& coordinates,
                                                   GradType& gradient) const
  {
    const typename MatType::elem_type objective =
        static_cast<const Function<FunctionType,
                                   MatType,
                                   GradType>*>(this)->Evaluate(coordinates);
    static_cast<const Function<FunctionType,
                               MatType,
                               GradType>*>(this)->Gradient(coordinates,
                                                           gradient);
    return objective;
  }
};

/**
 * The AddEvaluateWithGradientStatic mixin class will provide a
 * static EvaluateWithGradient() method if the given FunctionType has both
 * static Evaluate() and static Gradient(), or it will provide nothing
 * otherwise.
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasEvaluateGradient =
             traits::HasEvaluate<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     EvaluateStaticForm
             >::value &&
             traits::HasGradient<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     GradientStaticForm
             >::value,
         bool HasEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::TypedForms<MatType,
                                    GradType>::template
                     EvaluateWithGradientStaticForm
             >::value>
class AddEvaluateWithGradientStatic
{
 public:
  // Provide a dummy overload so the name 'EvaluateWithGradient' exists for this
  // object.
  static typename MatType::elem_type EvaluateWithGradient(
      traits::UnconstructableType&);
};

/**
 * Reflect the existing EvaluateWithGradient().
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasEvaluateGradient>
class AddEvaluateWithGradientStatic<FunctionType,
                                    MatType,
                                    GradType,
                                    HasEvaluateGradient,
                                    true>
{
 public:
  // Reflect the existing EvaluateWithGradient().
  static typename MatType::elem_type EvaluateWithGradient(
      const MatType& coordinates, GradType& gradient)
  {
    return FunctionType::EvaluateWithGradient(coordinates, gradient);
  }
};

/**
 * If the FunctionType has static Evaluate() and static Gradient(), provide
 * static EvaluateWithGradient().
 */
template<typename FunctionType, typename MatType, typename GradType>
class AddEvaluateWithGradientStatic<FunctionType,
                                    MatType,
                                    GradType,
                                    true,
                                    false>
{
 public:
  /**
   * Return both the evaluated objective function and its gradient, storing the
   * gradient in the given matrix.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param gradient Matrix to store the gradient into.
   */
  static typename MatType::elem_type EvaluateWithGradient(
      const MatType& coordinates, GradType& gradient)
  {
    const typename MatType::elem_type objective =
        FunctionType::Evaluate(coordinates);
    FunctionType::Gradient(coordinates, gradient);
    return objective;
  }
};

} // namespace ens

#endif
