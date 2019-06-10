/**
 * @file add_gradient.hpp
 * @author Ryan Curtin
 *
 * This file defines a mixin for the Function class that will ensure that the
 * function Gradient() is avaiable if EvaluateWithGradient() is available.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_FUNCTION_ADD_GRADIENT_HPP
#define ENSMALLEN_FUNCTION_ADD_GRADIENT_HPP

#include "traits.hpp"

namespace ens {

/**
 * The AddGradient mixin class will provide a Gradient() method if the given
 * FunctionType has EvaluateWithGradient(), or nothing otherwise.
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     EvaluateWithGradientForm
             >::value,
         bool HasGradient = traits::HasGradient<FunctionType,
             traits::TypedForms<MatType, GradType>::template 
                     GradientForm>::value>
class AddGradient
{
 public:
  // Provide a dummy overload so the name 'Gradient' exists for this object.
  void Gradient(traits::UnconstructableType&) { }
};

/**
 * Reflect the existing Gradient().
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasEvaluateWithGradient>
class AddGradient<FunctionType,
                  MatType,
                  GradType,
                  HasEvaluateWithGradient,
                  true>
{
 public:
  // Reflect the existing Gradient().
  void Gradient(const MatType& coordinates, GradType& gradient)
  {
    static_cast<FunctionType*>(
        static_cast<Function<FunctionType,
                             MatType,
                             GradType>*>(this))->Gradient(coordinates,
                                                          gradient);
  }
};

/**
 * If we have EvaluateWithGradient() but no existing Gradient(), add an
 * Gradient() without a using directive to make the base Gradient() accessible.
 */
template<typename FunctionType, typename MatType, typename GradType>
class AddGradient<FunctionType, MatType, GradType, true, false>
{
 public:
  /**
   * Calculate the gradient and store it in the given matrix.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param gradient Matrix to store the gradient into.
   */
  void Gradient(const MatType& coordinates, GradType& gradient)
  {
    // The returned objective value will be ignored.
    (void) static_cast<Function<FunctionType,
                                MatType,
                                GradType>*>(this)->EvaluateWithGradient(
        coordinates, gradient);
  }
};

/**
 * The AddGradient mixin class will provide a const Gradient() method if the
 * given FunctionType has EvaluateWithGradient() const, or nothing otherwise.
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::TypedForms<MatType,
                                    GradType>::template
                     EvaluateWithGradientConstForm
             >::value,
         bool HasGradient = traits::HasGradient<FunctionType,
             traits::TypedForms<MatType, GradType>::template GradientConstForm
         >::value>
class AddGradientConst
{
 public:
  // Provide a dummy overload so the name 'Gradient' exists for this object.
  void Gradient(traits::UnconstructableType&) const { }
};

/**
 * Reflect the existing Gradient().
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasEvaluateWithGradient>
class AddGradientConst<FunctionType,
                       MatType,
                       GradType,
                       HasEvaluateWithGradient,
                       true>
{
 public:
  // Reflect the existing Gradient().
  void Gradient(const MatType& coordinates, GradType& gradient) const
  {
    static_cast<const FunctionType*>(
        static_cast<const Function<FunctionType,
                                   MatType,
                                   GradType>*>(this))->Gradient(coordinates,
                                                                gradient);
  }
};

/**
 * If we have EvaluateWithGradient() but no existing Gradient(), add a
 * Gradient() without a using directive to make the base Gradient() accessible.
 */
template<typename FunctionType, typename MatType, typename GradType>
class AddGradientConst<FunctionType, MatType, GradType, true, false>
{
 public:
  /**
   * Calculate the gradient and store it in the given matrix.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param gradient Matrix to store the gradient into.
   */
  void Gradient(const MatType& coordinates, GradType& gradient) const
  {
    // The returned objective value will be ignored.
    (void) static_cast<
        const Function<FunctionType,
                       MatType,
                       GradType>*>(this)->EvaluateWithGradient(coordinates,
                                                               gradient);
  }
};

/**
 * The AddGradient mixin class will provide a static Gradient() method if the
 * given FunctionType has static EvaluateWithGradient(), or nothing otherwise.
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::TypedForms<MatType,
                                    GradType>::template
                     EvaluateWithGradientStaticForm
             >::value,
         bool HasGradient = traits::HasGradient<FunctionType,
             traits::TypedForms<MatType, GradType>::template GradientStaticForm
         >::value>
class AddGradientStatic
{
 public:
  // Provide a dummy overload so the name 'Gradient' exists for this object.
  static void Gradient(traits::UnconstructableType&) { }
};

/**
 * Reflect the existing Gradient().
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasEvaluateWithGradient>
class AddGradientStatic<FunctionType,
                        MatType,
                        GradType,
                        HasEvaluateWithGradient,
                        true>
{
 public:
  // Reflect the existing Gradient().
  static void Gradient(const MatType& coordinates, GradType& gradient)
  {
    FunctionType::Gradient(coordinates, gradient);
  }
};

/**
 * If we have EvaluateWithGradient() but no existing Gradient(), add a
 * Gradient() without a using directive to make the base Gradient() accessible.
 */
template<typename FunctionType, typename MatType, typename GradType>
class AddGradientStatic<FunctionType, MatType, GradType, true, false>
{
 public:
  /**
   * Calculate the gradient and store it in the given matrix.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param gradient Matrix to store the gradient into.
   */
  static void Gradient(const MatType& coordinates, GradType& gradient)
  {
    // The returned objective value will be ignored.
    (void) FunctionType::EvaluateWithGradient(coordinates, gradient);
  }
};

} // namespace ens

#endif
