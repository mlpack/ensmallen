/**
 * @file add_evaluate.hpp
 * @author Ryan Curtin
 *
 * This file defines a mixin for the Function class that will ensure that the
 * function Evaluate() is avaliable if EvaluateWithGradient() is available.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_FUNCTION_ADD_EVALUATE_HPP
#define ENSMALLEN_FUNCTION_ADD_EVALUATE_HPP

#include "traits.hpp"

namespace ens {

/**
 * The AddEvaluate mixin class will provide an Evaluate() method if the given
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
         bool HasEvaluate =
             traits::HasEvaluate<FunctionType,
                  traits::TypedForms<MatType, GradType>::template
                     EvaluateForm>::value>
class AddEvaluate
{
 public:
  // Provide a dummy overload so the name 'Evaluate' exists for this object.
  typename MatType::elem_type Evaluate(traits::UnconstructableType&);
};

/**
 * Reflect the existing Evaluate().
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasEvaluateWithGradient>
class AddEvaluate<FunctionType,
                  MatType,
                  GradType,
                  HasEvaluateWithGradient,
                  true>
{
 public:
  // Reflect the existing Evaluate().
  typename MatType::elem_type Evaluate(const MatType& coordinates)
  {
    return static_cast<FunctionType*>(
        static_cast<Function<FunctionType,
                             MatType, GradType>*>(this))->Evaluate(coordinates);
  }
};

/**
 * If we have EvaluateWithGradient() but no existing Evaluate(), add an
 * Evaluate() method.
 */
template<typename FunctionType, typename MatType, typename GradType>
class AddEvaluate<FunctionType, MatType, GradType, true, false>
{
 public:
  /**
   * Return the objective function for the given coordinates.
   *
   * @param coordinates Coordinates to evaluate the function at.
   */
  typename MatType::elem_type Evaluate(const MatType& coordinates)
  {
    GradType gradient; // This will be ignored.
    return static_cast<Function<FunctionType,
                                MatType,
                                GradType>*>(this)->EvaluateWithGradient(
        coordinates, gradient);
  }
};

/**
 * The AddEvaluateConst mixin class will provide a const Evaluate() method if
 * the given FunctionType has EvaluateWithGradient() const, or nothing
 * otherwise.
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
         bool HasEvaluate =
             traits::HasEvaluate<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     EvaluateConstForm
             >::value>
class AddEvaluateConst
{
 public:
  // Provide a dummy overload so the name 'Evaluate' exists for this object.
  typename MatType::elem_type Evaluate(traits::UnconstructableType&) const;
};

/**
 * Reflect the existing Evaluate().
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasEvaluateWithGradient>
class AddEvaluateConst<FunctionType,
                       MatType,
                       GradType,
                       HasEvaluateWithGradient,
                       true>
{
 public:
  // Reflect the existing Evaluate().
  typename MatType::elem_type Evaluate(const MatType& coordinates) const
  {
    return static_cast<const FunctionType*>(
        static_cast<const Function<FunctionType,
                                   MatType,
                                   GradType>*>(this))->Evaluate(coordinates);
  }
};

/**
 * If we have EvaluateWithGradient() but no existing Evaluate(), add an
 * Evaluate() without a using directive to make the base Evaluate() accessible.
 */
template<typename FunctionType, typename MatType, typename GradType>
class AddEvaluateConst<FunctionType, MatType, GradType, true, false>
{
 public:
  /**
   * Return the objective function for the given coordinates.
   *
   * @param coordinates Coordinates to evaluate the function at.
   */
  typename MatType::elem_type Evaluate(const MatType& coordinates) const
  {
    GradType gradient; // This will be ignored.
    return static_cast<
        const Function<FunctionType,
                       MatType,
                       GradType>*>(this)->EvaluateWithGradient(coordinates,
                                                               gradient);
  }
};

/**
 * The AddEvaluateStatic mixin class will provide a static Evaluate() method if
 * the given FunctionType has EvaluateWithGradient() static, or nothing
 * otherwise.
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
         bool HasEvaluate =
             traits::HasEvaluate<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                 EvaluateStaticForm
             >::value>
class AddEvaluateStatic
{
 public:
  // Provide a dummy overload so the name 'Evaluate' exists for this object.
  static typename MatType::elem_type Evaluate(traits::UnconstructableType&);
};

/**
 * Reflect the existing Evaluate().
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasEvaluateWithGradient>
class AddEvaluateStatic<FunctionType,
                        MatType,
                        GradType,
                        HasEvaluateWithGradient,
                        true>
{
 public:
  // Reflect the existing Evaluate().
  static typename MatType::elem_type Evaluate(
      const MatType& coordinates)
  {
    return FunctionType::Evaluate(coordinates);
  }
};

/**
 * If we have EvaluateWithGradient() but no existing Evaluate(), add an
 * Evaluate() without a using directive to make the base Evaluate() accessible.
 */
template<typename FunctionType, typename MatType, typename GradType>
class AddEvaluateStatic<FunctionType, MatType, GradType, true, false>
{
 public:
  /**
   * Return the objective function for the given coordinates.
   *
   * @param coordinates Coordinates to evaluate the function at.
   */
  static typename MatType::elem_type Evaluate(const MatType& coordinates)
  {
    GradType gradient; // This will be ignored.
    return FunctionType::EvaluateWithGradient(coordinates, gradient);
  }
};

} // namespace ens

#endif
