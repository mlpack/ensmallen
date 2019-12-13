/**
 * @file add_separable_evaluate.hpp
 * @author Ryan Curtin
 *
 * Adds a separable Evaluate() function if a separable
 * EvaluateWithGradient() function exists.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_FUNCTION_ADD_DECOMPOSABLE_EVALUATE_HPP
#define ENSMALLEN_FUNCTION_ADD_DECOMPOSABLE_EVALUATE_HPP

#include "traits.hpp"

namespace ens {

/**
 * The AddSeparableEvaluate mixin class will add a separable Evaluate()
 * method if a separable EvaluateWithGradient() function exists, or nothing
 * otherwise.
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasSeparableEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     SeparableEvaluateWithGradientForm
             >::value,
         bool HasSeparableEvaluate =
             traits::HasEvaluate<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                      SeparableEvaluateForm>::value>
class AddSeparableEvaluate
{
 public:
  // Provide a dummy overload so the name 'Evaluate' exists for this object.
  typename MatType::elem_type Evaluate(traits::UnconstructableType&,
                                       const size_t,
                                       const size_t);
};

/**
 * Reflect the existing Evaluate().
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasSeparableEvaluateWithGradient>
class AddSeparableEvaluate<FunctionType, MatType, GradType,
    HasSeparableEvaluateWithGradient, true>
{
 public:
  // Reflect the existing Evaluate().
  typename MatType::elem_type Evaluate(const MatType& coordinates,
                                       const size_t begin,
                                       const size_t batchSize)
  {
    return static_cast<FunctionType*>(
        static_cast<Function<FunctionType,
                             MatType,
                             GradType>*>(this))->Evaluate(coordinates,
                                                          begin,
                                                          batchSize);
  }
};

/**
 * If we have a separable EvaluateWithGradient() but not a separable
 * Evaluate(), add a separable Evaluate() method.
 */
template<typename FunctionType, typename MatType, typename GradType>
class AddSeparableEvaluate<FunctionType, MatType, GradType, true, false>
{
 public:
  /**
   * Return the objective function for the given coordinates, starting at the
   * given separable function using the given batch size.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param begin Index of first function to evaluate.
   * @param batchSize Number of functions to evaluate.
   */
  double Evaluate(const MatType& coordinates,
                  const size_t begin,
                  const size_t batchSize)
  {
    GradType gradient; // This will be ignored.
    return static_cast<Function<FunctionType,
                                MatType,
                                GradType>*>(this)->EvaluateWithGradient(
        coordinates, begin, gradient, batchSize);
  }
};

/**
 * The AddSeparableEvaluateConst mixin class will add a separable const
 * Evaluate() method if a separable const EvaluateWithGradient() function
 * exists, or nothing otherwise.
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasSeparableEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     SeparableEvaluateWithGradientConstForm>::value,
         bool HasSeparableEvaluate =
             traits::HasEvaluate<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     SeparableEvaluateConstForm>::value>
class AddSeparableEvaluateConst
{
 public:
  // Provide a dummy overload so the name 'Evaluate' exists for this object.
  typename MatType::elem_type Evaluate(traits::UnconstructableType&,
                                       const size_t,
                                       const size_t) const;
};

/**
 * Reflect the existing Evaluate().
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasSeparableEvaluateWithGradient>
class AddSeparableEvaluateConst<FunctionType, MatType, GradType,
    HasSeparableEvaluateWithGradient, true>
{
 public:
  // Reflect the existing Evaluate().
  typename MatType::elem_type Evaluate(const MatType& coordinates,
                                       const size_t begin,
                                       const size_t batchSize) const
  {
    return static_cast<const FunctionType*>(
        static_cast<const Function<FunctionType,
                                   MatType,
                                   GradType>*>(this))->Evaluate(coordinates,
                                                                begin,
                                                                batchSize);
  }
};

/**
 * If we have a separable const EvaluateWithGradient() but not a separable
 * const Evaluate(), add a separable const Evaluate() method.
 */
template<typename FunctionType, typename MatType, typename GradType>
class AddSeparableEvaluateConst<FunctionType, MatType, GradType, true, false>
{
 public:
  /**
   * Return the objective function for the given coordinates, starting at the
   * given separable function using the given batch size.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param begin Index of first function to evaluate.
   * @param batchSize Number of functions to evaluate.
   */
  typename MatType::elem_type Evaluate(const MatType& coordinates,
                                       const size_t begin,
                                       const size_t batchSize) const
  {
    GradType gradient; // This will be ignored.
    return static_cast<const Function<FunctionType,
                                      MatType,
                                      GradType>*>(this)->EvaluateWithGradient(
        coordinates, begin, gradient, batchSize);
  }
};

/**
 * The AddSeparableEvaluateStatic mixin class will add a separable static
 * Evaluate() method if a separable static EvaluateWithGradient() function
 * exists, or nothing otherwise.
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasSeparableEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     SeparableEvaluateWithGradientStaticForm>::value,
         bool HasSeparableEvaluate =
             traits::HasEvaluate<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     SeparableEvaluateStaticForm>::value>
class AddSeparableEvaluateStatic
{
 public:
  // Provide a dummy overload so the name 'Evaluate' exists for this object.
  static typename MatType::elem_type Evaluate(traits::UnconstructableType&,
                                              const size_t,
                                              const size_t);
};

/**
 * Reflect the existing Evaluate().
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasSeparableEvaluateWithGradient>
class AddSeparableEvaluateStatic<FunctionType, MatType, GradType,
    HasSeparableEvaluateWithGradient, true>
{
 public:
  // Reflect the existing Evaluate().
  static typename MatType::elem_type Evaluate(const MatType& coordinates,
                                              const size_t begin,
                                              const size_t batchSize)
  {
    return FunctionType::Evaluate(coordinates, begin, batchSize);
  }
};

/**
 * If we have a separable EvaluateWithGradient() but not a separable
 * Evaluate(), add a separable Evaluate() method.
 */
template<typename FunctionType, typename MatType, typename GradType>
class AddSeparableEvaluateStatic<FunctionType, MatType, GradType, true,
    false>
{
 public:
  /**
   * Return the objective function for the given coordinates, starting at the
   * given separable function using the given batch size.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param begin Index of first function to evaluate.
   * @param batchSize Number of functions to evaluate.
   */
  static typename MatType::elem_type Evaluate(const MatType& coordinates,
                                              const size_t begin,
                                              const size_t batchSize)
  {
    GradType gradient; // This will be ignored.
    return FunctionType::EvaluateWithGradient(coordinates, begin, gradient,
        batchSize);
  }
};

} // namespace ens

#endif
