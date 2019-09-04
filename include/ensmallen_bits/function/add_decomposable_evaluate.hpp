/**
 * @file add_decomposable_evaluate.hpp
 * @author Ryan Curtin
 *
 * Adds a decomposable Evaluate() function if a decomposable
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
 * The AddDecomposableEvaluate mixin class will add a decomposable Evaluate()
 * method if a decomposable EvaluateWithGradient() function exists, or nothing
 * otherwise.
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasDecomposableEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     DecomposableEvaluateWithGradientForm
             >::value,
         bool HasDecomposableEvaluate =
             traits::HasEvaluate<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                      DecomposableEvaluateForm>::value>
class AddDecomposableEvaluate
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
         bool HasDecomposableEvaluateWithGradient>
class AddDecomposableEvaluate<FunctionType, MatType, GradType,
    HasDecomposableEvaluateWithGradient, true>
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
 * If we have a decomposable EvaluateWithGradient() but not a decomposable
 * Evaluate(), add a decomposable Evaluate() method.
 */
template<typename FunctionType, typename MatType, typename GradType>
class AddDecomposableEvaluate<FunctionType, MatType, GradType, true, false>
{
 public:
  /**
   * Return the objective function for the given coordinates, starting at the
   * given decomposable function using the given batch size.
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
 * The AddDecomposableEvaluateConst mixin class will add a decomposable const
 * Evaluate() method if a decomposable const EvaluateWithGradient() function
 * exists, or nothing otherwise.
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasDecomposableEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     DecomposableEvaluateWithGradientConstForm>::value,
         bool HasDecomposableEvaluate =
             traits::HasEvaluate<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     DecomposableEvaluateConstForm>::value>
class AddDecomposableEvaluateConst
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
         bool HasDecomposableEvaluateWithGradient>
class AddDecomposableEvaluateConst<FunctionType, MatType, GradType,
    HasDecomposableEvaluateWithGradient, true>
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
 * If we have a decomposable const EvaluateWithGradient() but not a decomposable
 * const Evaluate(), add a decomposable const Evaluate() method.
 */
template<typename FunctionType, typename MatType, typename GradType>
class AddDecomposableEvaluateConst<FunctionType, MatType, GradType, true, false>
{
 public:
  /**
   * Return the objective function for the given coordinates, starting at the
   * given decomposable function using the given batch size.
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
 * The AddDecomposableEvaluateStatic mixin class will add a decomposable static
 * Evaluate() method if a decomposable static EvaluateWithGradient() function
 * exists, or nothing otherwise.
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasDecomposableEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     DecomposableEvaluateWithGradientStaticForm>::value,
         bool HasDecomposableEvaluate =
             traits::HasEvaluate<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     DecomposableEvaluateStaticForm>::value>
class AddDecomposableEvaluateStatic
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
         bool HasDecomposableEvaluateWithGradient>
class AddDecomposableEvaluateStatic<FunctionType, MatType, GradType,
    HasDecomposableEvaluateWithGradient, true>
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
 * If we have a decomposable EvaluateWithGradient() but not a decomposable
 * Evaluate(), add a decomposable Evaluate() method.
 */
template<typename FunctionType, typename MatType, typename GradType>
class AddDecomposableEvaluateStatic<FunctionType, MatType, GradType, true,
    false>
{
 public:
  /**
   * Return the objective function for the given coordinates, starting at the
   * given decomposable function using the given batch size.
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
