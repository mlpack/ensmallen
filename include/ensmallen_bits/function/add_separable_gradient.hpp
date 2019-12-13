/**
 * @file add_separable_gradient.hpp
 * @author Ryan Curtin
 *
 * Adds a separable Gradient() function if a separable
 * EvaluateWithGradient() function exists.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_FUNCTION_ADD_DECOMPOSABLE_GRADIENT_HPP
#define ENSMALLEN_FUNCTION_ADD_DECOMPOSABLE_GRADIENT_HPP

#include "traits.hpp"

namespace ens {

/**
 * The AddSeparableGradient mixin class will add a separable Gradient()
 * method if a separable EvaluateWithGradient() function exists, or nothing
 * otherwise.
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasSeparableEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     SeparableEvaluateWithGradientForm>::value,
         bool HasSeparableGradient =
             traits::HasGradient<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     SeparableGradientForm>::value>
class AddSeparableGradient
{
 public:
  // Provide a dummy overload so the name 'Gradient' exists for this object.
  void Gradient(traits::UnconstructableType&, const size_t, const size_t);
};

/**
 * Reflect the existing Gradient().
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasSeparableEvaluateWithGradient>
class AddSeparableGradient<FunctionType, MatType, GradType,
    HasSeparableEvaluateWithGradient, true>
{
 public:
  // Reflect the existing Gradient().
  void Gradient(const MatType& coordinates,
                const size_t begin,
                GradType& gradient,
                const size_t batchSize)
  {
    static_cast<FunctionType*>(
        static_cast<Function<FunctionType,
                             MatType,
                             GradType>*>(this))->Gradient(
        coordinates, begin, gradient, batchSize);
  }
};

/**
 * If we have a separable EvaluateWithGradient() but not a separable
 * Gradient(), add a separable Evaluate() method.
 */
template<typename FunctionType, typename MatType, typename GradType>
class AddSeparableGradient<FunctionType, MatType, GradType, true, false>
{
 public:
  /**
   * Calculate the gradient and store it in the given matrix, starting at the
   * given separable function index and using the given batch size.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param begin Index of separable function to start at.
   * @param gradient Matrix to store the gradient into.
   * @param batchSize Number of separable functions to calculate for.
   */
  void Gradient(const MatType& coordinates,
                const size_t begin,
                GradType& gradient,
                const size_t batchSize)
  {
    // The returned objective value will be ignored.
    (void) static_cast<Function<FunctionType,
                                MatType,
                                GradType>*>(this)->EvaluateWithGradient(
        coordinates, begin, gradient, batchSize);
  }
};

/**
 * The AddSeparableGradientConst mixin class will add a separable const
 * Gradient() method if a separable const EvaluateWithGradient() function
 * exists, or nothing otherwise.
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasSeparableEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     SeparableEvaluateWithGradientConstForm>::value,
         bool HasSeparableGradient =
             traits::HasGradient<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     SeparableGradientConstForm>::value>
class AddSeparableGradientConst
{
 public:
  // Provide a dummy overload so the name 'Gradient' exists for this object.
  void Gradient(traits::UnconstructableType&, const size_t, const size_t) const;
};

/**
 * Reflect the existing Gradient().
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasSeparableEvaluateWithGradient>
class AddSeparableGradientConst<FunctionType, MatType, GradType,
    HasSeparableEvaluateWithGradient, true>
{
 public:
  // Reflect the existing Gradient().
  void Gradient(const MatType& coordinates,
                const size_t begin,
                GradType& gradient,
                const size_t batchSize) const
  {
    static_cast<const FunctionType*>(
        static_cast<const Function<FunctionType,
                                   MatType,
                                   GradType>*>(this))->Gradient(coordinates,
        begin, gradient, batchSize);
  }
};

/**
 * If we have a separable const EvaluateWithGradient() but not a separable
 * const Gradient(), add a separable const Gradient() method.
 */
template<typename FunctionType, typename MatType, typename GradType>
class AddSeparableGradientConst<FunctionType, MatType, GradType, true, false>
{
 public:
  /**
   * Calculate the gradient and store it in the given matrix, starting at the
   * given separable function index and using the given batch size.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param begin Index of separable function to start at.
   * @param gradient Matrix to store the gradient into.
   * @param batchSize Number of separable functions to calculate for.
   */
  void Gradient(const MatType& coordinates,
                const size_t begin,
                GradType& gradient,
                const size_t batchSize) const
  {
    // The returned objective value will be ignored.
    (void) static_cast<
        const Function<FunctionType,
                       MatType,
                       GradType>*>(this)->EvaluateWithGradient(
        coordinates, begin, gradient, batchSize);
  }
};

/**
 * The AddSeparableEvaluateStatic mixin class will add a separable static
 * Gradient() method if a separable static EvaluateWithGradient() function
 * exists, or nothing otherwise.
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasSeparableEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     SeparableEvaluateWithGradientStaticForm>::value,
         bool HasSeparableGradient =
             traits::HasGradient<FunctionType,
                 traits::TypedForms<MatType, GradType>::template
                     SeparableGradientStaticForm>::value>
class AddSeparableGradientStatic
{
 public:
  // Provide a dummy overload so the name 'Gradient' exists for this object.
  static void Gradient(traits::UnconstructableType&,
                       const size_t,
                       const size_t);
};

/**
 * Reflect the existing Gradient().
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         bool HasSeparableEvaluateWithGradient>
class AddSeparableGradientStatic<FunctionType, MatType, GradType,
    HasSeparableEvaluateWithGradient, true>
{
 public:
  // Reflect the existing Gradient().
  static void Gradient(const MatType& coordinates,
                       const size_t begin,
                       GradType& gradient,
                       const size_t batchSize)
  {
    FunctionType::Gradient(coordinates, begin, gradient, batchSize);
  }
};

/**
 * If we have a separable EvaluateWithGradient() but not a separable
 * Gradient(), add a separable Gradient() method.
 */
template<typename FunctionType, typename MatType, typename GradType>
class AddSeparableGradientStatic<FunctionType, MatType, GradType, true,
    false>
{
 public:
  /**
   * Calculate the gradient and store it in the given matrix, starting at the
   * given separable function index and using the given batch size.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param begin Index of separable function to start at.
   * @param gradient Matrix to store the gradient into.
   * @param batchSize Number of separable functions to calculate for.
   */
  static void Gradient(const MatType& coordinates,
                       const size_t begin,
                       GradType& gradient,
                       const size_t batchSize)
  {
    // The returned objective value will be ignored.
    (void) FunctionType::EvaluateWithGradient(coordinates, begin, gradient,
        batchSize);
  }
};

} // namespace ens

#endif
