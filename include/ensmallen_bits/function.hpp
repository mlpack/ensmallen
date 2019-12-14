/**
 * @file function.hpp
 * @author Ryan Curtin
 *
 * The Function class is a wrapper class for any objective function that
 * provides any of the functions that an optimizer might use.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_FUNCTION_HPP
#define ENSMALLEN_FUNCTION_HPP

namespace ens {

template<typename FunctionType, typename MatType, typename GradType>
class Function;

} // namespace ens

#include "function/traits.hpp"
#include "function/static_checks.hpp"
#include "function/add_evaluate.hpp"
#include "function/add_gradient.hpp"
#include "function/add_evaluate_with_gradient.hpp"
#include "function/add_separable_evaluate.hpp"
#include "function/add_separable_gradient.hpp"
#include "function/add_separable_evaluate_with_gradient.hpp"

namespace ens {

/**
 * The Function class is a wrapper class for any FunctionType that will add any
 * possible derived methods.  For instance, if the given FunctionType has
 * Evaluate() and Gradient(), then Function<FunctionType> will have
 * EvaluateWithGradient().  This infrastructure allows two things:
 *
 *   1. Optimizers can expect FunctionTypes to have a wider array of functions
 *      than those FunctionTypes may actually implement.
 *
 *   2. FunctionTypes don't need to implement every single method that an
 *      optimizer might require, just those from which every method can be
 *      inferred.
 *
 * This class works by inheriting from a large set of "mixin" classes that
 * provide missing functions, if needed.  For instance, the AddGradient<> mixin
 * will provide a Gradient() method if the given FunctionType implements an
 * EvaluateWithGradient() method.
 *
 * Since all of the casting is static and each of the mixin classes is an empty
 * class, there should be no runtime overhead at all for this functionality.  In
 * addition, this class does not (to the best of my knowledge) rely on any
 * undefined behavior.
 */
template<typename FunctionType, typename MatType, typename GradType>
class Function :
    public AddSeparableEvaluateWithGradientStatic<FunctionType, MatType,
        GradType>,
    public AddSeparableEvaluateWithGradientConst<FunctionType, MatType,
        GradType>,
    public AddSeparableEvaluateWithGradient<FunctionType, MatType, GradType>,
    public AddSeparableGradientStatic<FunctionType, MatType, GradType>,
    public AddSeparableGradientConst<FunctionType, MatType, GradType>,
    public AddSeparableGradient<FunctionType, MatType, GradType>,
    public AddSeparableEvaluateStatic<FunctionType, MatType, GradType>,
    public AddSeparableEvaluateConst<FunctionType, MatType, GradType>,
    public AddSeparableEvaluate<FunctionType, MatType, GradType>,
    public AddEvaluateWithGradientStatic<FunctionType, MatType, GradType>,
    public AddEvaluateWithGradientConst<FunctionType, MatType, GradType>,
    public AddEvaluateWithGradient<FunctionType, MatType, GradType>,
    public AddGradientStatic<FunctionType, MatType, GradType>,
    public AddGradientConst<FunctionType, MatType, GradType>,
    public AddGradient<FunctionType, MatType, GradType>,
    public AddEvaluateStatic<FunctionType, MatType, GradType>,
    public AddEvaluateConst<FunctionType, MatType, GradType>,
    public AddEvaluate<FunctionType, MatType, GradType>,
    public FunctionType
{
 public:
  // All of the mixin classes either reflect existing functionality or provide
  // an unconstructable overload with the same name, so we can use using
  // declarations here to ensure that they are all accessible.  Since we don't
  // know what FunctionType has, we can't use any using declarations there.
  using AddSeparableEvaluateWithGradientStatic<
      FunctionType, MatType, GradType>::EvaluateWithGradient;
  using AddSeparableEvaluateWithGradientConst<
      FunctionType, MatType, GradType>::EvaluateWithGradient;
  using AddSeparableEvaluateWithGradient<
      FunctionType, MatType, GradType>::EvaluateWithGradient;
  using AddSeparableGradientStatic<
      FunctionType, MatType, GradType>::Gradient;
  using AddSeparableGradientConst<FunctionType, MatType, GradType>::Gradient;
  using AddSeparableGradient<FunctionType, MatType, GradType>::Gradient;
  using AddSeparableEvaluateStatic<
      FunctionType, MatType, GradType>::Evaluate;
  using AddSeparableEvaluateConst<FunctionType, MatType, GradType>::Evaluate;
  using AddSeparableEvaluate<FunctionType, MatType, GradType>::Evaluate;
  using AddEvaluateWithGradientStatic<FunctionType, MatType, GradType>::EvaluateWithGradient;
  using AddEvaluateWithGradientConst<FunctionType, MatType, GradType>::EvaluateWithGradient;
  using AddEvaluateWithGradient<FunctionType, MatType, GradType>::EvaluateWithGradient;
  using AddGradientStatic<FunctionType, MatType, GradType>::Gradient;
  using AddGradientConst<FunctionType, MatType, GradType>::Gradient;
  using AddGradient<FunctionType, MatType, GradType>::Gradient;
  using AddEvaluateStatic<FunctionType, MatType, GradType>::Evaluate;
  using AddEvaluateConst<FunctionType, MatType, GradType>::Evaluate;
  using AddEvaluate<FunctionType, MatType, GradType>::Evaluate;
};

} // namespace ens

#endif
