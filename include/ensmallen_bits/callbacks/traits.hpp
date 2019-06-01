/**
 * @file traits.hpp
 * @author Marcus Edel
 *
 * This file provides metaprogramming utilities for detecting certain members of
 * CallbackType classes.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CALLBACKS_TRAITS_HPP
#define ENSMALLEN_CALLBACKS_TRAITS_HPP

#include <ensmallen_bits/function/sfinae_utility.hpp>

namespace ens {
namespace callbacks {
namespace traits {

//! Detect an Evaluate() method.
ENS_HAS_EXACT_METHOD_FORM(Evaluate, HasEvaluate)
//! Detect an Gradient() method.
ENS_HAS_EXACT_METHOD_FORM(Gradient, HasGradient)
//! Detect an BeginEpoch() method.
ENS_HAS_EXACT_METHOD_FORM(BeginEpoch, HasBeginEpoch)
//! Detect an EndEpoch() method.
ENS_HAS_EXACT_METHOD_FORM(EndEpoch, HasEndEpoch)

template<typename OptimizerType,
         typename FunctionType,
         typename MatType,
         typename GradType = MatType>
struct TypedForms
{
  //! This is the form of a Evaluate() callback method.
  template<typename CallbackType>
  using EvaluateForm =
      void(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            const MatType&,
                            const double);

  //! This is the form of a Gradient() callback method.
  template<typename CallbackType>
  using GradientForm =
      void(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            const MatType&,
                            const MatType&);

  //! This is the form of a BeginEpoch() callback method.
  template<typename CallbackType>
  using BeginEpochForm =
      void(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            const MatType&,
                            const size_t,
                            const double);

  //! This is the form of a EndEpoch() callback method.
  template<typename CallbackType>
  using EndEpochForm =
      void(CallbackType::*)(OptimizerType&,
                            FunctionType&,
                            const MatType&,
                            const size_t,
                            const double);
};

} // namespace traits
} // namespace callbacks
} // namespace ens

#endif
