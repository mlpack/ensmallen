/**
 * @file traits.hpp
 * @author Ryan Curtin
 *
 * This file provides metaprogramming utilities for detecting certain members of
 * FunctionType classes.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CALLBACKS_TRAITS_HPP
#define ENSMALLEN_CALLBACKS_TRAITS_HPP

#include "../function/sfinae_utility.hpp"

namespace ens {
namespace traits {

ENS_HAS_MEM_FUNC_ANY(Evaluate, HasEvaluateVoid)
ENS_HAS_MEM_FUNC_ANY(Evaluate, HasEvaluateBool)

} // namespace traits
} // namespace ens

#endif
