/**
 * @file callbacks.hpp
 * @author Marcus Edel
 *
 * The Callback class will invoke the specified callbacks.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CALLBACKS_CALLBACKS_HPP
#define ENSMALLEN_CALLBACKS_CALLBACKS_HPP

#include <ensmallen_bits/callbacks/traits.hpp>

namespace ens {

class Callback
{
 public:
  template<typename CallbackFunctionType, typename... Targs>
  static typename std::enable_if<
    traits::HasEvaluateBool<CallbackFunctionType, bool, Targs...>::value &&
    !traits::HasEvaluateVoid<CallbackFunctionType, void, Targs...>::value, bool>::type
  Evaluate(CallbackFunctionType& callback, Targs... t)
  {
    return callback.Evaluate(t...);;
  }

  template<typename CallbackFunctionType, typename... Targs>
  static typename std::enable_if<
    !traits::HasEvaluateBool<CallbackFunctionType, bool, Targs...>::value &&
    traits::HasEvaluateVoid<CallbackFunctionType, void, Targs...>::value, bool>::type
  Evaluate(CallbackFunctionType& callback, Targs... t)
  {
    callback.Evaluate(t...);
    return false;
  }

  template<typename CallbackFunctionType, typename... Targs>
  static typename std::enable_if<
    !traits::HasEvaluateBool<CallbackFunctionType, bool, Targs...>::value &&
    !traits::HasEvaluateVoid<CallbackFunctionType, void, Targs...>::value, bool>::type
  Evaluate(CallbackFunctionType& callback, Targs... t)
  {
    return false;
  }

  template<typename CallbackFunctionType, typename... Targs>
  static typename std::enable_if<
    traits::HasEvaluateBool<CallbackFunctionType, bool, Targs...>::value &&
    traits::HasEvaluateVoid<CallbackFunctionType, void, Targs...>::value, bool>::type
  Evaluate(CallbackFunctionType& callback, Targs... t)
  {
    Warn << "Call to member function 'Evaluate' is ambiguous." << std::endl;
    return false;
  }

  template<class... CallbackFunctionTypes, typename... Targs>
  static bool Callbacks(CallbackFunctionTypes... callbackFunctions, Targs... t)
  {
    // This will return immediately once a callback returns true.
    bool result = false;
    (void)std::initializer_list<bool>{ result =
        result || Callback::Evaluate(callbackFunctions, t...)... };
    return result;
  }
};

} // namespace ens

#endif
