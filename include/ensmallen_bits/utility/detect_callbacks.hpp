/**
 * @file ensmallen_bits/utility/detect_callbacks.hpp
 * @author Ryan Curtin
 *
 * This provides the IsAllNonMatrix utility struct, meant to be used with SFINAE
 * to ensure that template arguments are only non-Armadillo classes.  (This does
 * not actually check that callback functions are implemented!)
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENS_CORE_UTIL_DETECT_CALLBACKS_HPP
#define ENS_CORE_UTIL_DETECT_CALLBACKS_HPP

namespace ens {

template<typename... Ts>
struct IsAllNonMatrix;

template<typename T, typename... Ts>
struct IsAllNonMatrix<T, Ts...>
{
  constexpr static bool tIsClass = std::is_class<typename std::remove_cv<
          typename std::remove_reference<T>::type>::type>::value;

  constexpr static bool value =
      tIsClass && !IsMatrixType<T>::value &&
      IsAllNonMatrix<Ts...>::value;
};

template<>
struct IsAllNonMatrix<>
{
  constexpr static bool value = true;
};

} // namespace ens

#endif
