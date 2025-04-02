/**
 * @file proxies.hpp
 * @author Marcus Edel
 *
 * Simple proxies that based on the data type forwards to `coot` or `arma`.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_UTILITY_PROXIES_HPP
#define ENSMALLEN_UTILITY_PROXIES_HPP

namespace ens {

template<typename OutputType>
inline static typename std::enable_if<
    !arma::is_arma_type<OutputType>::value, OutputType>::type
linspace(const double start, const double end, const size_t num = 100u)
{
  #ifdef ENS_HAS_COOT
  return coot::linspace<OutputType>(start, end, num);
  #else
  return arma::linspace<OutputType>(start, end, num);
  #endif
}

template<typename OutputType>
inline static typename std::enable_if<
    arma::is_arma_type<OutputType>::value, OutputType>::type
linspace(const double start, const double end, const size_t num = 100u)
{
  return arma::linspace<OutputType>(start, end, num);
}

template<typename InputType>
inline static typename std::enable_if<
  !arma::is_arma_type<InputType>::value &&
  ens::IsCootType<InputType>::value, InputType>::type
shuffle(const InputType& input)
{
  #ifdef ENS_HAS_COOT
  return coot::shuffle(input);
  #else
  return arma::shuffle(input);
  #endif
}

} // namespace ens

#endif
