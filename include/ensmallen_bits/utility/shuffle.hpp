/**
 * @file shuffle.hpp
 * @author Marcus Edel
 *
 * A simple `shuffle` wrapper that based on the data type forwards to
 * `coot::shuffle` or `arma::shuffle`.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_UTILITY_SHUFFLE_HPP
#define ENSMALLEN_UTILITY_SHUFFLE_HPP

namespace ens {

// Forward to `coot::shuffle`.
template<typename InputType>
inline static typename std::enable_if<
  !arma::is_arma_type<InputType>::value &&
  ens::IsCootType<InputType>::value, InputType>::type
shuffle(const InputType& input)
{
  return coot::shuffle(input);
}

} // namespace ens

#endif
