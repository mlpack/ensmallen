/**
 * @file conv_to.hpp
 * @author Marcus Edel
 *
 * A simple `linspace` wrapper that based on the data type forwards to
 * `coot::linspace` or `arma::linspace`.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_UTILITY_LINSPACE_HPP
#define ENSMALLEN_UTILITY_LINSPACE_HPP

namespace ens {

// Forward to `coot::linspace`.
template<typename OutputType>
inline static typename std::enable_if<
    !arma::is_arma_type<OutputType>::value, OutputType>::type
linspace(const double start, const double end, const size_t num = 100u)
{
  return coot::linspace<OutputType>(start, end, num);
}

// Forward to `arma::linspace`.
template<typename OutputType>
inline static typename std::enable_if<
    arma::is_arma_type<OutputType>::value, OutputType>::type
linspace(const double start, const double end, const size_t num = 100u)
{
  return arma::linspace<OutputType>(start, end, num);
}

} // namespace ens

#endif
