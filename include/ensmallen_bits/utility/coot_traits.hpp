/**
 * @file coot_traits.hpp
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Some traits used for template metaprogramming (SFINAE) with Bandicoot types.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_UTILITY_COOT_TRAITS_HPP
#define ENSMALLEN_UTILITY_COOT_TRAITS_HPP

namespace ens {

// Structs have public members by default (that's why they are chosen over
// classes).

/**
 * If value == true, then MatType is some sort of Bandicoot vector or subview.
 * You might use this struct like this:
 *
 * @code
 * // Only accepts VecTypes that are actually Bandicoot vector types.
 * template<typename MatType>
 * void Function(const MatType& argumentA,
 *               typename std::enable_if_t<IsCootType<MatType>::value>* = 0);
 * @endcode
 *
 * The use of the enable_if_t object allows the compiler to instantiate
 * Function() only if VecType is one of the Bandicoot vector types.  It has a
 * default argument because it isn't meant to be used in either the function
 * call or the function body.
 */
template<typename MatType>
struct IsCootType
{
  const static bool value = false;
};

#ifdef USE_COOT

// Commenting out the first template per case, because
// Visual Studio doesn't like this instantiaion pattern (error C2910).
// template<>
template<typename eT>
struct IsCootType<coot::Col<eT> >
{
  const static bool value = true;
};

// template<>
template<typename eT>
struct IsCootType<coot::Row<eT> >
{
  const static bool value = true;
};

// template<>
template<typename eT>
struct IsCootType<coot::subview<eT> >
{
  const static bool value = true;
};

// template<>
template<typename eT>
struct IsCootType<coot::subview_col<eT> >
{
  const static bool value = true;
};

// template<>
template<typename eT>
struct IsCootType<coot::subview_row<eT> >
{
  const static bool value = true;
};

// template<>
template<typename eT>
struct IsCootType<coot::Mat<eT> >
{
  const static bool value = true;
};

#endif

} // namespace ens

#endif
