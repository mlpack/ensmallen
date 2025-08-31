/**
 * @file arma_traits.hpp
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Some traits used for template metaprogramming (SFINAE) with Armadillo types.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_UTILITY_ARMA_TRAITS_HPP
#define ENSMALLEN_UTILITY_ARMA_TRAITS_HPP

namespace ens {

// Structs have public members by default (that's why they are chosen over
// classes).

template<typename MatType> struct IsArmaType;
template<typename MatType> struct IsCootType;

/**
 * If value == true, then MatType is a matrix type matching the Armadillo API
 * that is supported by ensmallen.
 */
template<typename MatType>
struct IsMatrixType
{ 
  const static bool value = IsArmaType<MatType>::value ||
                            IsCootType<MatType>::value;
};

/**
 * If value == true, then MatType is an Armadillo sparse matrix.
 */
template<typename MatType>
struct IsSparseMatrixType
{
  const static bool value = false;
};

template<typename eT>
struct IsSparseMatrixType<arma::SpMat<eT>>
{
  const static bool value = true;
};

/**
 * If value == true, then MatType is some sort of Armadillo vector or subview.
 * You might use this struct like this:
 *
 * @code
 * // Only accepts VecTypes that are actually Armadillo vector types.
 * template<typename MatType>
 * void Function(const MatType& argumentA,
 *               typename std::enable_if_t<IsArmaType<MatType>::value>* = 0);
 * @endcode
 *
 * The use of the enable_if_t object allows the compiler to instantiate
 * Function() only if VecType is one of the Armadillo vector types.  It has a
 * default argument because it isn't meant to be used in either the function
 * call or the function body.
 */
template<typename MatType>
struct IsArmaType
{
  const static bool value = false;
};

template<typename eT>
struct IsArmaType<arma::Col<eT> >
{
  const static bool value = true;
};

template<typename eT>
struct IsArmaType<arma::SpCol<eT> >
{
  const static bool value = true;
};

template<typename eT>
struct IsArmaType<arma::Row<eT> >
{
  const static bool value = true;
};

template<typename eT>
struct IsArmaType<arma::SpRow<eT> >
{
  const static bool value = true;
};

template<typename eT>
struct IsArmaType<arma::subview<eT> >
{
  const static bool value = true;
};

template<typename eT>
struct IsArmaType<arma::subview_col<eT> >
{
  const static bool value = true;
};

template<typename eT>
struct IsArmaType<arma::subview_row<eT> >
{
  const static bool value = true;
};

template<typename eT>
struct IsArmaType<arma::SpSubview<eT> >
{
  const static bool value = true;
};

template<typename eT>
struct IsArmaType<arma::SpSubview_col<eT> >
{
  const static bool value = true;
};

template<typename eT>
struct IsArmaType<arma::SpSubview_row<eT> >
{
  const static bool value = true;
};

template<typename eT>
struct IsArmaType<arma::Mat<eT> >
{
  const static bool value = true;
};

template<typename eT>
struct IsArmaType<arma::SpMat<eT> >
{
  const static bool value = true;
};

template<typename eT>
struct IsArmaType<arma::Cube<eT> >
{
  const static bool value = true;
};

template<typename eT>
struct IsArmaType<arma::subview_cube<eT> >
{
  const static bool value = true;
};

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

#ifdef ENS_HAVE_COOT

template<typename eT>
struct IsCootType<coot::Col<eT> >
{
  const static bool value = true;
};

template<typename eT>
struct IsCootType<coot::Row<eT> >
{
  const static bool value = true;
};

template<typename eT>
struct IsCootType<coot::subview<eT> >
{
  const static bool value = true;
};

template<typename eT>
struct IsCootType<coot::subview_col<eT> >
{
  const static bool value = true;
};

template<typename eT>
struct IsCootType<coot::subview_row<eT> >
{
  const static bool value = true;
};

template<typename eT>
struct IsCootType<coot::Mat<eT> >
{
  const static bool value = true;
};

template<typename eT>
struct IsCootType<coot::Cube<eT> >
{
  const static bool value = true;
};

template<typename eT>
struct IsCootType<coot::subview_cube<eT> >
{
  const static bool value = true;
};

#endif


template <int N, typename... T>
struct tuple_element;

template <typename T0, typename... T>
struct tuple_element<0, T0, T...> {
    typedef T0 type;
};
template <int N, typename T0, typename... T>
struct tuple_element<N, T0, T...> {
    typedef typename tuple_element<N-1, T...>::type type;
};

} // namespace ens

#endif
