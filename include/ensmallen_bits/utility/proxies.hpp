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

#include "function_traits.hpp"

namespace ens {

/**
 * Helper struct that based on the data type `MatType` forwards to the
 * corresponding `coot` or `arma` types. For example:
 * If `MatType` is an `arma::mat`, then `ForwardType<MatType>::bmat` will be an
 * `arma::Mat<ElemType>`.
 * If `MatType` is a `coot::mat`, then `ForwardType<MatType>::bmat` will be a
 * `coot::Mat<ElemType>`.
 *
 * This allows for writing generic code that can work with both `coot` and
 * `arma` types without needing to know which library is being used at compile
 * time.
 */
template<typename MatType, typename ElemType = typename MatType::elem_type>
struct ForwardType
{
    // If MatType is an Bandicoot type, then we use Bandicoot types.
    #ifdef USE_COOT

    // `uword` is a typedef for an unsigned integer type; it is used for matrix
    // indices as well as all internal counters and loops.
    typedef typename std::conditional<IsArmaType<MatType>::value,
        arma::uword, coot::uword>::type uword;

    // `vec` is a typedef for column vectors (dense matrices with one column).
    typedef typename std::conditional<IsArmaType<MatType>::value,
        arma::vec, coot::vec>::type vec;

    // `bvec` (base vector) is a typedef for a vector type, in comparison to
    // `vec`, `bvec` uses the given element type `ElemType`.
    typedef typename std::conditional<IsArmaType<MatType>::value,
        arma::Col<ElemType>, coot::Col<ElemType>>::type bvec;

    // `bcol` (base col) is a typedef for a column vector type, in comparison to
    // `col`, `bcol` uses the given element type `ElemType`.
    typedef typename std::conditional<IsArmaType<MatType>::value,
        arma::Col<ElemType>, coot::Col<ElemType>>::type bcol;

    // `brow` (base row) is a typedef for a row vector type, isn comparison to
    // `row`, brow uses the given element type ElemType.
    typedef typename std::conditional<IsArmaType<MatType>::value,
        arma::Row<ElemType>, coot::Row<ElemType>>::type brow;

    // `mat` is a typedef for dense matrices, with elements stored in
    // column-major ordering (ie. column by column).
    typedef typename std::conditional<IsArmaType<MatType>::value,
        arma::mat, coot::mat>::type mat;

    // `bmat` (base matrix) is a typedef for a matrix type, in comparison to
    // `mat`, `bmat` uses the given element type `ElemType`.
    typedef typename std::conditional<IsArmaType<MatType>::value,
        arma::Mat<ElemType>, coot::Mat<ElemType>>::type bmat;

    // `cube` is a typedef for 3D matrices (cubes), with elements stored in
    // column-major ordering (ie. column by column, then page by page).
    typedef typename std::conditional<IsArmaType<MatType>::value,
        arma::cube, coot::cube>::type cube;

    // `bcube` (base cube) is a typedef for a cube type, in comparison to `cube`,
    // `bcube` uses the given element type `ElemType`.
    typedef typename std::conditional<IsArmaType<MatType>::value,
        arma::Cube<ElemType>, coot::Cube<ElemType>>::type bcube;

    // `umat` is a typedef for unsigned integer matrices, with elements stored in
    // column-major ordering (ie. column by column).
    typedef typename std::conditional<IsArmaType<MatType>::value,
        arma::umat, coot::umat>::type umat;

    // `uvec` is a typedef for unsigned integer vectors (dense matrices with one
    // column).
    typedef typename std::conditional<IsArmaType<MatType>::value,
        arma::uvec, coot::uvec>::type uvec;
    // `ucolvec` is a typedef for unsigned integer column vectors (dense matrices
    // with one column).
    typedef typename std::conditional<IsArmaType<MatType>::value,
        arma::ucolvec, coot::ucolvec>::type ucolvec;

    // `urowvec` is a typedef for unsigned integer row vectors (dense matrices
    // with one row).
    typedef typename std::conditional<IsArmaType<MatType>::value,
        arma::urowvec, coot::urowvec>::type urowvec;

    // `distr_param` is a typedef for the distribution parameters used in
    // random number generation.
    typedef typename std::conditional<IsArmaType<MatType>::value,
        arma::distr_param, coot::distr_param>::type distr_param;
    #else
    // If MatType is an Armadillo type, then we use Armadillo types.

    // `uword` is a typedef for an unsigned integer type; it is used for matrix
    // indices as well as all internal counters and loops.
    typedef arma::uword uword;

    // `vec` is a typedef for column vectors (dense matrices with one column).
    typedef arma::vec vec;

    // `bvec` (base vector) is a typedef for a vector type, in comparison to
    // `vec`, `bvec` uses the given element type `ElemType`.
    typedef arma::Col<ElemType> bvec;

    // `bcol` (base col) is a typedef for a column vector type, in comparison to
    // `col`, `bcol` uses the given element type `ElemType`.
    typedef arma::Col<ElemType> bcol;

    // `brow` (base row) is a typedef for a row vector type, in comparison to
    // `row`, `brow` uses the given element type `ElemType`.
    typedef arma::Row<ElemType> brow;

    // `mat` is a typedef for dense matrices, with elements stored in
    // column-major ordering (ie. column by column).
    typedef arma::mat mat;

    // `bmat` (base matrix) is a typedef for a matrix type, in comparison to
    // `mat`, `bmat` uses the given element type `ElemType`.
    typedef arma::Mat<ElemType> bmat;

    // `cube` is a typedef for 3D matrices (cubes), with elements stored in
    // column-major ordering (ie. column by column, then page by page).
    typedef arma::cube cube;

    // `bcube` (base cube) is a typedef for a cube type, in comparison to `cube`,
    // `bcube` uses the given element type `ElemType`.
    typedef arma::Cube<ElemType> bcube;

    // `umat` is a typedef for unsigned integer matrices, with elements stored in
    // column-major ordering (ie. column by column).
    typedef arma::umat umat;

    // `uvec` is a typedef for unsigned integer vectors (dense matrices with one
    // column).
    typedef arma::uvec uvec;

    // `ucolvec` is a typedef for unsigned integer column vectors (dense matrices
    // with one column).
    typedef arma::ucolvec ucolvec;

    // `urowvec` is a typedef for unsigned integer row vectors (dense matrices
    // with one row).
    typedef arma::urowvec urowvec;

    // `distr_param` is a typedef for the distribution parameters used in
    // random number generation.
    typedef arma::distr_param distr_param;

    #endif
};

// Proxy for `arma::shuffle` or `coot::shuffle` based on the data type.
template<typename MatType>
typename std::enable_if<
    IsCootType<MatType>::value, typename ForwardType<MatType>::uvec>::type
sort_index(const MatType& input)
{
  #ifdef USE_COOT
  return coot::sort_index(input);
  #else
  return arma::sort_index(input);
  #endif
}

// Armadillo proxy for `arma::sort_index` if `MatType` is not a Coot type.
template<typename MatType>
typename std::enable_if<
    !IsCootType<MatType>::value, typename ForwardType<MatType>::uvec>::type
sort_index(const MatType& input)
{
  return arma::sort_index(input);
}

// Proxy for `arma::randn` or `coot::randn` based on the data type.
template<typename ElemType>
typename std::enable_if<IsCootType<ElemType>::value, ElemType>::type
randn(const size_t rows, const size_t cols)
{
  #ifdef USE_COOT
  return coot::randn<ElemType>(rows, cols);
  #else
  return arma::randn<ElemType>(rows, cols);
  #endif
}

// Armadillo proxy for `arma::randn` if `ElemType` is not a Coot type.
template<typename ElemType>
typename std::enable_if<!IsCootType<ElemType>::value, ElemType>::type
randn(const size_t rows, const size_t cols)
{
  return arma::randn<ElemType>(rows, cols);
}

// Proxy for `arma::randi` or `coot::randi` based on the data type.
template<typename ElemType>
typename std::enable_if<IsCootType<ElemType>::value, ElemType>::type
randi(const size_t rows, const size_t cols, const arma::distr_param& param)
{
  #ifdef USE_COOT
  return coot::randi<ElemType>(rows, cols,
      coot::distr_param(param.a_int, param.b_int));
  #else
  return arma::randi<ElemType>(rows, cols, param);
  #endif
}

// Armadillo proxy for `arma::randi` if `ElemType` is not a Coot type.
template<typename ElemType>
typename std::enable_if<!IsCootType<ElemType>::value, ElemType>::type
randi(const size_t rows, const size_t cols, const arma::distr_param& param)
{
  return arma::randi<ElemType>(rows, cols, param);
}

// Proxy for `arma::randu` or `coot::randu` based on the data type.
template<typename ElemType>
typename std::enable_if<IsCootType<ElemType>::value, ElemType>::type
randu(const size_t rows, const size_t cols)
{
  #ifdef USE_COOT
  return coot::randu<ElemType>(rows, cols);
  #else
  return arma::randu<ElemType>(rows, cols);
  #endif
}

// Armadillo proxy for `arma::randu` if `ElemType` is not a Coot type.
template<typename ElemType>
typename std::enable_if<!IsCootType<ElemType>::value, ElemType>::type
randu(const size_t rows, const size_t cols)
{
  return arma::randu<ElemType>(rows, cols);
}

// Proxy for `arma::linspace` or `coot::linspace` based on the data type.
template<typename OutputType>
inline static typename std::enable_if<
    !arma::is_arma_type<OutputType>::value, OutputType>::type
linspace(const double start, const double end, const size_t num = 100u)
{
  #ifdef USE_COOT
  return coot::linspace<OutputType>(start, end, num);
  #else
  return arma::linspace<OutputType>(start, end, num);
  #endif
}

// Armadillo proxy for `arma::linspace` if `OutputType` is an Armadillo type.
template<typename OutputType>
inline static typename std::enable_if<
    arma::is_arma_type<OutputType>::value, OutputType>::type
linspace(const double start, const double end, const size_t num = 100u)
{
  return arma::linspace<OutputType>(start, end, num);
}

// Proxy for `arma::regspace` or `coot::regspace` based on the data type.
template<typename OutputType>
inline static typename std::enable_if<
    !arma::is_arma_type<OutputType>::value, OutputType>::type
regspace(const double start, const double end, const size_t num = 100u)
{
  #ifdef USE_COOT
  return coot::regspace<OutputType>(start, end, num);
  #else
  return arma::regspace<OutputType>(start, end, num);
  #endif
}

// Armadillo proxy for `arma::regspace` if `OutputType` is an Armadillo type.
template<typename OutputType>
inline static typename std::enable_if<
    arma::is_arma_type<OutputType>::value, OutputType>::type
regspace(const double start, const double end, const size_t num = 100u)
{
  return arma::regspace<OutputType>(start, end, num);
}

} // namespace ens

#endif
