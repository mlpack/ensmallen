/**
 * @file randi.hpp
 * @author Marcus Edel
 *
 * A simple `randi` wrapper that based on the data type forwards to
 * `coot::randi` or `arma::randi`.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_UTILITY_RANDU_HPP
#define ENSMALLEN_UTILITY_RANDU_HPP

namespace ens {

template<typename ElemType>
typename std::enable_if<coot::is_coot_type<ElemType>::value, ElemType>::type
randu(const size_t rows, const size_t cols)
{
  /* return coot::randu<ElemType>(rows, cols); */
  return coot::randu<ElemType>(rows, cols);
  /* ElemType foo; */
  /* return foo; */
}

template<typename ElemType>
typename std::enable_if<!coot::is_coot_type<ElemType>::value, ElemType>::type
randu(const size_t rows, const size_t cols)
{
  return arma::randu<ElemType>(rows, cols);
}

/* template<typename ElemType> */
/* typename std::enable_if<coot::is_coot_type<ElemType>::value, ElemType>::type */
/* randi(const size_t size, const arma::distr_param& param) */
/* { */
/*   return coot::randi<ElemType>(size, */
/*       coot::distr_param(param.a_int, param.b_int)); */
/* } */

/* template<typename ElemType> */
/* typename std::enable_if<!coot::is_coot_type<ElemType>::value, ElemType>::type */
/* randi(const size_t size, const arma::distr_param& param) */
/* { */
/*   return coot::randi<ElemType>(size, param); */
/* } */

/* // Forward to `coot::randi`. */
/* template<typename OutputType> */
/* inline static typename std::enable_if< */
/*     !arma::is_arma_type<OutputType>::value, void>::type */
/* randi(const size_t rows, const size_t cols, const int a, const int b, OutputType& out) */
/* { */
/*   out = coot::conv_to<OutputType>::from( */
/*       coot::randi(rows, cols, coot::distr_param(a, b))); */
/* } */

/* // Forward to `arma::randi`. */
/* template<typename OutputType> */
/* inline static typename std::enable_if< */
/*     arma::is_arma_type<OutputType>::value, void>::type */
/* randi(const size_t rows, const size_t cols, const int a, const int b, OutputType& out) */
/* { */
/*   out = arma::conv_to<OutputType>::from( */
/*       arma::randi(rows, cols, arma::distr_param(a, b))); */
/* } */

} // namespace ens

#endif
