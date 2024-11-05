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

template<typename ElemType>
typename std::enable_if<!IsCootType<ElemType>::value, ElemType>::type
randu(const size_t rows, const size_t cols)
{
  return arma::randu<ElemType>(rows, cols);
}

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

template<typename ElemType>
typename std::enable_if<!IsCootType<ElemType>::value, ElemType>::type
randn(const size_t rows, const size_t cols)
{
  return arma::randn<ElemType>(rows, cols);
}

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

template<typename ElemType>
typename std::enable_if<!IsCootType<ElemType>::value, ElemType>::type
randi(const size_t rows, const size_t cols, const arma::distr_param& param)
{
  return arma::randi<ElemType>(rows, cols, param);
}

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
  #ifdef USE_COOT
  return coot::shuffle(input);
  #else
  return arma::shuffle(input);
  #endif
}

/**
 * A utility class that based on the data type forwards to `coot::conv_to` or
 * `arma::conv_to`.
 *
 * @tparam OutputType The data type to convert to.
 */
template<typename OutputType>
class conv_to
{
  public:
   /**
    * Convert from one matrix type to another by forwarding to `coot::conv_to`.
    *
    * @param input The input that is converted.
    */
   template<typename InputType, typename foo = OutputType>
   inline static typename std::enable_if<
      !IsArmaType<foo>::value, OutputType>::type
   from(const InputType& input)
   {
     #ifdef USE_COOT
     return coot::conv_to<OutputType>::from(input);
     #else
     return arma::conv_to<OutputType>::from(input);
     #endif
   }

   /**
    * Convert from one matrix type to another by forwarding to `arma::conv_to`.
    *
    * @param input The input that is converted.
    */
   template<typename InputType, typename foo = OutputType>
   inline static typename std::enable_if<
      IsArmaType<foo>::value, OutputType>::type
   from(const InputType& input)
   {
     return arma::conv_to<OutputType>::from(input);
   }
};

} // namespace ens

#endif
