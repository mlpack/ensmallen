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
typename std::enable_if<coot::is_coot_type<ElemType>::value, ElemType>::type
randu(const size_t rows, const size_t cols)
{
  return coot::randu<ElemType>(rows, cols);
}

template<typename ElemType>
typename std::enable_if<!coot::is_coot_type<ElemType>::value, ElemType>::type
randu(const size_t rows, const size_t cols)
{
  return arma::randu<ElemType>(rows, cols);
}

template<typename ElemType>
typename std::enable_if<coot::is_coot_type<ElemType>::value, ElemType>::type
randn(const size_t rows, const size_t cols)
{
  return coot::randn<ElemType>(rows, cols);
}

template<typename ElemType>
typename std::enable_if<!coot::is_coot_type<ElemType>::value, ElemType>::type
randn(const size_t rows, const size_t cols)
{
  return arma::randn<ElemType>(rows, cols);
}

template<typename ElemType>
typename std::enable_if<coot::is_coot_type<ElemType>::value, ElemType>::type
randi(const size_t rows, const size_t cols, const arma::distr_param& param)
{
  return coot::randi<ElemType>(rows, cols,
      coot::distr_param(param.a_int, param.b_int));
}

template<typename ElemType>
typename std::enable_if<!coot::is_coot_type<ElemType>::value, ElemType>::type
randi(const size_t rows, const size_t cols, const arma::distr_param& param)
{
  return arma::randi<ElemType>(rows, cols, param);
}

template<typename OutputType>
inline static typename std::enable_if<
    !arma::is_arma_type<OutputType>::value, OutputType>::type
linspace(const double start, const double end, const size_t num = 100u)
{
  return coot::linspace<OutputType>(start, end, num);
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
  return coot::shuffle(input);
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
     return coot::conv_to<OutputType>::from(input);
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
