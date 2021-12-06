/**
 * @file accu.hpp
 * @author Marcus Edel
 *
 * A simple and dangerous implementation of 'Any' that can be used when the
 * class needs to hold some specific information for which the type is not
 * known.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_UTILITY_ACCU_HPP
#define ENSMALLEN_UTILITY_ACCU_HPP

namespace ens {

/**
 * A utility class that based on the data type forwards to `coot::conv_to` or
 * `arma::conv_to`.
 *
 * @tparam OutputType The data type to convert to.
 */
/*template<typename OutputType>*/
/*class conv_to*/
/*{*/
/*  public:*/
/*   /***/
/*    * Convert from one matrix type to another by forwarding to `coot::conv_to`.*/
/*    **/
/*    * @param input The input that is converted.*/
/*    */*/
/*   template<typename InputType>*/
/*   inline static typename std::enable_if<*/
/*      !arma::is_arma_type<InputType>::value, OutputType>::type*/
/*   from(const InputType& input)*/
/*   {*/
/*     return coot::conv_to<OutputType>::from(input);*/
/*   }*/

/*   /***/
/*    * Convert from one matrix type to another by forwarding to `arma::conv_to`.*/
/*    **/
/*    * @param input The input that is converted.*/
/*    */*/
/*   template<typename InputType>*/
/*   inline static typename std::enable_if<*/
/*      arma::is_arma_type<InputType>::value, OutputType>::type*/
/*   from(const InputType& input)*/
/*   {*/
/*     return arma::conv_to<OutputType>::from(input);*/
/*   }*/
/*};*/

} // namespace ens

#endif
