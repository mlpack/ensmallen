/**
 * @file beale_function_impl.hpp
 * @author Suryoday Basak
 *
 * Implementation of the Beale function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_BEALE_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_BEALE_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "beale_function.hpp"

namespace ens {
namespace test {

inline BealeFunction::BealeFunction() { /* Nothing to do here */ }

inline void BealeFunction::Shuffle() { /* Nothing to do here */ }

inline double BealeFunction::Evaluate(const arma::mat& coordinates,
                                      const size_t /* begin */,
                                      const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  const double objective = std::pow((1.5 - x1 + (x1 * x2)), 2) + 
	  		   std::pow((2.25 - x1 + (x1 * std::pow(x2, 2))), 2) + 
			   std::pow((2.625 - x1 + (x1 * std::pow(x2, 3))), 2);

  return objective;
}

inline double BealeFunction::Evaluate(const arma::mat& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

inline void BealeFunction::Gradient(const arma::mat& coordinates,
                                    const size_t /* begin */,
                                    arma::mat& gradient,
                                    const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  //Aliases for different terms in the expression of the gradient
  const double x2_sq = std::pow(x2, 2);
  const double x2_cub = std::pow(x2, 3);

  gradient.set_size(2, 1);
  gradient(0) = (((2 * x2) - 2) * ((x1 * x2) - x1 + 1.5)) +
	  	(((2 * x2_sq) - 2) * ((x1 * x2_sq) - x1 + 2.25)) + 
		(((2 * x2_cub) - 2) * ((x1 * x2_cub) -x1 + 2.625));
  gradient(1) = (6 * x1 * x2_sq * ((x1 * x2_cub) - x1 + 2.625)) +
	  	(4 * x1 * x2 * ((x1 * x2_sq) -x1 + 2.25)) +
		(2 * x1 * ((x1 * x2) - x1 + 1.5));
}

inline void BealeFunction::Gradient(const arma::mat& coordinates,
                                    arma::mat& gradient)
{
  Gradient(coordinates, 0, gradient, 1);
}

} // namespace test
} // namespace ens

#endif
