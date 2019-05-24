/**
 * @file goldstein_price_function_impl.hpp
 * @author Suryoday Basak
 *
 * Implementation of the Goldstein-Price function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_GOLDSTEIN_PRICE_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_GOLDSTEIN_PRICE_FUNCTION_IMPL_HPP
using namespace std;

// In case it hasn't been included yet.
#include "goldstein_price_function.hpp"

namespace ens {
namespace test {

inline GoldsteinPriceFunction::GoldsteinPriceFunction() { /* Nothing to do here */ }

inline void GoldsteinPriceFunction::Shuffle() { /* Nothing to do here */ }

inline double GoldsteinPriceFunction::Evaluate(const arma::mat& coordinates,
                                      const size_t /* begin */,
                                      const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  const double x1_sq = pow(x1, 2);
  const double x2_sq = pow(x2, 2);
  const double x1_x2 = x1 * x2
  const double objective = (1 + pow((x1 + x2 + 1), 2) * (19 - (14 * x1) +
			   (3 * x1_sq) - (14 * x2) + (6 * x1_x2) +
			   (3 * x2_sq))) * (30 + pow(((2 * x1) - (3 * x2)), 2) *
			   (18 - (32 * x1) + (12 * x1_sq) + (48 * x2) -
			   (36 * x1_x2) + (27 * x2_sq)));

  return objective;
}

inline double GoldsteinPriceFunction::Evaluate(const arma::mat& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

inline void GoldsteinPriceFunction::Gradient(const arma::mat& coordinates,
                                    const size_t /* begin */,
                                    arma::mat& gradient,
                                    const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  //Aliases for different terms in the expression of the gradient
  const double x1_sq = pow(x1, 2);
  const double x2_sq = pow(x2, 2);
  const double sum1 = (2 * x1) - (3 * x2)
  const double sum2 = x1 + x2 + 1
  const double sum3 = (6 * x1) + (6 * x2) + 14
  const double sum4 = ((12 * x1_sq) - (36 * x1 * x2) - (32 * x1) +
	              (27 * x2_sq) + (48 * x2) + 18)
  const double sum5 = ((3 * x1_sq) + (6 * x1 * x2) - (14 * x1) + 
		      (3 * x2_sq) - (14 * x2) + 19)
  const double sum6 = (6 * x1) + (6 * x2) - 14
  const double sum7 = ((6 * x1) - (9 * x2) - 8)
  const double sum1_sq = pow(sum1, 2)
  const double sum2_sq = pow(sum2, 2)
  gradient.set_size(2, 1);

  gradient(0) = ((sum1_sq * 4 * sum7) +
	  	(4 * sum1 * sum4 * ((sum2_sq * sum5) + 1))) +
		(((sum1_sq * sum4) + 30) * ((sum2_sq * sum3) +
		(2 * sum2 * sum5)));


  gradient(1) = (-6 * sum1 * sum4) + (-6 * sum1_sq * sum7) *
	  	((sum2_sq * sum5) + 1) + (((sum1_sq * sum4) + 30) *
		((sum2_sq * sum3) + (2 * sum1 * sum5)));
}

inline void GoldsteinPriceFunction::Gradient(const arma::mat& coordinates,
                                    arma::mat& gradient)
{
  Gradient(coordinates, 0, gradient, 1);
}

} // namespace test
} // namespace ens

#endif
