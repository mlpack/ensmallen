/**
 * @file levi_function_n13_impl.hpp
 * @author Suryoday Basak
 *
 * Implementation of the Levi function N.13.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_LEVI_FUNCTION_N13_IMPL_HPP
#define ENSMALLEN_PROBLEMS_LEVI_FUNCTION_N13_IMPL_HPP

// In case it hasn't been included yet.
#include "levi_function_n13.hpp"
using namespace std;
using namespace arma::datum;

namespace ens {
namespace test {

inline LeviFunctionN13::LeviFunctionN13() { /* Nothing to do here */ }

inline void LeviFunctionN13::Shuffle() { /* Nothing to do here */ }

inline double LeviFunctionN13::Evaluate(const arma::mat& coordinates,
                                      const size_t /* begin */,
                                      const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  const double objective = pow(sin(3 * pi * x1), 2) +
	  		   (pow((x1 - 1), 2) *
			   (1 + pow(sin(3 * pi * x2), 2))) +
			   (pow((x2 - 1), 2) * 
			   (1 + pow(sin(2 * pi * x2), 2)));
  return objective;
}

inline double LeviFunctionN13::Evaluate(const arma::mat& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

inline void LeviFunctionN13::Gradient(const arma::mat& coordinates,
                                    const size_t /* begin */,
                                    arma::mat& gradient,
                                    const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);
  gradient.set_size(2, 1);

  gradient(0) = (((2 * x1) - 2) * (pow(sin(3 * pi), 2) + 1)) +
	  	(6 * pi * sin(3 * pi * x1) * cos(3 * pi * x1));
  gradient(1) = (6 * pi * (pow(x1 - 1, 2) * sin(3 * pi * x2) * cos(3 * pi * x2))) +
	  	(4 * pi * (pow(x2 - 1, 2) * sin(2 * pi * x2) * cos(2 * pi * x2))) +
		(((2 * x2) - 2) * pow(sin(2 * pi * x2), 2) + 1);
}

inline void LeviFunctionN13::Gradient(const arma::mat& coordinates,
                                    arma::mat& gradient)
{
  Gradient(coordinates, 0, gradient, 1);
}

} // namespace test
} // namespace ens

#endif
