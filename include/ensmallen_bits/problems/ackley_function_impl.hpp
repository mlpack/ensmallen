/**
 * @file ackley_function_impl.hpp
 * @author Suryoday Basak
 * @author Marcus Edel
 *
 * Implementation of the Ackley function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_ACKLEY_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_ACKLEY_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "ackley_function.hpp"

namespace ens {
namespace test {

inline AckleyFunction::AckleyFunction() { /* Nothing to do here */ }

inline void AckleyFunction::Shuffle() { /* Nothing to do here */ }

inline double AckleyFunction::Evaluate(const arma::mat& coordinates,
                                      const size_t /* begin */,
                                      const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  const double objective = -20*std::exp(-0.2*std::sqrt(0.5*(std::pow(x1,2) + 
				std::pow(x2,2)))) - 
	  			std::exp(0.5*(std::cos(2*3.1415*x1) + 
				std::cos(2*3.1415*x2))) +
				std::exp(1) + 20;

  return objective;
}

inline double AckleyFunction::Evaluate(const arma::mat& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

inline void AckleyFunction::Gradient(const arma::mat& coordinates,
                                    const size_t /* begin */,
                                    arma::mat& gradient,
                                    const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  //Aliases for different terms in the expression of the gradient
  const double pi = 3.1415;
  const double t0 = std::sqrt(0.5*(std::pow(x1,2) + std::pow(x2,2)));
  const double t1 = 2.0*std::exp(-0.2*t0)/t0
  const double t2 = pi*std::exp(0.5*(std::cos(2*pi*x1) + std::cos(2*pi*x2)));

  gradient.set_size(2, 1);
  gradient(0) = (x1*t1) + t2;
  gradient(1) = (x2*t1) + t2;
}

inline void AckleyFunction::Gradient(const arma::mat& coordinates,
                                    arma::mat& gradient)
{
  Gradient(coordinates, 0, gradient, 1);
}

} // namespace test
} // namespace ens

#endif
