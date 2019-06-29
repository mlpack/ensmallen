/**
 * @file ackley_function_impl.hpp
 * @author Suryoday Basak
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
using namespace std;

namespace ens {
namespace test {

inline AckleyFunction::AckleyFunction(const double c, const double epsilon) :
    c(c), epsilon(epsilon)
{ /* Nothing to do here */}

inline void AckleyFunction::Shuffle() { /* Nothing to do here */ }

inline double AckleyFunction::Evaluate(const arma::mat& coordinates,
                                       const size_t /* begin */,
                                       const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const double x1 = coordinates(0);
  const double x2 = coordinates(1);

  const double objective = -20 * exp(-0.2 * sqrt(0.5 * (x1 * x1 + x2 * x2))) -
      exp(0.5 * (cos(c * x1) + cos(c * x2))) + exp(1) + 20;

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

  // Aliases for different terms in the expression of the gradient.
  const double t0 = sqrt(0.5 * (x1 * x1 + x2 * x2));
  const double t1 = 2.0 * exp(- 0.2 * t0) / (t0 + epsilon);
  const double t2 = 0.5 * c * exp(0.5 * (cos(c * x1) +
      cos(c * x2)));

  gradient.set_size(2, 1);
  gradient(0) = (x1 * t1) + (t2 * sin(c * x1));
  gradient(1) = (x2 * t1) + (t2 * sin(c * x2));
}

inline void AckleyFunction::Gradient(const arma::mat& coordinates,
                                     arma::mat& gradient)
{
  Gradient(coordinates, 0, gradient, 1);
}

} // namespace test
} // namespace ens

#endif
