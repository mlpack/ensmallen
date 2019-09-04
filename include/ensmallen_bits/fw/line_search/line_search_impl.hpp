/**
 * @file line_search_impl.hpp
 * @author Chenzhe Diao
 *
 * Implementation of line search with secant method.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_LINE_SEARCH_LINE_SEARCH_IMPL_HPP
#define ENSMALLEN_LINE_SEARCH_LINE_SEARCH_IMPL_HPP

// In case it hasn't been included yet.
#include "line_search.hpp"

#include <ensmallen_bits/function.hpp>

namespace ens {

template<typename FunctionType, typename MatType, typename GradType>
typename MatType::elem_type LineSearch::Optimize(FunctionType& function,
                                                 const MatType& x1,
                                                 MatType& x2)
{
  typedef typename MatType::elem_type ElemType;

  typedef Function<FunctionType, MatType, GradType> FullFunctionType;
  FullFunctionType& f = static_cast<FullFunctionType&>(function);

  // Check that we have all the functions we will need.
  traits::CheckFunctionTypeAPI<FullFunctionType, MatType, GradType>();

  // Set up the search line, that is,
  // find the zero of der(gamma) = Derivative(gamma).
  MatType deltaX = x2 - x1;
  ElemType gamma = 0;
  ElemType derivative = Derivative<FunctionType, MatType, GradType>(f, x1,
      deltaX, 0);
  ElemType derivativeNew = Derivative<FunctionType, MatType, GradType>(f, x1,
      deltaX, 1);
  ElemType secant = derivativeNew - derivative;

  if (derivative >= 0.0) // Optimal solution at left endpoint.
  {
    x2 = x1;
    return f.Evaluate(x1);
  }
  else if (derivativeNew <= 0.0) // Optimal solution at right endpoint.
  {
    return f.Evaluate(x2);
  }
  else if (secant < tolerance) // function too flat, just take left endpoint.
  {
    x2 = x1;
    return f.Evaluate(x1);
  }

  // Line search by Secant Method.
  for (size_t k = 0; k < maxIterations; ++k)
  {
    // secant should always >=0 for convex function.
    if (secant < 0.0)
    {
      Warn << "LineSearchSecant: Function is not convex!" << std::endl;
      x2 = x1;
      return function.Evaluate(x1);
    }

    // Solve new gamma.
    ElemType gammaNew = gamma - derivative / secant;
    gammaNew = std::max(gammaNew, ElemType(0.0));
    gammaNew = std::min(gammaNew, ElemType(1.0));

    // Update secant, gamma and derivative.
    derivativeNew = Derivative<FunctionType, MatType, GradType>(function, x1,
        deltaX, gammaNew);
    secant = (derivativeNew - derivative) / (gammaNew - gamma);
    gamma = gammaNew;
    derivative = derivativeNew;

    if (std::fabs(derivative) < tolerance)
    {
      Info << "LineSearchSecant: minimized within tolerance "
          << tolerance << "; " << "terminating optimization." << std::endl;
      x2 = (1 - gamma) * x1 + gamma * x2;
      return f.Evaluate(x2);
    }
  }

  Info << "LineSearchSecant: maximum iterations (" << maxIterations
      << ") reached; " << "terminating optimization." << std::endl;

  x2 = (1 - gamma) * x1 + gamma * x2;
  return f.Evaluate(x2);
}  // Optimize


//! Derivative of the function along the search line.
template<typename FunctionType, typename MatType, typename GradType>
typename MatType::elem_type LineSearch::Derivative(FunctionType& function,
                                                   const MatType& x0,
                                                   const MatType& deltaX,
                                                   const double gamma)
{
  GradType gradient(x0.n_rows, x0.n_cols);
  function.Gradient(x0 + gamma * deltaX, gradient);
  return arma::dot(gradient, deltaX);
}

} // namespace ens

#endif
