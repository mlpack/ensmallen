/**
 * @file nelder_mead_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of an incremental Quasi-Newton with local superlinear
 * convergence rate as proposed by A. Mokhtari et al. in "IQN: An Incremental
 * Quasi-Newton Method with Local Superlinear Convergence Rate".
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_NELDER_MEAD_IMPL_HPP
#define ENSMALLEN_NELDER_MEAD_IMPL_HPP

// In case it hasn't been included yet.
#include "nelder_mead.hpp"

#include <ensmallen_bits/function.hpp>

namespace ens {

template<typename SimplexerType>
inline NelderMeadType<SimplexerType>::NelderMeadType(
    const size_t maxIterations,
    const double tolerance,
    const SimplexerType& simplexer) :
    alpha(0),
    beta(0),
    gamma(0),
    delta(0),
    maxIterations(maxIterations),
    tolerance(tolerance),
    simplexer(simplexer),
    adaptiveDefault(true)
{ /* Nothing to do. */ }

template<typename SimplexerType>
inline NelderMeadType<SimplexerType>::NelderMeadType(
    const double alpha,
    const double beta,
    const double gamma,
    const double delta,
    const size_t maxIterations,
    const double tolerance,
    const SimplexerType& simplexer) :
    alpha(alpha),
    beta(beta),
    gamma(gamma),
    delta(delta),
    maxIterations(maxIterations),
    tolerance(tolerance),
    simplexer(simplexer),
    adaptiveDefault(false)
{ /* Nothing to do. */ }

//! Optimize the function.
template<typename SimplexerType>
template<typename ArbitraryFunctionType,
         typename MatType,
         typename... CallbackTypes>
typename MatType::elem_type NelderMeadType<SimplexerType>::Optimize(
    ArbitraryFunctionType& function,
    MatType& iterate,
    CallbackTypes&&... callbacks)
{
  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;

  // Make sure that we have the methods that we need.  Long name...
  traits::CheckArbitraryFunctionTypeAPI<ArbitraryFunctionType,
      BaseMatType>();
  RequireDenseFloatingPointType<BaseMatType>();

  // Controls early termination of the optimization process.
  bool terminate = false;

  // Controls the shrink step.
  bool shrink = false;

  // Problem dimensionality.
  const size_t dim = iterate.n_rows;

  // Use default adaptive parameters scheme as used in "Implementing the
  // Nelder-Mead simplex algorithm with adaptive parameters" by F. Gao and L.
  // Han (2010).
  if (adaptiveDefault)
  {
    alpha = 1;
    beta = 1.0 + 2.0 / (double) dim;
    gamma = 0.75 - 0.5 / (double) dim;
    delta = 1.0 - 1.0 / (double) dim;
  }

  // Construct the initial simplex.
  MatType simplex;
  simplexer.Simplex(simplex, iterate, function);

  arma::rowvec fx(dim + 1);
  for (size_t i = 0; i < dim + 1; ++i)
  {
    fx(i) = function.Evaluate(simplex.col(i));
    Callback::Evaluate(*this, function, simplex.col(i), fx(i), callbacks...);
  }

  // Get the indices that correspond to the ordering of the function values
  // at the vertices. order(0) is the index in the simplex of the vertex
  // with the lowest function value, and order(dim) is the index in the
  // simplex of the vertex with the highest function value.
  arma::uvec order = arma::sort_index(fx);

  terminate |= Callback::BeginOptimization(*this, function, iterate,
      callbacks...);
  for (size_t i = 1; i != maxIterations && !terminate; ++i)
  {
    const MatType M = arma::mean(simplex.cols(order.subvec(0, dim - 1)), 1);
    const MatType xref = (1.0 + alpha) * M - alpha * simplex.col(order(dim));
    const ElemType fref = function.Evaluate(xref);
    Callback::Evaluate(*this, function, xref, fref, callbacks...);

    if (fref < fx(order(0)))
    {
      // Compute expansion.
      const MatType xexp = (1.0 + alpha * beta) * M - alpha * beta *
          simplex.col(order(dim));
      const ElemType fexp = function.Evaluate(xexp);
      Callback::Evaluate(*this, function, xexp, fexp, callbacks...);

      // Shift values by one to the right and swap the last with the first.
      const size_t fLowest = order(dim);
      for (size_t d = dim; d > 0; --d)
        order(d) = order(d - 1);
      order(0) = fLowest;

      if (fexp < fref)
      {
        simplex.col(order(0)) = xexp;
        fx(order(0)) = fexp;
      }
      else
      {
        simplex.col(order(0)) = xref;
        fx(order(0)) = fref;
      }
    }
    else if (fref < fx(order(dim - 1)))
    {
      // Accept reflection point.
      const size_t s = ShiftOrder(order, fx, fref);
      simplex.col(order(s)) = xref;
      fx(order(s)) = fref;
    }
    else
    {
      if (fref < fx(order(dim)))
      {
        // Outside contraction.
        const MatType xoc = (1.0 + alpha * gamma) * M - alpha * gamma *
            simplex.col(order(dim));
        const ElemType foc = function.Evaluate(xoc);
        Callback::Evaluate(*this, function, xoc, foc, callbacks...);

        if (foc <= fref)
        {
          const size_t s = ShiftOrder(order, fx, fref);
          simplex.col(order(s)) = xoc;
          fx(order(s)) = foc;
        }
        else
        {
          shrink = true;
        }
      }
      else
      {
        // Inside contraction.
        const MatType xic = (1.0 - gamma) * M + gamma *
            simplex.col(order(dim));
        const ElemType fic = function.Evaluate(xic);
        Callback::Evaluate(*this, function, xic, fic, callbacks...);

        if (fic < fx(order(dim)))
        {
          const size_t s = ShiftOrder(order, fx, fic);
          simplex.col(order(s)) = xic;
          fx(order(s)) = fic;
        }
        else
        {
          shrink = true;
        }
      }
    }

    if (shrink)
    {
      for (size_t d = 1; d < dim; ++d)
      {
        simplex.col(order(d)) = simplex.col(order(0)) + delta *
            (simplex.col(order(d)) - simplex.col(order(0)));
        fx(order(d)) = function.Evaluate(simplex.col(order(d)));

        Callback::Evaluate(*this, function, simplex.col(order(d)),
            fx(order(d)), callbacks...);
      }

      // Sorting can be slow, however there is overwhelming evidence that shrink
      // transformations almost never happen in practice. See "Efficient
      // Implementation of the Nelder–Mead Search Algorithm" by S. Singer and S.
      // Singer.
      order = arma::sort_index(fx);
      fx = fx.cols(order);
      simplex = simplex.cols(order);

      shrink = false;
    }

    // Check for termination criteria.
    if (std::abs(fx(order(dim)) - fx(order(0))) < tolerance)
    {
      Info << "NelderMead: minimized within tolerance " << tolerance << "; "
           << "terminating optimization." << std::endl;
      break;
    }
  }

  // Set the best candidate.
  iterate = simplex.col(0);

  const ElemType objective = function.Evaluate(iterate);
  Callback::Evaluate(*this, function, iterate, objective, callbacks...);

  Callback::EndOptimization(*this, function, iterate, callbacks...);
  return objective;
}

} // namespace ens

#endif
