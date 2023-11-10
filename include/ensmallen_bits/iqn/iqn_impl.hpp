/**
 * @file iqn_impl.hpp
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
#ifndef ENSMALLEN_IQN_IQN_IMPL_HPP
#define ENSMALLEN_IQN_IQN_IMPL_HPP

// In case it hasn't been included yet.
#include "iqn.hpp"

#include <ensmallen_bits/function.hpp>

namespace ens {

inline IQN::IQN(const double stepSize,
                const size_t batchSize,
                const size_t maxIterations,
                const double tolerance) :
    stepSize(stepSize),
    batchSize(batchSize),
    maxIterations(maxIterations),
    tolerance(tolerance)
{ /* Nothing to do. */ }

//! Optimize the function (minimize).
template<typename SeparableFunctionType,
         typename MatType,
         typename GradType,
         typename... CallbackTypes>
typename std::enable_if<IsArmaType<GradType>::value,
typename MatType::elem_type>::type
IQN::Optimize(SeparableFunctionType& functionIn,
              MatType& iterateIn,
              CallbackTypes&&... callbacks)
{
  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;
  typedef typename MatTypeTraits<GradType>::BaseMatType BaseGradType;

  typedef Function<SeparableFunctionType, BaseMatType, BaseGradType>
      FullFunctionType;
  FullFunctionType& function(static_cast<FullFunctionType&>(functionIn));

  // Make sure we have all the methods that we need.
  traits::CheckSeparableFunctionTypeAPI<FullFunctionType, BaseMatType,
      BaseGradType>();
  RequireDenseFloatingPointType<BaseMatType>();
  RequireDenseFloatingPointType<BaseGradType>();
  RequireSameInternalTypes<BaseMatType, BaseGradType>();

  traits::CheckSeparableFunctionTypeAPI<SeparableFunctionType,
      BaseMatType, BaseGradType>();

  // Find the number of functions.
  const size_t numFunctions = function.NumFunctions();
  size_t numBatches = numFunctions / batchSize;
  if (numFunctions % batchSize != 0)
    ++numBatches; // Capture last few.

  BaseMatType& iterate = (BaseMatType&) iterateIn;

  // To keep track of where we are and how things are going.
  ElemType overallObjective = 0;

  // Controls early termination of the optimization process.
  bool terminate = false;

  std::vector<BaseGradType> y(numBatches, BaseGradType(iterate.n_rows,
      iterate.n_cols));
  std::vector<BaseMatType> t(numBatches, BaseMatType(iterate.n_rows,
      iterate.n_cols));
  std::vector<BaseMatType> Q(numBatches, BaseMatType(iterate.n_elem,
      iterate.n_elem));
  BaseMatType initialIterate = arma::randn<arma::Mat<ElemType>>(iterate.n_rows,
      iterate.n_cols);
  BaseGradType B(iterate.n_elem, iterate.n_elem);
  B.eye();

  BaseGradType g(iterate.n_rows, iterate.n_cols);
  g.zeros();
  for (size_t i = 0, f = 0; i < numFunctions; f++)
  {
    // Find the effective batch size (the last batch may be smaller).
    const size_t effectiveBatchSize = std::min(batchSize, numFunctions - i);

    // It would be nice to avoid this copy but it is difficult to be generic to
    // any MatType and still do that.
    t[f] = initialIterate;
    function.Gradient(initialIterate, i, y[f], effectiveBatchSize);

    terminate |= Callback::Gradient(*this, function, initialIterate,
        y[f], callbacks...);

    Q[f].eye();
    g += y[f];
    y[f] /= (double) effectiveBatchSize;

    i += effectiveBatchSize;
  }
  g /= numFunctions;

  BaseGradType gradient(iterate.n_rows, iterate.n_cols);
  BaseMatType u = t[0];

  Callback::BeginOptimization(*this, function, iterate, callbacks...);
  for (size_t i = 1; i != maxIterations && !terminate; ++i)
  {
    for (size_t j = 0, f = 0; f < numFunctions; j++)
    {
      // Cyclicly iterating through the number of functions.
      const size_t it = ((j + 1) % numBatches);

      // Find the effective batch size (the last batch may be smaller).
      const size_t effectiveBatchSize = std::min(batchSize, numFunctions -
          it * batchSize);

      if (arma::norm(iterate - t[it]) > 0)
      {
        function.Gradient(iterate, it * batchSize, gradient,
            effectiveBatchSize);
        gradient /= effectiveBatchSize;

        terminate |= Callback::Gradient(*this, function, iterate, gradient,
            callbacks...);

        const BaseMatType s = arma::vectorise(iterate - t[it]);
        const BaseGradType yy = arma::vectorise(gradient - y[it]);

        const BaseGradType stochasticHessian = Q[it] + yy * yy.t() /
            arma::as_scalar(yy.t() * s) - Q[it] * s * s.t() *
            Q[it] / arma::as_scalar(s.t() * Q[it] * s);

        // Update aggregate Hessian approximation.
        B += (1.0 / numBatches) * (stochasticHessian - Q[it]);

        // Update aggregate Hessian-variable product.
        u += arma::reshape((1.0 / numBatches) * (stochasticHessian *
            arma::vectorise(iterate) - Q[it] * arma::vectorise(t[it])),
            u.n_rows, u.n_cols);;

        // Update aggregate gradient.
        g += (1.0 / numBatches) * (gradient - y[it]);

        // Update the function information tables.
        Q[it] = std::move(stochasticHessian);
        y[it] = std::move(gradient);
        t[it] = iterate;

        iterate = arma::reshape(stepSize * B.i() * (u.t() - arma::vectorise(g)),
            iterate.n_rows, iterate.n_cols) + (1 - stepSize) * iterate;

        terminate |= Callback::StepTaken(*this, function, iterate,
            callbacks...);
      }

      f += effectiveBatchSize;
    }

    overallObjective = 0;
    for (size_t f = 0; f < numFunctions; f += batchSize)
    {
      const size_t effectiveBatchSize = std::min(batchSize, numFunctions - f);
      const ElemType objective = function.Evaluate(iterate, f,
          effectiveBatchSize);
      overallObjective += objective;

      terminate |= Callback::Evaluate(*this, function, iterate, objective,
          callbacks...);
    }
    overallObjective /= numFunctions;

    // Output current objective function.
    Info << "IQN: iteration " << i << ", objective " << overallObjective
        << "." << std::endl;

    if (std::isnan(overallObjective) || std::isinf(overallObjective))
    {
      Warn << "IQN: converged to " << overallObjective << "; terminating"
          << " with failure.  Try a smaller step size?" << std::endl;

      Callback::EndOptimization(*this, function, iterate, callbacks...);
      return overallObjective;
    }

    if (overallObjective < tolerance)
    {
      Info << "IQN: minimized within tolerance " << tolerance << "; "
          << "terminating optimization." << std::endl;

      Callback::EndOptimization(*this, function, iterate, callbacks...);
      return overallObjective;
    }
  }

  Info << "IQN: maximum iterations (" << maxIterations << ") reached; "
      << "terminating optimization." << std::endl;

  Callback::EndOptimization(*this, function, iterate, callbacks...);
  return overallObjective;
}

} // namespace ens

#endif
