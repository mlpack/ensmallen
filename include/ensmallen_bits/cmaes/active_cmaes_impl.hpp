/**
 * @file active_cmaes_impl.hpp
 * @author Marcus Edel
 * @author Suvarsha Chennareddy
 *
 * Implementation of the Active Covariance Matrix Adaptation Evolution Strategy
 * as proposed by G.A Jastrebski and D.V Arnold in "Improving Evolution
 * Strategies through Active Covariance Matrix Adaptation".
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CMAES_ACTIVE_CMAES_IMPL_HPP
#define ENSMALLEN_CMAES_ACTIVE_CMAES_IMPL_HPP

// In case it hasn't been included yet.
#include "active_cmaes.hpp"

#include "not_empty_transformation.hpp"
#include <ensmallen_bits/function.hpp>

namespace ens {

template<typename SelectionPolicyType, typename TransformationPolicyType>
ActiveCMAES<SelectionPolicyType, TransformationPolicyType>::ActiveCMAES(
                                  const size_t lambda,
                                  const TransformationPolicyType& 
                                        transformationPolicy,
                                  const size_t batchSize,
                                  const size_t maxIterations,
                                  const double tolerance,
                                  const SelectionPolicyType& selectionPolicy,
                                  double stepSizeIn) :
    lambda(lambda),
    batchSize(batchSize),
    maxIterations(maxIterations),
    tolerance(tolerance),
    selectionPolicy(selectionPolicy),
    transformationPolicy(transformationPolicy),
    stepSize(stepSizeIn)
{ /* Nothing to do. */ }

template<typename SelectionPolicyType, typename TransformationPolicyType>
ActiveCMAES<SelectionPolicyType, TransformationPolicyType>::ActiveCMAES(
                                  const size_t lambda,
                                  const double lowerBound,
                                  const double upperBound,
                                  const size_t batchSize,
                                  const size_t maxIterations,
                                  const double tolerance,
                                  const SelectionPolicyType& selectionPolicy,
                                  double stepSizeIn) :
    lambda(lambda),
    batchSize(batchSize),
    maxIterations(maxIterations),
    tolerance(tolerance),
    selectionPolicy(selectionPolicy),
    stepSize(stepSizeIn)
{
  Warn << "This is a deprecated constructor and will be removed in a "
    "future version of ensmallen" << std::endl;
  NotEmptyTransformation<TransformationPolicyType, EmptyTransformation<>> d;
  d.Assign(transformationPolicy, lowerBound, upperBound);
}

//! Optimize the function (minimize).
template<typename SelectionPolicyType, typename TransformationPolicyType>
template<typename SeparableFunctionType,
         typename MatType,
         typename... CallbackTypes>
typename MatType::elem_type ActiveCMAES<SelectionPolicyType, 
  TransformationPolicyType>::Optimize(
    SeparableFunctionType& function,
    MatType& iterateIn,
    CallbackTypes&&... callbacks)
{
  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;

  // Make sure that we have the methods that we need.  Long name...
  traits::CheckArbitrarySeparableFunctionTypeAPI<
      SeparableFunctionType, BaseMatType>();
  RequireDenseFloatingPointType<BaseMatType>();

  BaseMatType& iterate = (BaseMatType&) iterateIn;

  // Find the number of functions to use.
  const size_t numFunctions = function.NumFunctions();

  // Population size.
  if (lambda == 0)
    lambda = (4 + std::round(3 * std::log(iterate.n_elem))) * 10;

  // Parent number.
  const size_t mu = std::round(lambda / 4);

  // Recombination weight (w = 1 / (parent number)).
  const ElemType w = 1.0 / mu;

  // Number of effective solutions.
  const ElemType muEffective = mu;

  // Step size control parameters.
  BaseMatType sigma(2, 1); // sigma is vector-shaped.
  if (stepSize == 0) 
    sigma(0) = transformationPolicy.InitialStepSize();
  else 
    sigma(0) = stepSize;

  const ElemType cs = 4.0 / (iterate.n_elem + 4);
  const ElemType ds = 1 + cs;
  const ElemType enn = std::sqrt(iterate.n_elem) * (1.0 - 1.0 /
      (4.0 * iterate.n_elem) + 1.0 / (21 * std::pow(iterate.n_elem, 2)));

  // Covariance update parameters. Cumulation for distribution.
  const ElemType cc = cs;
  const ElemType ccov = 2.0 / std::pow((iterate.n_elem + std::sqrt(2)), 2);
  const ElemType beta = (4.0 * mu - 2.0) / (std::pow((iterate.n_elem + 12), 2) 
      + 4 * mu);

  std::vector<BaseMatType> mPosition(2, BaseMatType(iterate.n_rows,
      iterate.n_cols));
  mPosition[0] = iterate;

  BaseMatType step(iterate.n_rows, iterate.n_cols);
  step.zeros();

  BaseMatType transformedIterate = transformationPolicy.Transform(iterate);

  // Controls early termination of the optimization process.
  bool terminate = false;

  // Calculate the first objective function.
  ElemType currentObjective = 0;
  for (size_t f = 0; f < numFunctions; f += batchSize)
  {
    const size_t effectiveBatchSize = std::min(batchSize, numFunctions - f);
    const ElemType objective = function.Evaluate(transformedIterate, f,
        effectiveBatchSize);
    currentObjective += objective;

    terminate |= Callback::Evaluate(*this, function, transformedIterate,
        objective, callbacks...);
  }

  ElemType overallObjective = currentObjective;
  ElemType lastObjective = std::numeric_limits<ElemType>::max();

  // Population parameters.
  std::vector<BaseMatType> pStep(lambda, BaseMatType(iterate.n_rows,
      iterate.n_cols));
  std::vector<BaseMatType> pPosition(lambda, BaseMatType(iterate.n_rows,
      iterate.n_cols));
  BaseMatType pObjective(lambda, 1); // pObjective is vector-shaped.
  std::vector<BaseMatType> ps(2, BaseMatType(iterate.n_rows, iterate.n_cols));
  ps[0].zeros();
  ps[1].zeros();
  std::vector<BaseMatType> pc = ps;
  std::vector<BaseMatType> C(2, BaseMatType(iterate.n_elem, iterate.n_elem));
  C[0].eye();

  // Covariance matrix parameters.
  arma::Col<ElemType> eigval;
  BaseMatType eigvec;
  BaseMatType eigvalZero(iterate.n_elem, 1); // eigvalZero is vector-shaped.
  eigvalZero.zeros();

  // The current visitation order (sorted by population objectives).
  arma::uvec idx = arma::linspace<arma::uvec>(0, lambda - 1, lambda);

  // Now iterate!
  Callback::BeginOptimization(*this, function, transformedIterate,
      callbacks...);

  size_t idx0, idx1;

  // The number of generations to wait after the minimum loss has
  // been reached or no improvement has been made before terminating.
  size_t patience = 10 + (30 * iterate.n_elem / lambda) + 1;
  size_t steps = 0;

  for (size_t i = 1; (i != maxIterations) && !terminate; ++i)
  {
    // To keep track of where we are.
    idx0 = (i - 1) % 2;
    idx1 = i % 2;

    // Perform Cholesky decomposition. If the matrix is not positive definite,
    // add a small value and try again.
    BaseMatType covLower;
    while (!arma::chol(covLower, C[idx0], "lower"))
      C[idx0].diag() += std::numeric_limits<ElemType>::epsilon();

    arma::eig_sym(eigval, eigvec, C[idx0]);

    for (size_t j = 0; j < lambda; ++j)
    {
      if (iterate.n_rows > iterate.n_cols)
      {
        pStep[idx(j)] = covLower *
          arma::randn<BaseMatType>(iterate.n_rows, iterate.n_cols);
      }
      else
      {
        pStep[idx(j)] = arma::randn<BaseMatType>(iterate.n_rows, iterate.n_cols)
          * covLower.t();
      }

      pPosition[idx(j)] = mPosition[idx0] + sigma(idx0) * pStep[idx(j)];

      // Calculate the objective function.
      pObjective(idx(j)) = selectionPolicy.Select(function, batchSize,
          transformationPolicy.Transform(pPosition[idx(j)]), terminate,
          callbacks...);
    }

    // Sort population.
    idx = arma::sort_index(pObjective);

    step = w * pStep[idx(0)];
    for (size_t j = 1; j < mu; ++j)
      step += w * pStep[idx(j)];

    mPosition[idx1] = mPosition[idx0] + sigma(idx0) * step;

    // Calculate the objective function.
    currentObjective = selectionPolicy.Select(function, batchSize,
        transformationPolicy.Transform(mPosition[idx1]), terminate,
        callbacks...);

    // Update best parameters.
    if (currentObjective < overallObjective)
    {
      overallObjective = currentObjective;
      iterate = mPosition[idx1];

      transformedIterate = transformationPolicy.Transform(iterate);
      terminate |= Callback::StepTaken(*this, function,
          transformedIterate, callbacks...);
    }

    // Update Step Size.
    if (iterate.n_rows > iterate.n_cols)
    {
      ps[idx1] = (1 - cs) * ps[idx0] + std::sqrt(
        cs * (2 - cs) * muEffective) *
        eigvec * diagmat(1 / eigval) * eigvec.t() * step;
    }
    else
    {
      ps[idx1] = (1 - cs) * ps[idx0] + std::sqrt(
          cs * (2 - cs) * muEffective) * step *
          eigvec * diagmat(1 / eigval) * eigvec.t();
    }

    const ElemType psNorm = arma::norm(ps[idx1]);
    sigma(idx1) = sigma(idx0) * std::exp(cs / ds * (psNorm / enn - 1));

    if (std::isnan(sigma(idx1)) || sigma(idx1) > 1e14)
    {
      Warn << "The step size diverged to " << sigma(idx1) << "; "
          << "terminating with failure.  Try a smaller step size?" << std::endl;

      iterate = transformationPolicy.Transform(iterate);

      Callback::EndOptimization(*this, function, iterate, callbacks...);
      return overallObjective;
    }

    pc[idx1] = (1 - cc) * pc[idx0] + std::sqrt(cc * (2 - cc) *
        muEffective) * step;

    if (iterate.n_rows > iterate.n_cols)
    {
      C[idx1] = (1 - ccov) * C[idx0] + ccov *
          (pc[idx1] * pc[idx1].t());

      for (size_t j = 0; j < mu; ++j)
      {
        C[idx1] = C[idx1] + beta * w *
            pStep[idx(j)] * pStep[idx(j)].t();
      }

      for (size_t j = lambda - mu; j < lambda; ++j)
      {
        C[idx1] = C[idx1] - beta * w *
            pStep[idx(j)] * pStep[idx(j)].t();
      }
    }
    else
    {
      C[idx1] = (1 - ccov) * C[idx0] + ccov *
          (pc[idx1].t() * pc[idx1]);

      for (size_t j = 0; j < mu; ++j)
      {
        C[idx1] = C[idx1] + beta * w *
            pStep[idx(j)].t() * pStep[idx(j)];
      }

      for (size_t j = lambda - mu; j < lambda; ++j)
      {
        C[idx1] = C[idx1] - beta * w *
            pStep[idx(j)].t() * pStep[idx(j)];
      }
    }

    arma::eig_sym(eigval, eigvec, C[idx1]);
    const arma::uvec negativeEigval = arma::find(eigval < 0, 1);
    if (!negativeEigval.is_empty())
    {
      if (negativeEigval(0) == 0)
      {
        C[idx1].zeros();
      }
      else
      {
        C[idx1] = eigvec.cols(0, negativeEigval(0) - 1) *
            arma::diagmat(eigval.subvec(0, negativeEigval(0) - 1)) *
            eigvec.cols(0, negativeEigval(0) - 1).t();
      }
    }

    // Output current objective function.
    Info << "Active CMA-ES: iteration " << i << ", objective " << overallObjective
        << "." << std::endl;

    if (std::isnan(overallObjective) || std::isinf(overallObjective))
    {
      Warn << "Active CMA-ES: converged to " << overallObjective << "; "
          << "terminating with failure.  Try a smaller step size?" << std::endl;

      iterate = transformationPolicy.Transform(iterate);
      Callback::EndOptimization(*this, function, iterate, callbacks...);
      return overallObjective;
    }

    if (std::abs(lastObjective - overallObjective) < tolerance)
    {
      if (steps > patience)
      {
        Info << "Active CMA-ES: minimized within tolerance " << tolerance << "; "
            << "terminating optimization." << std::endl;

        iterate = transformationPolicy.Transform(iterate);
        Callback::EndOptimization(*this, function, iterate, callbacks...);
        return overallObjective;
      }
    }
    else
    {
      steps = 0;
    }

    steps++;
    lastObjective = overallObjective;
  }

  iterate = transformationPolicy.Transform(iterate);
  Callback::EndOptimization(*this, function, iterate, callbacks...);
  return overallObjective;
}

} // namespace ens

#endif
