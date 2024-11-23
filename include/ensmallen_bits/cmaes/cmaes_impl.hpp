/**
 * @file cmaes_impl.hpp
 * @author Marcus Edel
 * @author Kartik Nighania
 *
 * Implementation of the Covariance Matrix Adaptation Evolution Strategy as
 * proposed by N. Hansen et al. in "Completely Derandomized Self-Adaptation in
 * Evolution Strategies".
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CMAES_CMAES_IMPL_HPP
#define ENSMALLEN_CMAES_CMAES_IMPL_HPP

// In case it hasn't been included yet.
#include "cmaes.hpp"

#include "not_empty_transformation.hpp"
#include <ensmallen_bits/function.hpp>

namespace ens {

template<typename SelectionPolicyType, typename TransformationPolicyType>
CMAES<SelectionPolicyType, TransformationPolicyType>::CMAES(const size_t lambda,
                                  const TransformationPolicyType& 
                                        transformationPolicy,
                                  const size_t batchSize,
                                  const size_t maxIterations,
                                  const double tolerance,
                                  const SelectionPolicyType& selectionPolicy,
                                  double stepSizeIn,
                                  const int maxFunctionEvaluations,
                                  const double minObjective,
                                  const size_t toleranceConditionCov,
                                  const double toleranceNoEffectCoord,
                                  const double toleranceNoEffectAxis,
                                  const double toleranceRange,
                                  const size_t tolerancePatienceRange):
    lambda(lambda),
    batchSize(batchSize),
    maxIterations(maxIterations),
    tolerance(tolerance),
    selectionPolicy(selectionPolicy),
    transformationPolicy(transformationPolicy),
    stepSize(stepSizeIn),
    maxFunctionEvaluations(maxFunctionEvaluations),
    minObjective(minObjective),
    toleranceConditionCov(toleranceConditionCov),
    toleranceNoEffectCoord(toleranceNoEffectCoord),
    toleranceNoEffectAxis(toleranceNoEffectAxis),
    toleranceRange(toleranceRange),
    toleranceRangePatience(toleranceRangePatience)
{ /* Nothing to do. */ }

template<typename SelectionPolicyType, typename TransformationPolicyType>
CMAES<SelectionPolicyType, TransformationPolicyType>::CMAES(const size_t lambda,
                                  const double lowerBound,
                                  const double upperBound,
                                  const size_t batchSize,
                                  const size_t maxIterations,
                                  const double tolerance,
                                  const SelectionPolicyType& selectionPolicy,
                                  double stepSizeIn,
                                  const int maxFunctionEvaluations,
                                  const double minObjective,
                                  const size_t toleranceConditionCov,
                                  const double toleranceNoEffectCoord,
                                  const double toleranceNoEffectAxis,
                                  const double toleranceRange,
                                  const size_t toleranceRangePatience):
    lambda(lambda),
    batchSize(batchSize),
    maxIterations(maxIterations),
    tolerance(tolerance),
    selectionPolicy(selectionPolicy),
    stepSize(stepSizeIn),
    maxFunctionEvaluations(maxFunctionEvaluations),
    minObjective(minObjective),
    toleranceConditionCov(toleranceConditionCov),
    toleranceNoEffectCoord(toleranceNoEffectCoord),
    toleranceNoEffectAxis(toleranceNoEffectAxis),
    toleranceRange(toleranceRange),
    toleranceRangePatience(toleranceRangePatience)
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
typename MatType::elem_type CMAES<SelectionPolicyType, 
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
    lambda = (4 + std::round(3 * std::log(iterate.n_elem)));

  // Stagnation
  const size_t historySize = std::max(
      static_cast<size_t>(120 + 30 * iterate.n_elem / lambda),
      std::min(static_cast<size_t>(0.2 * maxIterations), static_cast<size_t>(20000))
  );

  std::vector<ElemType> bestFitnessHistory;
  std::vector<ElemType> medianFitnessHistory;

  // Parent weights.
  const size_t mu = std::round(lambda / 2);
  BaseMatType w = std::log(mu + 0.5) - arma::log(
      arma::linspace<BaseMatType>(0, mu - 1, mu) + 1.0);
  w /= arma::accu(w);

  // Number of effective solutions.
  const double muEffective = 1 / arma::accu(arma::pow(w, 2));

  // Step size control parameters.
  BaseMatType sigma(2, 1); // sigma is vector-shaped.
  if (stepSize == 0) 
    sigma(0) = transformationPolicy.InitialStepSize();
  else 
    sigma(0) = stepSize;

  const double cs = (muEffective + 2) / (iterate.n_elem + muEffective + 3);
  const double ds = 1 + cs + 2 * std::max(std::sqrt((muEffective - 1) /
      (iterate.n_elem + 1)) - 1, 0.0);
  const double enn = std::sqrt(iterate.n_elem) * (1.0 - 1.0 /
      (4.0 * iterate.n_elem) + 1.0 / (21 * std::pow(iterate.n_elem, 2)));

  // Covariance update parameters.
  // Cumulation for distribution.
  const double cc = (4 + muEffective / iterate.n_elem) /
      (4 + iterate.n_elem + 2 * muEffective / iterate.n_elem);
  const double h = (1.4 + 2.0 / (iterate.n_elem + 1.0)) * enn;

  const double c1 = 2 / (std::pow(iterate.n_elem + 1.3, 2) + muEffective);
  const double alphaMu = 2;
  const double cmu = std::min(1 - c1, alphaMu * (muEffective - 2 + 1 /
      muEffective) / (std::pow(iterate.n_elem + 2, 2) +
      alphaMu * muEffective / 2));

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
  arma::Col<ElemType> eigval; // TODO: might need a more general type.
  BaseMatType eigvec;
  BaseMatType eigvalZero(iterate.n_elem, 1); // eigvalZero is vector-shaped.
  eigvalZero.zeros();

  // The current visitation order (sorted by population objectives).
  arma::uvec idx = arma::linspace<arma::uvec>(0, lambda - 1, lambda);

  // Now iterate!
  Callback::BeginOptimization(*this, function, transformedIterate,
      callbacks...);

  // The number of generations to wait after the minimum loss has
  // been reached or no improvement has been made before terminating.
  size_t patience = 10 + (30 * iterate.n_elem / lambda) + 1;
  size_t steps = 0;
  size_t stepsRange = 0;

  // The min and max objective values in the range. For early termination.
  double minObjectiveInPatience = std::numeric_limits<double>::infinity();
  double maxObjectiveInPatience = -std::numeric_limits<double>::infinity();

  for (size_t i = 1; (i != maxIterations) && !terminate; ++i)
  {
    // To keep track of where we are.
    const size_t idx0 = (i - 1) % 2;
    const size_t idx1 = i % 2;

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

    step = w(0) * pStep[idx(0)];
    for (size_t j = 1; j < mu; ++j)
      step += w(j) * pStep[idx(j)];

    mPosition[idx1] = mPosition[idx0] + sigma(idx0) * step;

    // Calculate the objective function.
    currentObjective = selectionPolicy.Select(function, batchSize,
        transformationPolicy.Transform(mPosition[idx1]), terminate,
        callbacks...);

    functionEvaluations += lambda;
    
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

    // Terminate if sigma/sigma0 is above 10^4 * max(D), where C = B^T * D * B.
    if (sigma(idx1) / sigma(0) > 1e4 * std::sqrt(eigval.max()))
    {
      Info << "The step size ratio is too large; "
        << "terminating optimization. Try a smaller step size?" << std::endl;

      iterate = transformationPolicy.Transform(iterate);
      Callback::EndOptimization(*this, function, iterate, callbacks...);
      return overallObjective;
    }

    // Update covariance matrix.
    if ((psNorm / sqrt(1 - std::pow(1 - cs, 2 * i))) < h)
    {
      pc[idx1] = (1 - cc) * pc[idx0] + std::sqrt(cc * (2 - cc) *
        muEffective) * step;

      if (iterate.n_rows > iterate.n_cols)
      {
        C[idx1] = (1 - c1 - cmu) * C[idx0] + c1 *
          (pc[idx1] * pc[idx1].t());
      }
      else
      {
        C[idx1] = (1 - c1 - cmu) * C[idx0] + c1 *
          (pc[idx1].t() * pc[idx1]);
      }
    }
    else
    {
      pc[idx1] = (1 - cc) * pc[idx0];

      if (iterate.n_rows > iterate.n_cols)
      {
        C[idx1] = (1 - c1 - cmu) * C[idx0] + c1 * (pc[idx1] *
          pc[idx1].t() + (cc * (2 - cc)) * C[idx0]);
      }
      else
      {
        C[idx1] = (1 - c1 - cmu) * C[idx0] + c1 *
          (pc[idx1].t() * pc[idx1] + (cc * (2 - cc)) * C[idx0]);
      }
    }

    if (iterate.n_rows > iterate.n_cols)
    {
      for (size_t j = 0; j < mu; ++j)
      {
        C[idx1] = C[idx1] + cmu * w(j) *
          pStep[idx(j)] * pStep[idx(j)].t();
      }
    }
    else
    {
      for (size_t j = 0; j < mu; ++j)
      {
        C[idx1] = C[idx1] + cmu * w(j) *
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

    minObjectiveInPatience = std::min(minObjectiveInPatience, static_cast<double>(currentObjective));
    maxObjectiveInPatience = std::max(maxObjectiveInPatience, static_cast<double>(currentObjective));

    // Output current objective function.
    Info << "CMA-ES: iteration " << i << ", objective " << overallObjective
      << "." << std::endl;

    if (std::isnan(overallObjective) || std::isinf(overallObjective))
    {
      Warn << "CMA-ES: converged to " << overallObjective << "; "
        << "terminating with failure. Try a smaller step size?" << std::endl;

      iterate = transformationPolicy.Transform(iterate);
      Callback::EndOptimization(*this, function, iterate, callbacks...);
      return overallObjective;
    }

    if ((std::abs(lastObjective - overallObjective) < tolerance))
    {
      if (steps > patience)
      {
        Info << "CMA-ES: minimized within tolerance " << tolerance << "; "
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

    // Other termination criteria.
    // Terminate if the objective is less than the minimum objective.
    if (overallObjective < minObjective)
    {
      Info << "CMA-ES: minimized below minObjective " << minObjective << "; "
        << "terminating optimization." << std::endl;

      iterate = transformationPolicy.Transform(iterate);
      Callback::EndOptimization(*this, function, iterate, callbacks...);
      return overallObjective;
    }

    // Terminate if the range of the objective is below tolerance.
    if (maxObjectiveInPatience - minObjectiveInPatience < toleranceRange)
    {
      if (stepsRange > toleranceRangePatience)
      {
        Info << "CMA-ES: range of function values below " << toleranceRange <<
          "; terminating optimization." << std::endl;

        iterate = transformationPolicy.Transform(iterate);
        Callback::EndOptimization(*this, function, iterate, callbacks...);
        return overallObjective;
      }
      stepsRange++;
    }
    else
    {
      minObjectiveInPatience = std::numeric_limits<double>::infinity();
      maxObjectiveInPatience = -std::numeric_limits<double>::infinity();
      stepsRange = 0;
    }

    // Terminate if the maximum number of function evaluations has been reached.
    if (functionEvaluations >= maxFunctionEvaluations)
    {
      Info << "CMA-ES: maximum number of function evaluations ("
        << maxFunctionEvaluations << ") reached; "
        << "terminating optimization." << std::endl;

      iterate = transformationPolicy.Transform(iterate);
      Callback::EndOptimization(*this, function, iterate, callbacks...);
      return overallObjective;
    }

    // Check if the condition number of the covariance matrix is too high.
    if (eigval(eigval.n_elem - 1) / eigval(0) > toleranceConditionCov)
    {
      Info << "CMA-ES: covariance matrix condition number is too high; "
        << "terminating optimization." << std::endl;

      iterate = transformationPolicy.Transform(iterate);
      Callback::EndOptimization(*this, function, iterate, callbacks...);
      return overallObjective;
    }

    // Terminate if adding 0.1 std * d * b does not change the m.
    size_t i_aux = (i % iterate.n_elem);
    BaseMatType norm_eigvec = eigvec.col(i_aux) / arma::norm(eigvec.col(i_aux), 2);
    BaseMatType positionChange = arma::abs(mPosition[idx0] - (mPosition[idx0] + 
    0.1 * sigma(idx0) * std::sqrt(eigval(i_aux)) * norm_eigvec[i_aux]));

    BaseMatType axisChange = arma::abs(mPosition[idx0] - (mPosition[idx0] + positionChange));

    bool tooSmallChange = true;
    for (size_t j = 0; j < axisChange.n_elem; ++j)
    {
        if (axisChange(j) >= toleranceNoEffectAxis)
        {
            tooSmallChange = false;
            break;
        }
    }

    if (tooSmallChange)
    {
        Info << "CMA-ES: change in axis is too small; "
          << "terminating optimization." << std::endl;
        iterate = transformationPolicy.Transform(iterate);

        Callback::EndOptimization(*this, function, iterate, callbacks...);
        return overallObjective;
    }

    // Terminate if adding 0.2 std in each coordinate does not change the m.
    for (size_t j = 0; j < iterate.n_elem; ++j)
    {
      double perturbation = 0.2 * sigma(idx0) * std::sqrt(C[idx0](j, j));

      if (std::abs(mPosition[idx0](j) - (mPosition[idx0](j) + perturbation)) < toleranceNoEffectCoord)
      {
        Info << "CMA-ES: change in coordinate is too small; "
          << "terminating optimization." << std::endl;

        iterate = transformationPolicy.Transform(iterate);
        Callback::EndOptimization(*this, function, iterate, callbacks...);
        return overallObjective;
      }
    }

    // Update histories for stagnation
    bestFitnessHistory.push_back(pObjective(idx(0)));
    
    // Calculate median fitness
    std::vector<ElemType> sortedFitness(pObjective.begin(), pObjective.end());
    std::nth_element(sortedFitness.begin(), sortedFitness.begin() + lambda / 2, sortedFitness.end());
    ElemType medianFitness = sortedFitness[lambda / 2];
    medianFitnessHistory.push_back(medianFitness);

    // Check for stagnation
    if (bestFitnessHistory.size() >= historySize)
    {
      size_t third = historySize * 0.3;

      // Calculate medians for first and last 30% of history
      std::vector<ElemType> firstThirdBest(bestFitnessHistory.begin(), 
                                           bestFitnessHistory.begin() + third);
      std::vector<ElemType> lastThirdBest(bestFitnessHistory.end() - third, 
                                          bestFitnessHistory.end());
      std::vector<ElemType> firstThirdMedian(medianFitnessHistory.begin(), 
                                             medianFitnessHistory.begin() + third);
      std::vector<ElemType> lastThirdMedian(medianFitnessHistory.end() - third, 
                                            medianFitnessHistory.end());

      std::nth_element(firstThirdBest.begin(), firstThirdBest.begin() + third / 2, firstThirdBest.end());
      std::nth_element(lastThirdBest.begin(), lastThirdBest.begin() + third / 2, lastThirdBest.end());
      std::nth_element(firstThirdMedian.begin(), firstThirdMedian.begin() + third / 2, firstThirdMedian.end());
      std::nth_element(lastThirdMedian.begin(), lastThirdMedian.begin() + third / 2, lastThirdMedian.end());

      ElemType medianFirstBest = firstThirdBest[third / 2];
      ElemType medianLastBest = lastThirdBest[third / 2];
      ElemType medianFirstMedian = firstThirdMedian[third / 2];
      ElemType medianLastMedian = lastThirdMedian[third / 2];

      if (medianLastBest >= medianFirstBest && 
          medianLastMedian >= medianFirstMedian)
      {
        Info << "CMA-ES: Stagnation detected; "
            << "terminating optimization." << std::endl;
        
        iterate = transformationPolicy.Transform(iterate);
        Callback::EndOptimization(*this, function, iterate, callbacks...);
        return overallObjective;
      }

      // Remove oldest entry to maintain history size
      bestFitnessHistory.erase(bestFitnessHistory.begin());
      medianFitnessHistory.erase(medianFitnessHistory.begin());
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
