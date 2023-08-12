/**
 * @file sa_acmes_impl.hpp
 * @author Suvarsha Chennareddy
 *
 * Implementation of the Self-Adaptive Surrogate-Assisted Covariance Matrix
 * Adaptation Evolution Strategy as proposed by Ilya Loshchilov, Marc Schoenauer,
 * and Michèle Sebag in "Self-Adaptive Surrogate-Assisted Covariance Matrix 
 * Adaptation Evolution Strategy".
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CMAES_SA_ACMES_IMPL_HPP
#define ENSMALLEN_CMAES_SA_ACMES_IMPL_HPP

// In case it hasn't been included yet.
#include "sa_acmes.hpp"

#include "store_function_evaluations.hpp"
#include "ranking_svm/ranking_svm.hpp"
#include "ranking_svm/surrogate_error.hpp"
#include <ensmallen_bits/function.hpp>

/*
 Function used to extract the latest points added to the archive which will be
 used to train the surrogate model.
*/
template<typename CoordinateType>
std::vector<std::pair<CoordinateType, double>> getLastPoints(
  std::vector<std::pair<CoordinateType, double>>& inputVector,
  int subvectorSize) {

  // Check if the subvector size is valid
  if (subvectorSize <= 0 || subvectorSize > inputVector.size()) {
    std::cerr << "Invalid subvector size." << std::endl;
    return {};
  }

  std::vector<std::pair<CoordinateType, double>> subvector(
    inputVector.end() - subvectorSize, inputVector.end());
  return subvector;
}

namespace ens {

template<typename OptCMAESType, typename SurrCMAESType>
saACMES<OptCMAESType, SurrCMAESType>::saACMES(
                                  const OptCMAESType& optCMAES,
                                  const SurrCMAESType& surrCMAES,
                                  const size_t maxIterations,
                                  const size_t gStart,
                                  const double errorThreshold,
                                  const size_t maxSurrLifeLength,
                                  const size_t maxNumTrainingPoints,
                                  const size_t numTestingPoints,
                                  const double errorRelaxationConstant) :
    optCMAES(optCMAES),
    surrCMAES(surrCMAES),
    gStart(gStart),
    maxIterations(maxIterations),
    errorThreshold(errorThreshold),
    maxSurrLifeLength(maxSurrLifeLength),
    maxNumTrainingPoints(maxNumTrainingPoints),
    numTestingPoints(numTestingPoints),
    beta(errorRelaxationConstant)
{ /* Nothing to do. */ }

//! Optimize the function (minimize).
template<typename OptCMAESType, typename SurrCMAESType>
template<typename SeparableFunctionType,
         typename MatType,
         typename... CallbackTypes>
typename MatType::elem_type saACMES<OptCMAESType, SurrCMAESType>::Optimize(
    SeparableFunctionType& function,
    MatType& iterateIn,
    CallbackTypes&&... callbacks)
{

  // Make sure states will be saved during optimization.
  surrCMAES.SaveState() = optCMAES.SaveState() = true;

  // This callback will be used to maintain an archive to train the surrogate
  // model (Ranking SVM).
  StoreFunctionEvaluations<MatType> sfe;

  // Initialize the surrogate error.
  double err = 0.5;

  // Initialize the function objective.
  typename MatType::elem_type objective = 0;

  // surrCMAES will run for (at most) 2 - 1 = 1 generation. 
  surrCMAES.MaxIterations() = 2;

  // Initialize the surrogate life length.
  size_t surrLifeLength = 0.5 * maxSurrLifeLength;

  // Initialize the surrogate hyper-parameters.
  MatType surrHyperParameters(1, 2);
  surrHyperParameters(0) = 0.5 * maxNumTrainingPoints;
  surrHyperParameters(1) = 3;

  // optCMAES will run for (at most) gStart + 1 - 1 = gStart generations. 
  optCMAES.MaxIterations() = gStart + 1;
  objective = optCMAES.Optimize(function, iterateIn, sfe, callbacks...);

  // In the case that the optimization process was terminated early.
  if (optCMAES.Terminate())
  {
    Warn << "The optimization process of optCMAES terminated early." 
      << std::endl;

    Callback::EndOptimization(*this, function, iterateIn, callbacks...);
    return objective;
  }

  // Get the last few points (along with their function values) in the Archive.
  sfe.Archive() = getLastPoints(sfe.Archive(),
    maxNumTrainingPoints + numTestingPoints);

  // Initialize the surrogate model, the Ranking SVM.
  RankingSVM<MatType> rsvm(sfe.Archive(), numTestingPoints,
    arma::inv(optCMAES.CurrentCovMat()),
    optCMAES.CurrentStepSize());

  // Initialize the surrogate error function.
  SurrogateError<MatType> se(rsvm);

  size_t i = gStart;

  while (i < maxIterations) {

    // Update the ranking svm model.
    if (i != gStart) {
      rsvm.Data() = getLastPoints(sfe.Archive(), 
        maxNumTrainingPoints + numTestingPoints);
      rsvm.Kernel().CovarianceMatrixInv() = arma::inv(optCMAES.CurrentCovMat());
      rsvm.Kernel().Sigma() = optCMAES.CurrentStepSize();

      rsvm.Train((size_t) surrHyperParameters(0), surrHyperParameters(1));
    }

    // Use the surrogate ranking svm model for optimization. 
    optCMAES.MaxIterations() = surrLifeLength + 1;
    optCMAES.Optimize(rsvm, iterateIn, callbacks...);

    i += surrLifeLength;


    optCMAES.MaxIterations() = 2;
    objective = optCMAES.Optimize(function, iterateIn, sfe, callbacks...);

    if (optCMAES.Terminate())
    {
      Warn << "The optimization process of optCMAES terminated."
        << std::endl;

      Callback::EndOptimization(*this, function, iterateIn, callbacks...);
      return objective;
    }

    if (i >= maxIterations) break;

    // Calculate the error function value with the current 
    // surrogate hyper-parameters.
    typename MatType::elem_type errh =
      se.Evaluate(surrHyperParameters, 0, se.NumFunctions());

    // Compute the surrogate error.
    err = (1 - beta) * err + beta * errh;

    // Recompute next surrogate life length.
    surrLifeLength = std::floor((std::max((errorThreshold - err), 0.0) / errorThreshold) * 
      maxSurrLifeLength);

    // Optimize the surrogate error function for one generation
    // to find next set of surrogate hyper-parameters.
    surrCMAES.Optimize(se, surrHyperParameters, callbacks...);

  }

  Callback::EndOptimization(*this, function, iterateIn, callbacks...);
  return objective;
}

} // namespace ens

#endif
