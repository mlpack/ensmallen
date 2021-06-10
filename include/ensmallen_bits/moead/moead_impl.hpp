/**
 * @file moead_impl.hpp
 * @author Utkarsh Rai
 * @author Nanubala Gnana Sai
 *
 * Implementation of the MOEA/D-DE algorithm. Used for multi-objective
 * optimization problems on arbitrary functions.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more Information.
 */

#ifndef ENSMALLEN_MOEAD_MOEAD_IMPL_HPP
#define ENSMALLEN_MOEAD_MOEAD_IMPL_HPP

#include "moead.hpp"
#include <assert.h>

namespace ens {

inline MOEAD::MOEAD(const size_t populationSize,
                    const size_t maxGenerations,
                    const double crossoverProb,
                    const size_t neighborSize,
                    const double distributionIndex,
                    const double neighborProb,
                    const double differentialWeight,
                    const size_t maxReplace,
                    const arma::vec& lowerBound,
                    const arma::vec& upperBound) :
    populationSize(populationSize),
    maxGenerations(maxGenerations),
    crossoverProb(crossoverProb),
    neighborSize(neighborSize),
    distributionIndex(distributionIndex),
    neighborProb(neighborProb),
    differentialWeight(differentialWeight),
    maxReplace(maxReplace),
    lowerBound(lowerBound),
    upperBound(upperBound),
    numObjectives(0)
  { /* Nothing to do here. */ }

inline MOEAD::MOEAD(const size_t populationSize,
                    const size_t maxGenerations,
                    const double crossoverProb,
                    const size_t neighborSize,
                    const double distributionIndex,
                    const double neighborProb,
                    const double differentialWeight,
                    const size_t maxReplace,
                    const double lowerBound,
                    const double upperBound) :
    populationSize(populationSize),
    maxGenerations(maxGenerations),
    crossoverProb(crossoverProb),
    neighborSize(neighborSize),
    distributionIndex(distributionIndex),
    neighborProb(neighborProb),
    differentialWeight(differentialWeight),
    maxReplace(maxReplace),
    lowerBound(lowerBound * arma::ones(1, 1)),
    upperBound(upperBound * arma::ones(1, 1)),
    numObjectives(0)
  { /* Nothing to do here. */ }

//! Optimize the function.
template<typename MatType,
         typename... ArbitraryFunctionType,
         typename... CallbackTypes>
typename MatType::elem_type MOEAD::Optimize(std::tuple<ArbitraryFunctionType...>& objectives,
                                            MatType& iterateIn,
                                            CallbackTypes&&... callbacks)
{
  // Population Size must be at least 3 for MOEA/D-DE to work.
  if (populationSize < 3)
  {
    throw std::logic_error("MOEA/D-DE::Optimize(): population size should be at least"
        " 3!");
  }

  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;

  BaseMatType& iterate = (BaseMatType&) iterateIn;

  if (neighborSize < 2)
  {
    throw std::invalid_argument(
        "neighborSize should be atleast 2, however "
        + std::to_string(neighborSize) + " was detected."
    );
  }

  if (neighborSize > populationSize - 1u)
  {
    std::ostringstream oss;
    oss << "MOEAD::Optimize(): " << "neighborSize is " << neighborSize
        << " but populationSize is " << populationSize << "(should be"
        << " atleast " << (neighborSize + 1u) << ")" << std::endl;
    throw std::logic_error(oss.str());
  }

  // Check if lower bound is a vector of a single dimension.
  if (lowerBound.n_rows == 1)
    lowerBound = lowerBound(0, 0) * arma::ones(iterate.n_rows, iterate.n_cols);

  // Check if lower bound is a vector of a single dimension.
  if (upperBound.n_rows == 1)
    upperBound = upperBound(0, 0) * arma::ones(iterate.n_rows, iterate.n_cols);

  // Check the dimensions of lowerBound and upperBound.
  assert(lowerBound.n_rows == iterate.n_rows && "The dimensions of "
      "lowerBound are not the same as the dimensions of iterate.");
  assert(upperBound.n_rows == iterate.n_rows && "The dimensions of "
      "upperBound are not the same as the dimensions of iterate.");

  numObjectives = sizeof...(ArbitraryFunctionType);
  size_t numVariables = iterate.n_rows;

  //! Useful temporaries for float-like comparisons.
  BaseMatType castedLowerBound = arma::conv_to<BaseMatType>::from(lowerBound);
  BaseMatType castedUpperBound = arma::conv_to<BaseMatType>::from(upperBound);

  // Controls early termination of the optimization process.
  bool terminate = false;

  arma::uvec shuffle;
  // The weight matrix. Each vector represents a decomposition subproblem (M X N).
  arma::Mat<ElemType> weights(numObjectives, populationSize, arma::fill::randu);
  weights += 1E-10; // Numerical stability

  // 1.1 Storing the indices of nearest neighbors of each weight vector.
  arma::umat neighborIndices(neighborSize, populationSize);
  for (size_t i = 0; i < populationSize; ++i)
  {
    // Cache the distance between weights(i) and other weights.
    arma::Row<ElemType> distances(populationSize);
    distances =
        arma::sqrt(arma::sum(arma::square(weights.col(i) - weights.each_col())));
    arma::uvec sortedIndices = arma::stable_sort_index(distances);
    // Ignore distance from self.
    neighborIndices.col(i) = sortedIndices(arma::span(1, neighborSize));
  }

  // 1.2 Random generation of the initial population.
  std::vector<BaseMatType> population(populationSize);
  for (BaseMatType& individual : population)
  {
      individual = arma::randu<BaseMatType>(
          iterate.n_rows, iterate.n_cols) -0.5 + iterate;

      // Constrain all genes to be within bounds.
      individual = arma::min(arma::max(individual, castedLowerBound), castedUpperBound);
  }

  Info << "MOEA/D-DE initialized successfully. Optimization started." << std::endl;

  std::vector<arma::Col<ElemType>> populationFitness(populationSize);
  std::fill(populationFitness.begin(), populationFitness.end(),
      arma::Col<ElemType>(numObjectives, arma::fill::zeros));
  EvaluateObjectives(population, objectives, populationFitness);

  // 1.3 Initialize the ideal point z.
  arma::Col<ElemType> idealPoint(numObjectives);
  idealPoint.fill(std::numeric_limits<ElemType>::max());

  for (const arma::Col<ElemType>& individualFitness : populationFitness)
    idealPoint = arma::min(idealPoint, individualFitness);

  terminate |= Callback::BeginOptimization(*this, objectives, iterate, callbacks...);

  // 2 The main loop.
  for (size_t generation = 1; generation <= maxGenerations && !terminate; ++generation)
  {
    shuffle = arma::shuffle(
        arma::linspace<arma::uvec>(0, populationSize - 1, populationSize));
    for (size_t subProblemIdx : shuffle)
    {
      // 2.1 Randomly select two indices in neighborIndices(subProblemIdx) and use them
      // to make a child.
      size_t r1, r2, r3;
      r1 = subProblemIdx;
      // Randomly choose to sample from the population or the neighbors.
      bool sampleNeighbor = arma::randu() < neighborProb;
      std::tie(r2, r3) =
	        MatingSelection(subProblemIdx, neighborIndices, sampleNeighbor);

      // 2.2 - 2.3 Reproduction and Repair: Differential Operator followed by
      // Polynomial Mutation.
      BaseMatType candidate(iterate.n_rows, iterate.n_cols);
      double delta = arma::randu();
      for (size_t geneIdx = 0; geneIdx < numVariables; ++geneIdx)
      {
        if (delta < crossoverProb)
        {
          candidate(geneIdx) = population[r1](geneIdx) +
              differentialWeight * (population[r2](geneIdx) -
                  population[r3](geneIdx));

          // Boundary conditions.
          if (candidate(geneIdx) < castedLowerBound(geneIdx))
          {
            candidate(geneIdx) = castedLowerBound(geneIdx) +
                arma::randu() * (population[r1](geneIdx) - castedLowerBound(geneIdx));
          }
          if (candidate(geneIdx) > castedUpperBound(geneIdx))
          {
            candidate(geneIdx) = castedUpperBound(geneIdx) -
                arma::randu() * (castedUpperBound(geneIdx) - population[r1](geneIdx));
          }
        }

        else
          candidate(geneIdx) = population[r1](geneIdx);
      }

      Mutate(candidate, 1.0 / static_cast<double>(numVariables), castedLowerBound, castedUpperBound);

      arma::Col<ElemType> candidateFitness(numObjectives);
      //! Creating temp vectors to pass to EvaluateObjectives.
      std::vector<BaseMatType> candidateContainer{ candidate };
      std::vector<arma::Col<ElemType>> fitnessContainer { candidateFitness };
      EvaluateObjectives(candidateContainer, objectives, fitnessContainer);
      candidateFitness = std::move(fitnessContainer[0]);
      //! Flush out the dummy containers.
      fitnessContainer.clear();
      candidateContainer.clear();

      // 2.4 Update of ideal point.
      idealPoint = arma::min(idealPoint, candidateFitness);

      // 2.5 Update of the population.
      size_t replaceCounter = 0;
      size_t sampleSize = sampleNeighbor ? neighborSize : populationSize;

      arma::uvec idxShuffle = arma::shuffle(
          arma::linspace<arma::uvec>(0, sampleSize - 1, sampleSize));

      for (size_t idx : idxShuffle)
      {
        // Preserve diversity by controlling replacement of neighbors
        // by child solution.
        if (replaceCounter >= maxReplace)
          break;

        size_t pick = sampleNeighbor ? neighborIndices(idx, subProblemIdx) : idx;

        ElemType candidateDecomposition = DecomposeObjectives<ElemType>(
            weights.col(pick), idealPoint, candidateFitness);
        ElemType parentDecomposition = DecomposeObjectives<ElemType>(
            weights.col(pick), idealPoint, populationFitness[pick]);

        if (candidateDecomposition < parentDecomposition)
        {
          population[pick] = candidate;
          populationFitness[pick] = candidateFitness;
          ++replaceCounter;
        }
      }
    } // End of pass over all subproblems.

    //  The final population itself is the best front.
    const arma::uvec frontIndices = arma::shuffle(
        arma::linspace<arma::uvec>(0, populationSize - 1, populationSize));

    terminate |= Callback::GenerationalStepTaken(*this, objectives, iterate,
        populationFitness, frontIndices, callbacks...);
  } // End of pass over all the generations.

  // Set the candidates from the Pareto Set as the output.
  paretoSet.resize(population[0].n_rows, population[0].n_cols, population.size());

  // The Pareto Front is stored, can be obtained via ParetoSet() getter.
  for (size_t solutionIdx = 0; solutionIdx < population.size(); ++solutionIdx)
  {
    paretoSet.slice(solutionIdx) =
        arma::conv_to<arma::mat>::from(population[solutionIdx]);
  }

  EvaluateObjectives(population, objectives, populationFitness);
  // Set the candidates from the Pareto Front as the output.
  paretoFront.resize(populationFitness[0].n_rows, populationFitness[0].n_cols,
      populationFitness.size());

  // The Pareto Front is stored, can be obtained via ParetoFront() getter.
  for (size_t solutionIdx = 0; solutionIdx < populationFitness.size(); ++solutionIdx)
  {
    paretoFront.slice(solutionIdx) =
        arma::conv_to<arma::mat>::from(populationFitness[solutionIdx]);
  }

  Callback::EndOptimization(*this, objectives, iterate, callbacks...);

  ElemType performance = std::numeric_limits<ElemType>::max();

  for (size_t geneIdx = 0; geneIdx < numObjectives; ++geneIdx)
  {
    if (arma::accu(populationFitness[geneIdx]) < performance)
      performance = arma::accu(populationFitness[geneIdx]);
  }

  return performance;
}

//! Randomly chooses to select from parents or neighbors.
inline std::tuple<size_t, size_t>
MOEAD::MatingSelection(size_t subProblemIdx,
                       const arma::umat& neighborIndices,
                       bool sampleNeighbor)
{
  size_t k, l;

  k = sampleNeighbor
      ? neighborIndices(
            arma::randi(arma::distr_param(0, neighborSize - 1u)), subProblemIdx)
      : arma::randi(arma::distr_param(0, populationSize - 1u));

  l = sampleNeighbor
      ? neighborIndices(
            arma::randi(arma::distr_param(0, neighborSize - 1u)), subProblemIdx)
      : arma::randi(arma::distr_param(0, populationSize - 1u));

  if (k == l)
  {
    if (k == populationSize - 1u)
      --k;
    else
      ++k;
  }

  return std::make_tuple(k, l);
}

//! Perform Polynomial mutation of the candidate.
template<typename MatType>
inline void MOEAD::Mutate(MatType& candidate,
                          double mutationRate,
                          const MatType& lowerBound,
                          const MatType& upperBound)
{
    size_t numVariables = candidate.n_rows;
    for (size_t geneIdx=0; geneIdx < numVariables; ++geneIdx)
    {
      // Should this gene be mutated?
      if (arma::randu() > mutationRate)
        continue;

      const double geneRange = upperBound[geneIdx] - lowerBound[geneIdx];
      // Normalised distance from the bounds.
      const double lowerDelta = (candidate[geneIdx] - lowerBound[geneIdx]) / geneRange;
      const double upperDelta = (upperBound[geneIdx] - candidate[geneIdx]) / geneRange;
      const double mutationPower = 1. / (distributionIndex + 1.0);
      const double rand = arma::randu();
      double value, perturbationFactor;
      if(rand < 0.5)
      {
        value = 2. * rand + (1. - 2. * rand) *
            std::pow(upperDelta, distributionIndex + 1.0);
        perturbationFactor = std::pow(value, mutationPower) - 1.;
      }
      else
      {
        value = 2. * (1. - rand) + 2.*(rand - 0.5) *
            std::pow(lowerDelta, distributionIndex + 1.0);
        perturbationFactor = 1. - std::pow(value, mutationPower);
      }

      candidate[geneIdx] += perturbationFactor * geneRange;
    }
    //! Enforce bounds.
    candidate= arma::min(arma::max(candidate, lowerBound), upperBound);
}

//! Calculate the output for single objective function using the Tchebycheff
//! approach.
template<typename ElemType>
inline ElemType MOEAD::DecomposeObjectives(const arma::Col<ElemType>& subProblemWeight,
                                           const arma::Col<ElemType>& idealPoint,
                                           const arma::Col<ElemType>& candidateFitness)
{
  return arma::max(subProblemWeight % arma::abs(candidateFitness - idealPoint));
}

//! No objectives to evaluate.
template<std::size_t I,
         typename MatType,
         typename ...ArbitraryFunctionType>
typename std::enable_if<I == sizeof...(ArbitraryFunctionType), void>::type
MOEAD::EvaluateObjectives(
    std::vector<MatType>&,
    std::tuple<ArbitraryFunctionType...>&,
    std::vector<arma::Col<typename MatType::elem_type> >&)
{
  // Nothing to do here.
}

//! Evaluate the objectives for the entire population.
template<std::size_t I,
         typename MatType,
         typename ...ArbitraryFunctionType>
typename std::enable_if<I < sizeof...(ArbitraryFunctionType), void>::type
MOEAD::EvaluateObjectives(
    std::vector<MatType>& population,
    std::tuple<ArbitraryFunctionType...>& objectives,
    std::vector<arma::Col<typename MatType::elem_type> >& calculatedObjectives)
{
  for (size_t i = 0; i < population.size(); i++)
  {
    calculatedObjectives[i](I) = std::get<I>(objectives).Evaluate(population[i]);
    EvaluateObjectives<I+1, MatType, ArbitraryFunctionType...>(population, objectives,
                                                               calculatedObjectives);
  }
}

}  // namespace ens

#endif
