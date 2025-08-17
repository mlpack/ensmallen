/**
 * @file moead_impl.hpp
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
template <typename InitPolicyType,
          typename DecompPolicyType,
          typename MatType,
          typename ColType,
          typename CubeType>
inline MOEADType<InitPolicyType, DecompPolicyType, MatType, ColType, CubeType>::
MOEADType(
    const size_t populationSize,
    const size_t maxGenerations,
    const double crossoverProb,
    const double neighborProb,
    const size_t neighborSize,
    const double distributionIndex,
    const double differentialWeight,
    const size_t maxReplace,
    const double epsilon,
    const ColType& lowerBound,
    const ColType& upperBound,
    const InitPolicyType initPolicy,
    const DecompPolicyType decompPolicy) :
    populationSize(populationSize),
    maxGenerations(maxGenerations),
    crossoverProb(crossoverProb),
    neighborProb(neighborProb),
    neighborSize(neighborSize),
    distributionIndex(distributionIndex),
    differentialWeight(differentialWeight),
    maxReplace(maxReplace),
    epsilon(epsilon),
    lowerBound(lowerBound),
    upperBound(upperBound),
    initPolicy(initPolicy),
    decompPolicy(decompPolicy)
  { /* Nothing to do here. */ }

template <typename InitPolicyType,
          typename DecompPolicyType,
          typename MatType,
          typename ColType,
          typename CubeType>
inline MOEADType<InitPolicyType, DecompPolicyType, MatType, ColType, CubeType>::
MOEADType(
    const size_t populationSize,
    const size_t maxGenerations,
    const double crossoverProb,
    const double neighborProb,
    const size_t neighborSize,
    const double distributionIndex,
    const double differentialWeight,
    const size_t maxReplace,
    const double epsilon,
    const double lowerBound,
    const double upperBound,
    const InitPolicyType initPolicy,
    const DecompPolicyType decompPolicy) :
    populationSize(populationSize),
    maxGenerations(maxGenerations),
    crossoverProb(crossoverProb),
    neighborProb(neighborProb),
    neighborSize(neighborSize),
    distributionIndex(distributionIndex),
    differentialWeight(differentialWeight),
    maxReplace(maxReplace),
    epsilon(epsilon),
    lowerBound({ lowerBound })
    lowerBound({ upperBound })
    initPolicy(initPolicy),
    decompPolicy(decompPolicy)
  { /* Nothing to do here. */ }

//! Optimize the function.
template <typename InitPolicyType,
          typename DecompPolicyType,
          typename MatType,
          typename ColType,
          typename CubeType>
template<typename InputMatType,
         typename... ArbitraryFunctionType,
         typename... CallbackTypes>
typename InputMatType::elem_type MOEADType<
    InitPolicyType, DecompPolicyType, MatType, ColType, CubeType>::
Optimize(std::tuple<ArbitraryFunctionType...>& objectives,
         InputMatType& iterateIn,
         CallbackTypes&&... callbacks)
{
  // Population Size must be at least 3 for MOEA/D-DE to work.
  if (populationSize < 3)
  {
    throw std::logic_error(
        "MOEA/D-DE::Optimize(): population size should be at least 3!");
  }

  // Convenience typedefs.
  typedef typename InputMatType::elem_type ElemType;
  typedef typename MatTypeTraits<InputMatType>::BaseMatType BaseMatType;

  typedef typename ForwardType<MatType>::uvec UVecType;
  typedef typename ForwardType<MatType>::umat UMatType;
  typedef typename ForwardType<MatType>::brow BaseRowType;

  BaseMatType& iterate = (BaseMatType&) iterateIn;

  // Make sure that we have the methods that we need.  Long name...
  traits::CheckArbitraryFunctionTypeAPI<ArbitraryFunctionType...,
      BaseMatType>();
  RequireDenseFloatingPointType<BaseMatType>();

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

  BaseMatType castedLowerBound;
  BaseMatType castedUpperBound;

  // Check if lower/upper bound is a vector of a single dimension.
  if (lowerBound.n_elem == 1)
  {
    castedLowerBound = lowerBound(0) * BaseMatType(
        iterate.n_rows, iterate.n_cols, GetFillType<MatType>::ones);

    castedUpperBound = upperBound(0) * BaseMatType(
        iterate.n_rows, iterate.n_cols, GetFillType<MatType>::ones);
  }
  else
  {
    castedLowerBound = conv_to<BaseMatType>::from(lowerBound);
    castedUpperBound = conv_to<BaseMatType>::from(upperBound);
  }

  // Check the dimensions of lowerBound and upperBound.
  assert(lowerBound.n_rows == iterate.n_rows && "The dimensions of "
      "lowerBound are not the same as the dimensions of iterate.");
  assert(upperBound.n_rows == iterate.n_rows && "The dimensions of "
      "upperBound are not the same as the dimensions of iterate.");

  const size_t numObjectives = sizeof...(ArbitraryFunctionType);
  const size_t numVariables = iterate.n_rows;

  // Controls early termination of the optimization process.
  bool terminate = false;

  // The weight matrix. Each vector represents a decomposition
  // subproblem (M X N).
  const BaseMatType weights = initPolicy.template Generate<BaseMatType>(
      numObjectives, populationSize, epsilon);

  // 1.1 Storing the indices of nearest neighbors of each weight vector.
  UMatType neighborIndices(neighborSize, populationSize);
  for (size_t i = 0; i < populationSize; ++i)
  {
    // Cache the distance between weights[i] and other weights.
    const BaseRowType distances =
        conv_to<BaseRowType>::from(
        sqrt(sum(square(weights.col(i) - weights.each_col()))));
    UVecType sortedIndices = stable_sort_index(distances);
    // Ignore distance from self.
    neighborIndices.col(i) = sortedIndices(
          typename GetProxyType<UVecType>::span(1, neighborSize), 0);
  }

  // 1.2 Random generation of the initial population.
  std::vector<BaseMatType> population(populationSize);
  for (BaseMatType& individual : population)
  {
    individual = randu<BaseMatType>(
        iterate.n_rows, iterate.n_cols) - 0.5 + iterate;

    // Constrain all genes to be within bounds.
    individual = min(max(individual, castedLowerBound), castedUpperBound);
  }

  Info << "MOEA/D-DE initialized successfully. Optimization started."
      << std::endl;

  std::vector<ColType> populationFitness(populationSize);
  for (size_t i = 0; i < populationSize; ++i)
  {
    populationFitness[i].set_size(numObjectives);
    populationFitness[i].zeros();
  }
  EvaluateObjectives(population, objectives, populationFitness);

  // 1.3 Initialize the ideal point z.
  ColType idealPoint(numObjectives);
  idealPoint.fill(std::numeric_limits<ElemType>::max());

  for (const ColType& individualFitness : populationFitness)
    idealPoint = min(idealPoint, individualFitness);

  Callback::BeginOptimization(*this, objectives, iterate, callbacks...);

  // 2 The main loop.
  for (size_t generation = 1;
      generation <= maxGenerations && !terminate; ++generation)
  {
    // Shuffle indexes of subproblems.
    const UVecType shuffleTemp = shuffle(
        linspace<UVecType>(0, populationSize - 1, populationSize));

    for (size_t i = 0; i < shuffleTemp.n_elem; ++i)
    {
      const size_t subProblemIdx = shuffleTemp(i);
      // 2.1 Randomly select two indices in neighborIndices[subProblemIdx]
      // and use them to make a child.
      size_t r1, r2, r3;
      r1 = subProblemIdx;
      // Randomly choose to sample from the population or the neighbors.
      const bool sampleNeighbor = arma::randu() < neighborProb;
      std::tie(r2, r3) =
          Mating(subProblemIdx, neighborIndices, sampleNeighbor);

      // 2.2 - 2.3 Reproduction and Repair: Differential Operator followed by
      // Polynomial Mutation.
      BaseMatType candidate(iterate.n_rows, iterate.n_cols);

      for (size_t geneIdx = 0; geneIdx < numVariables; ++geneIdx)
      {
        if (arma::randu() < crossoverProb)
        {
          candidate(geneIdx) = population[r1](geneIdx) +
              differentialWeight * (population[r2](geneIdx) -
              population[r3](geneIdx));

          // Boundary conditions.
          if (candidate(geneIdx) < castedLowerBound(geneIdx))
          {
            candidate(geneIdx) = castedLowerBound(geneIdx) + arma::randu() *
                (population[r1](geneIdx) - castedLowerBound(geneIdx));
          }
          if (candidate(geneIdx) > castedUpperBound(geneIdx))
          {
            candidate(geneIdx) = castedUpperBound(geneIdx) - arma::randu() *
                (castedUpperBound(geneIdx) - population[r1](geneIdx));
          }
        }
        else
          candidate(geneIdx) = population[r1](geneIdx);
      }

      Mutate(candidate, 1.0 / static_cast<double>(numVariables),
          castedLowerBound, castedUpperBound);

      ColType candidateFitness(numObjectives);
      //! Creating temp vectors to pass to EvaluateObjectives.
      std::vector<BaseMatType> candidateContainer { candidate };
      std::vector<ColType> fitnessContainer { candidateFitness };
      EvaluateObjectives(candidateContainer, objectives, fitnessContainer);
      candidateFitness = std::move(fitnessContainer[0]);
      //! Flush out the dummy containers.
      fitnessContainer.clear();
      candidateContainer.clear();

      // 2.4 Update of ideal point.
      idealPoint = min(idealPoint, candidateFitness);

      // 2.5 Update of the population.
      size_t replaceCounter = 0;
      const size_t sampleSize = sampleNeighbor ? neighborSize : populationSize;

      const arma::uvec idxShuffle = shuffle(
          linspace<arma::uvec>(0, sampleSize - 1, sampleSize));

      for (size_t i = 0; i < idxShuffle.n_elem; ++i)
      {
        const size_t idx = idxShuffle(i);
        // Preserve diversity by controlling replacement of neighbors
        // by child solution.
        if (replaceCounter >= maxReplace)
          break;

        const size_t pick = sampleNeighbor ?
            neighborIndices(idx, subProblemIdx) : idx;

        const ElemType candidateDecomposition = decompPolicy.template
            Apply<ColType>(conv_to<ColType>::from(weights.col(pick)),
            idealPoint, candidateFitness);
        const ElemType parentDecomposition = decompPolicy.template
            Apply<ColType>(conv_to<ColType>::from(weights.col(pick)),
            idealPoint, populationFitness[pick]);

        if (candidateDecomposition < parentDecomposition)
        {
          population[pick] = candidate;
          populationFitness[pick] = candidateFitness;
          ++replaceCounter;
        }
      }
    } // End of pass over all subproblems.

    //  The final population itself is the best front.
    const std::vector<UVecType> frontIndices { shuffle(
        linspace<UVecType>(0, populationSize - 1, populationSize)) };

    terminate |= Callback::GenerationalStepTaken(*this, objectives, iterate,
        populationFitness, frontIndices, callbacks...);
  } // End of pass over all the generations.

  // Set the candidates from the Pareto Set as the output.
  paretoSet.set_size(
      population[0].n_rows, population[0].n_cols, population.size());

  // The Pareto Front is stored, can be obtained via ParetoSet() getter.
  for (size_t solutionIdx = 0; solutionIdx < population.size(); ++solutionIdx)
  {
    paretoSet.slice(solutionIdx) =
        conv_to<MatType>::from(population[solutionIdx]);
  }

  // Set the candidates from the Pareto Front as the output.
  paretoFront.set_size(populationFitness[0].n_rows, populationFitness[0].n_cols,
      populationFitness.size());

  // The Pareto Front is stored, can be obtained via ParetoFront() getter.
  for (size_t solutionIdx = 0;
      solutionIdx < populationFitness.size(); ++solutionIdx)
  {
    paretoFront.slice(solutionIdx) =
        conv_to<MatType>::from(populationFitness[solutionIdx]);
  }

  // Assign iterate to first element of the Pareto Set.
  iterate = population[0];

  Callback::EndOptimization(*this, objectives, iterate, callbacks...);

  ElemType performance = std::numeric_limits<ElemType>::max();

  for (size_t geneIdx = 0; geneIdx < numObjectives; ++geneIdx)
  {
    if (accu(populationFitness[geneIdx]) < performance)
      performance = accu(populationFitness[geneIdx]);
  }

  return performance;
}

//! Randomly chooses to select from parents or neighbors.
template <typename InitPolicyType,
          typename DecompPolicyType,
          typename MatType,
          typename ColType,
          typename CubeType>
template<typename IndexMatType>
inline std::tuple<size_t, size_t>
MOEADType<InitPolicyType, DecompPolicyType, MatType, ColType, CubeType>::
Mating(size_t subProblemIdx,
       const IndexMatType& neighborIndices,
       bool sampleNeighbor)
{
  //! Indexes of two points from the sample space.
  size_t pointA = sampleNeighbor
      ? neighborIndices(
            arma::randi(arma::distr_param(0, neighborSize - 1u)), subProblemIdx)
      : arma::randi(arma::distr_param(0, populationSize - 1u));

  size_t pointB = sampleNeighbor
      ? neighborIndices(
            arma::randi(arma::distr_param(0, neighborSize - 1u)), subProblemIdx)
      : arma::randi(arma::distr_param(0, populationSize - 1u));

  //! If the sampled points are equal, then modify one of them
  //! within reasonable bounds.
  if (pointA == pointB)
  {
    if (pointA == populationSize - 1u)
      --pointA;
    else
      ++pointA;
  }

  return std::make_tuple(pointA, pointB);
}

//! Perform Polynomial mutation of the candidate.
template <typename InitPolicyType,
          typename DecompPolicyType,
          typename MatType,
          typename ColType,
          typename CubeType>
template<typename InputMatType>
inline void MOEADType<
    InitPolicyType, DecompPolicyType, MatType, ColType, CubeType>::Mutate(
    InputMatType& candidate,
    double mutationRate,
    const InputMatType& lowerBound,
    const InputMatType& upperBound)
{
    const size_t numVariables = candidate.n_rows;
    for (size_t geneIdx = 0; geneIdx < numVariables; ++geneIdx)
    {
      // Should this gene be mutated?
      if (arma::randu() > mutationRate)
        continue;

      const double geneRange = upperBound(geneIdx) - lowerBound(geneIdx);
      // Normalised distance from the bounds.
      const double lowerDelta = (candidate(geneIdx) - lowerBound(geneIdx)) /
          geneRange;
      const double upperDelta = (upperBound(geneIdx) - candidate(geneIdx)) /
          geneRange;
      const double mutationPower = 1. / (distributionIndex + 1.0);
      const double rand = arma::randu();
      double value, perturbationFactor;
      if (rand < 0.5)
      {
        value = 2.0 * rand + (1.0 - 2.0 * rand) *
            std::pow(upperDelta, distributionIndex + 1.0);
        perturbationFactor = std::pow(value, mutationPower) - 1.0;
      }
      else
      {
        value = 2.0 * (1.0 - rand) + 2.0 *(rand - 0.5) *
            std::pow(lowerDelta, distributionIndex + 1.0);
        perturbationFactor = 1.0 - std::pow(value, mutationPower);
      }

      candidate(geneIdx) += perturbationFactor * geneRange;
    }
    //! Enforce bounds.
    candidate = min(max(candidate, lowerBound), upperBound);
}

//! No objectives to evaluate.
template <typename InitPolicyType,
          typename DecompPolicyType,
          typename MatType,
          typename ColType,
          typename CubeType>
template<std::size_t I,
         typename InputMatType,
         typename ...ArbitraryFunctionType>
typename std::enable_if<I == sizeof...(ArbitraryFunctionType), void>::type
MOEADType<InitPolicyType, DecompPolicyType, MatType, ColType, CubeType>::
EvaluateObjectives(
    std::vector<InputMatType>&,
    std::tuple<ArbitraryFunctionType...>&,
    std::vector<ColType>&)
{
 // Nothing to do here.
}

//! Evaluate the objectives for the entire population.
template <typename InitPolicyType,
          typename DecompPolicyType,
          typename MatType,
          typename ColType,
          typename CubeType>
template<std::size_t I,
         typename InputMatType,
         typename ...ArbitraryFunctionType>
typename std::enable_if<I < sizeof...(ArbitraryFunctionType), void>::type
MOEADType<InitPolicyType, DecompPolicyType, MatType, ColType, CubeType>::
EvaluateObjectives(
    std::vector<InputMatType>& population,
    std::tuple<ArbitraryFunctionType...>& objectives,
    std::vector<ColType>& calculatedObjectives)
{
  for (size_t i = 0; i < population.size(); i++)
  {
    calculatedObjectives[i](I) = std::get<I>(objectives).Evaluate(population[i]);
    EvaluateObjectives<I+1, InputMatType, ArbitraryFunctionType...>(
        population, objectives, calculatedObjectives);
  }
}

}  // namespace ens

#endif
