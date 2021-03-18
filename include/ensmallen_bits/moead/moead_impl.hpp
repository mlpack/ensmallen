/**
 * @file moead_impl.hpp
 * @authors Utkarsh Rai, Nanubala Gnana Sai
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
                                            MatType& iterate,
                                            CallbackTypes&&... callbacks)
{
  // Sanity checks.
  if (populationSize < 3)
  {
    throw std::invalid_argument(
        "populationSize should be atleast 3 however "
        + std::to_string(populationSize) + " was detected.");
  }

  if (crossoverProb > 1.0 || crossoverProb < 0.0)
  {
    throw std::invalid_argument(
        "The parameter crossoverProb needs to be in [0,1], however "
        + std::to_string(crossoverProb) + " was detected.");
  }

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

  if (distributionIndex < 0)
  {
    throw std::invalid_argument(
        "distributionIndex should be positive, however "
        + std::to_string(distributionIndex) + " was detected.");
  }

  if (differentialWeight > 1.0 || differentialWeight < 0.0)
  {
    throw std::invalid_argument(
        "The parameter differentialWeight needs to be in [0,1], however "
        + std::to_string(differentialWeight) + " was detected.");
  }

  if (neighborProb > 1.0 || neighborProb < 0.0)
  {
    throw std::invalid_argument(
        "The parameter neighborProb needs to be in [0,1], however "
        + std::to_string(neighborProb) + " was detected.");
  }

  // Check if lower bound is a vector of a single dimension.
  if (lowerBound.n_rows == 1)
    lowerBound = lowerBound(0, 0) * arma::ones(iterate.n_rows, iterate.n_cols);

  // Check if lower bound is a vector of a single dimension.
  if (upperBound.n_rows == 1)
    upperBound = upperBound(0, 0) * arma::ones(iterate.n_rows, iterate.n_cols);

  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;

  // Number of objective functions. Represented by M in the paper.
  numObjectives = sizeof...(ArbitraryFunctionType);
  // Dimensionality of variable space. Also referred to as number of genes.
  size_t numVariables = iterate.n_rows;

  if (numObjectives < 2u)
  {
    throw std::logic_error("MOEAD::Optimize(): This is a multiobjective problem, "
        "numObjectives must be atleast 2.");
  }

  if (lowerBound.size() != upperBound.size())
  {
    throw std::logic_error("MOEAD::Optimize(): size of lowerBound and upperBound "
        "must be equal.");
  }

  if (lowerBound.size() != iterate.n_elem)
  {
    throw std::logic_error("MOEAD::Optimize(): there should be a lower bound and "
        "an upper bound for each variable in the initial point.");
  }

  for (size_t geneIdx = 0; geneIdx < lowerBound.size(); geneIdx++)
  {
    if (lowerBound[geneIdx] >= upperBound[geneIdx])
    {
      std::ostringstream ss;
      ss << "MOEAD::Optimize():" << "the lowerBound value: " << lowerBound[geneIdx]
         << " is greater than upperBound value: " << upperBound[geneIdx]
         << " at index: " << geneIdx << std::endl;

      throw std::logic_error(ss.str());
    }
  }

  bool terminate = false;

  arma::uvec shuffle;
  // The weight matrix. Each vector represents a decomposition subproblem.
  arma::Mat<ElemType> weights(numObjectives, populationSize, arma::fill::randu);
  weights += 1E-10; // Numerical stability

  // 1.2 Storing the indices of nearest neighbors of each weight vector.
  arma::umat neighborIndices(neighborSize, populationSize);
  for (size_t i = 0; i < populationSize; ++i)
  {
    // Cache the distance between weights(i) and other weights.
    arma::rowvec distances(populationSize);
    distances =
        arma::sqrt(arma::sum(arma::square(weights.col(i) - weights.each_col())));
    arma::uvec sortedIndices = arma::stable_sort_index(distances);
    // Ignore distance from self.
    neighborIndices.col(i) = sortedIndices(arma::span(1, neighborSize));
  }

  // 1.2 Random generation of the initial population.
  std::vector<MatType> population(populationSize);
  for (size_t i = 0; i < populationSize; ++i)
  {
	  population[i] =
		    arma::randu<MatType>(iterate.n_rows, iterate.n_cols) - 0.5 + iterate;

	  for (size_t geneIdx = 0; geneIdx < numVariables; ++geneIdx)
	  {
	    if (population[i][geneIdx] < lowerBound[geneIdx])
        population[i][geneIdx] = lowerBound[geneIdx];
      else if (population[i][geneIdx] > upperBound[geneIdx])
        population[i][geneIdx] = upperBound[geneIdx];
	  }
  }

  arma::mat populationFitness(numObjectives, populationSize);
  EvaluateObjectives(population, objectives, populationFitness);

  // 1.3 Initialize the ideal point z.
  arma::vec idealPoint(numObjectives);
  idealPoint.fill(std::numeric_limits<double>::max());
  for (size_t i = 0; i < numObjectives; ++i)
  {
    for (size_t j = 0; j < populationSize; ++j)
    {
      idealPoint(i) = std::min(idealPoint(i), populationFitness(i, j));
    }
  }

  terminate |= Callback::BeginOptimization(*this, objectives, iterate, callbacks...);

  // 2 The main loop.
  for (size_t g = 0; g < maxGenerations; ++g)
  {
    shuffle = arma::shuffle(
        arma::linspace<arma::uvec>(0, populationSize - 1, populationSize));
    for (size_t i : shuffle)
    {
      terminate |= Callback::StepTaken(*this, objectives, iterate, callbacks...);

      Info << "MOEA/D-DE initialized successfully. Optimization started." << std::endl;

      // 2.1 Randomly select two indices in neighborIndices(i) and use them
      // to make a child.
      size_t r1, r2, r3;
      r1 = i;
      // Randomly choose to sample from the population or the neighbors.
      bool sampleNeighbor = arma::randu() < neighborProb;
	    std::tie(r2, r3) = MatingSelection(i, neighborIndices, sampleNeighbor);

      // 2.2 - 2.3 Reproduction and Repair: Differential Operator followed by
      // Mutation.
      MatType candidate(iterate.n_rows, iterate.n_cols);
      double delta = arma::randu();
      for (size_t geneIdx = 0; geneIdx < numVariables; ++geneIdx)
      {
        if (delta < crossoverProb)
        {
          candidate[geneIdx] = population[r1][geneIdx] +
              differentialWeight * (population[r2][geneIdx] -
                  population[r3][geneIdx]);

          // Boundary conditions.
          if (candidate[geneIdx] < lowerBound[geneIdx])
          {
            candidate[geneIdx] = lowerBound[geneIdx] +
                arma::randu() * (population[r1][geneIdx] - lowerBound[geneIdx]);
          }
          if (candidate[geneIdx] > upperBound[geneIdx])
          {
            candidate[geneIdx] = upperBound[geneIdx] -
                arma::randu() * (upperBound[geneIdx] - population[r1][geneIdx]);
          }
        }

        else
          candidate[geneIdx] = population[r1][geneIdx];
        }

      Mutate(candidate, 1.0 / static_cast<double>(numVariables), lowerBound, upperBound);

      arma::vec candidateFitness(numObjectives);
      EvaluateObjectives(std::vector<MatType>{candidate}, objectives, candidateFitness);

      // 2.4 Update of ideal point.
      for (size_t idx = 0; idx < numObjectives; ++idx)
      {
        idealPoint(idx) = std::min(idealPoint(idx),
            candidateFitness(idx));
      }

      // 2.5 Update of the population.
      size_t replaceCounter = 0;
      size_t sampleSize = sampleNeighbor ? neighborSize : populationSize;

      arma::uvec idxShuffle = arma::shuffle(
          arma::linspace<arma::uvec>(0, sampleSize - 1, sampleSize));

      for (size_t idx : idxShuffle)
      {
        if (replaceCounter >= maxReplace)
          break;

        size_t pick = sampleNeighbor ? neighborIndices(idx, i) : idx;

        double candidateDecomposition = DecomposeObjectives(
            weights.col(pick), idealPoint, candidateFitness);
        double parentDecomposition = DecomposeObjectives(
            weights.col(pick), idealPoint, populationFitness.col(pick));

        if (candidateDecomposition < parentDecomposition)
        {
          population[pick] = candidate;
          populationFitness.col(pick) = candidateFitness;
          replaceCounter++;
        }
      }
    }
  }

  bestFront = std::move(population);

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
MOEAD::MatingSelection(
    const size_t subProblemIdx,
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
inline void MOEAD::Mutate(MatType& child,
    const double& mutationRate,
    const arma::vec& lowerBound,
    const arma::vec& upperBound)
{
  size_t numVariables = lowerBound.n_elem;
  double rnd, delta1, delta2, mutationPower, deltaq;
  double current, currentLower, currentUpper, value, upperDelta;

  for (size_t j = 0; j < numVariables; ++j)
  {
    double determiner = arma::randu();
    if (determiner <= mutationRate && lowerBound(j) != upperBound(j))
    {
      current = child[j];
      currentLower = lowerBound(j);
      currentUpper = upperBound(j);
      delta1 = (current - currentLower) / (currentUpper - currentLower);
      delta2 = (currentUpper - current) / (currentUpper - currentLower);
      rnd = arma::randu();
      mutationPower = 1.0 /( distributionIndex + 1.0 );
      if (rnd < 0.5)
      {
        upperDelta = 1.0 - delta1;
        value = 2.0 * rnd + (1.0 - 2.0 * rnd) *
            (std::pow(upperDelta, (distributionIndex + 1.0)));
        deltaq = std::pow(value, mutationPower) - 1.0;
      }
      else
      {
        upperDelta = 1.0 - delta2;
        value = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) *
            (std::pow(upperDelta, (distributionIndex + 1.0)));
        deltaq = 1.0 - (std::pow(value, mutationPower));
      }

      current += deltaq * (currentUpper - currentLower);
      if (current < currentLower) current = currentLower;
      if (current > currentUpper) current = currentUpper;
      child[j] = current;
    }
  }
}

//! Calculate the output for single objective function using the Tchebycheff
//! approach.
inline double MOEAD::DecomposeObjectives(const arma::vec& weights,
    const arma::vec& idealPoint,
    const arma::vec& candidateFitness)
{
  return arma::max(weights % arma::abs(candidateFitness - idealPoint));
}

//! No objectives to evaluate.
template<std::size_t I,
  typename MatType,
  typename ...ArbitraryFunctionType>
  typename std::enable_if<I == sizeof...(ArbitraryFunctionType), void>::type
MOEAD::EvaluateObjectives(
    const std::vector<MatType>&,
    std::tuple<ArbitraryFunctionType...>&,
    arma::mat &)
{
  // Nothing to do here.
}

//! Evaluate the objectives for the entire population.
template<std::size_t I,
  typename MatType,
  typename ...ArbitraryFunctionType>
  typename std::enable_if<I < sizeof...(ArbitraryFunctionType), void>::type
MOEAD::EvaluateObjectives(
    const std::vector<MatType>& population,
    std::tuple<ArbitraryFunctionType...>& objectives,
    arma::mat& calculatedObjectives)
{
  for (size_t i = 0; i < population.size(); ++i)
  {
    calculatedObjectives(I, i) = std::get<I>(objectives).Evaluate(population[i]);
    EvaluateObjectives<I + 1, MatType, ArbitraryFunctionType...>(population, objectives,
        calculatedObjectives);
  }
}
}

#endif
