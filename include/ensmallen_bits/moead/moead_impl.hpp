/**
 * @file moead_impl.hpp
 * @author Utkarsh Rai
 *
 * Implementation of the MOEA/D algorithm. Used for multi-objective
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

namespace ens {

inline MOEAD::MOEAD(const size_t populationSize,
                    const size_t numGeneration,
                    const double crossoverProb,
                    const double mutationProb,
                    const double mutationStrength,
                    const size_t neighbourhoodSize,
                    const double distributionIndex,
                    const double neighbourhoodProb,
                    const arma::vec& lowerBound,
                    const arma::vec& upperBound) :
    populationSize(populationSize),
    numGeneration(numGeneration),
    crossoverProb(crossoverProb),
    mutationProb(mutationProb),
    mutationStrength(mutationStrength),
    neighbourhoodSize(neighbourhoodSize),
    distributionIndex(distributionIndex),
    neighbourhoodProb(neighbourhoodProb),
    lowerBound(lowerBound),
    upperBound(upperBound),
    numObjectives(0)
  { /* Nothing to do here. */ }

inline MOEAD::MOEAD(const size_t populationSize,
                    const size_t numGeneration,
                    const double crossoverProb,
                    const double mutationProb,
                    const double mutationStrength,
                    const size_t neighbourhoodSize,
                    const double distributionIndex,
                    const double neighbourhoodProb,
                    const double lowerBound,
                    const double upperBound) :
    populationSize(populationSize),
    numGeneration(numGeneration),
    crossoverProb(crossoverProb),
    mutationProb(mutationProb),
    mutationStrength(mutationStrength),
    neighbourhoodSize(neighbourhoodSize),
    distributionIndex(distributionIndex),
    neighbourhoodProb(neighbourhoodProb),
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
  // Check if lower bound is a vector of a single dimension.
  if (lowerBound.n_rows == 1)
    lowerBound = lowerBound(0, 0) * arma::ones(iterate.n_rows, iterate.n_cols);

  // Check if lower bound is a vector of a single dimension.
  if (upperBound.n_rows == 1)
    upperBound = upperBound(0, 0) * arma::ones(iterate.n_rows, iterate.n_cols);
  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;

  if(lowerBound.size() != upperBound.size())
  {
    throw std::logic_error("MOEAD::Optimize(): size of lowerBound and upperBound "
        "must be equal.");
  }

  if(lowerBound.size() != iterate.n_elem)
  {
    throw std::logic_error("MOEAD::Optimize(): there should be a lower bound and "
        "an upper bound for each variable in the initial point.");
  }

  // Number of objective functions. Represented by m in the paper.
  numObjectives = sizeof...(ArbitraryFunctionType);

  // Controls early termination of the optimization process.
  bool terminate = false;

  // 1.1 The external population, non-dominated solutions.
  std::vector<MatType> externalPopulation;
  std::vector<arma::vec> externalPopulationFValue;

  arma::Col<size_t> shuffle;
  // Weight vectors, where each one of them represents a decomposition.
  arma::Mat<ElemType> weights(numObjectives, populationSize, arma::fill::randu);

  // 1.2 Storing the indices of nearest neighbours of each weight vector.
  arma::Mat<size_t> weightNeighbourIndices(populationSize, neighbourhoodSize);
  for (size_t i = 0; i < populationSize; ++i)
  {
    // To temporarily store the distance between weights(i) and each other weights.
    arma::vec distances(populationSize);
    for (size_t j = 0; j < populationSize; ++j)
      distances(j) = std::sqrt(arma::accu(arma::pow(weights.col(i) - weights.each_col(), 2)));
    arma::uvec sortedIndices = arma::stable_sort_index(distances, "descend");
    for (size_t iter = 1; iter <= neighbourhoodSize; ++iter)
      weightNeighbourIndices(i, iter - 1) = sortedIndices(iter);
  }

  // 1.3 Random generation of the initial population.
  std::vector<MatType> population(populationSize);
  for (size_t i = 0; i < populationSize; ++i)
  {
    population[i] = arma::randu<MatType>(iterate.n_rows, iterate.n_cols)
        - 0.5 + iterate;
  }

  // 1.3 F-value initialisation for the population.
  arma::mat FValue(numObjectives, populationSize);
  EvaluateObjectives(population, objectives, FValue);

  // 1.4 Initialize the ideal point z.
  arma::mat idealPoint(numObjectives, 1);
  idealPoint.fill(std::numeric_limits<ElemType>::max());
  for (size_t i = 0; i < numObjectives; ++i)
  {
    for (size_t j = 0; j < populationSize; ++j)
    {
      idealPoint(i, 0) = std::min(idealPoint(i, 0), FValue(i, j));
    }
  }

  terminate |= Callback::BeginOptimization(*this, objectives, iterate, callbacks...);

  // 2 The main loop.
  for (size_t g = 0; g < numGeneration; ++g)
  {
    shuffle = std::get<0>(objectives).Shuffle(populationSize);
    for (size_t i : shuffle)
    {
      terminate |= Callback::StepTaken(*this, objectives, iterate, callbacks...);

      // 2.1 Randomly select two indices in weightNeighbourIndices(i) and use them
      // to make a child.
      size_t k, l;
      if(arma::randu() < neighbourhoodProb)
      {
        k = weightNeighbourIndices(i, arma::randi(arma::distr_param(0,  neighbourhoodSize - 1)));
        l = weightNeighbourIndices(i, arma::randi(arma::distr_param(0,  neighbourhoodSize - 1)));
        if(k == l)
        {
          if(k == neighbourhoodSize - 1)
            --k;
          else
            ++k;
        }
      }
      else
      {
        k = arma::randi(arma::distr_param(0, populationSize - 1));
        l = arma::randi(arma::distr_param(0, populationSize - 1));
        if(k == l)
        {
          if(k == populationSize - 1)
            --k;
          else
            ++k;
        }
      }
      std::vector<MatType> candidate(1);
      double determiner1 = arma::randu();
      if(determiner1 < crossoverProb)
      {
        candidate[0].resize(iterate.n_rows, iterate.n_cols);
        for (size_t idx = 0;idx < iterate.n_rows; ++idx)
        {
          double determiner2 = arma::randu();
          if (determiner2 < 0.5)
            candidate[0][idx] = population[k][idx];
          else
            candidate[0][idx] = population[l][idx];
          if(candidate[0][idx] < lowerBound(idx))
            candidate[0][idx] = lowerBound(idx);
          if(candidate[0][idx]>upperBound(idx))
            candidate[0][idx] = upperBound(idx);
        }
      }
      else
        candidate[0] = population[i];

      // 2.2 Improve the child.
      Mutate(candidate[0], 1 / numObjectives, lowerBound, upperBound);

      // Store solution for candidate.
      arma::mat evaluatedCandidate(numObjectives, 1);
      EvaluateObjectives(candidate, objectives, evaluatedCandidate);

      // 2.3 Update of ideal point.
      for (size_t idx = 0;idx < numObjectives;++idx)
      {
        idealPoint(idx, 0) = std::min(idealPoint(idx, 0),
            evaluatedCandidate(idx, 0));
      }

      // 2.4 Update of the neighbouring solutions.
      for (size_t idx = 0;idx < neighbourhoodSize;++idx)
      {
        if (DecomposedSingleObjective(weights.col(weightNeighbourIndices(i, idx)),
              idealPoint.col(0), evaluatedCandidate.col(0))
            <= DecomposedSingleObjective(
               weights.col(weightNeighbourIndices(i,idx)),
               idealPoint.col(0), FValue.col(weightNeighbourIndices(i, idx))))
        {
          population.at(weightNeighbourIndices(i, idx)) = candidate[0];
          FValue.col(weightNeighbourIndices(i, idx)) = evaluatedCandidate.col(0);
        }
      }

      // 2.5 Updating External Population.
      if (!externalPopulation.empty())
      {
        arma::mat first(numObjectives, 1);
        auto df = [&](MatType firstMat) -> bool
        {
          std::vector<MatType> wrapperFirst(1), wrapperSecond(1);
          wrapperFirst[0] = firstMat;
          EvaluateObjectives(wrapperFirst, objectives, first);
          return Dominates(evaluatedCandidate.col(0), first.col(0));
        };
        //! Remove the part that is dominated by candidate.
        externalPopulation.erase(
            std::remove_if(
              externalPopulation.begin(),
              externalPopulation.end(),
              df), externalPopulation.end());

        //! Check if any of the remaining members of external population dominate
        //! candidate.
        bool flag = 0;
        std::vector<MatType> wrapperFirst(1);
        for (size_t idx = 0; idx < externalPopulation.size(); ++idx)
        {
          wrapperFirst[0]=externalPopulation[idx];
          first.clear();
          first.resize(numObjectives, 1);
          EvaluateObjectives(wrapperFirst, objectives, first);
          if (Dominates(first.col(0), evaluatedCandidate.col(0)))
          {
            flag = 1;
            break;
          }
        }
        if (flag == 0)
        {
          externalPopulation.push_back(candidate[0]);
          externalPopulationFValue.push_back(evaluatedCandidate.col(0));
        }
      }
      else
      {
        externalPopulation.push_back(candidate[0]);
        externalPopulationFValue.push_back(evaluatedCandidate.col(0));
      }
    }
  }
  bestFront = std::move(externalPopulation);

  Callback::EndOptimization(*this, objectives, iterate, callbacks...);

  ElemType performance = std::numeric_limits<ElemType>::max();
  for (arma::Col<ElemType> objective: externalPopulationFValue)
  {
    if (arma::accu(objective) < performance)
      performance = arma::accu(objective);
  }

  return performance;
}

//! Perform mutation of the candidate.
template<typename MatType>
inline void MOEAD::Mutate(MatType& child,
    const double& rate,
    const arma::vec& lowerBound,
    const arma::vec& upperBound)
{
  size_t numVariables = lowerBound.n_elem;
  double rnd, delta1, delta2, mutationPower, deltaq;
  double current, currentLower, currentUpper, value, upperDelta;

  for (size_t j = 0; j < numVariables; ++j)
  {
    double determiner = arma::randu();
    if (determiner <= rate)
    {
      current = child[j];
      currentLower = lowerBound(j);
      currentUpper = upperBound(j);
      delta1 = (current - currentLower) / (currentUpper - currentLower);
      delta2 = (currentUpper - current) / (currentUpper - currentLower);    
      rnd = arma::randu();
      mutationPower= 1 / distributionIndex;
      if (rnd <= 0.5)
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
            (pow(upperDelta, (distributionIndex + 1.0)));
        deltaq = 1.0 - (pow(value, mutationPower));
      }
      current = current + deltaq * (currentUpper - currentLower);
      if (current < currentLower) current = currentLower;
      if (current > currentUpper) current = currentUpper;
      child[j] = current;
    }
  }
}

//! Calculate the output for single objective function using the Tchebycheff
//! approach.
inline double MOEAD::DecomposedSingleObjective(const arma::vec& weights,
    const arma::vec& idealPoint,
    const arma::vec& evaluatedCandidate)
{
  return arma::min(weights % arma::abs(evaluatedCandidate - idealPoint));
}

inline bool MOEAD::Dominates(const arma::vec& first,
    const arma::vec& second)
{
  int flag = 0;
  for (size_t i = 0; i < numObjectives; ++i)
  {
    if (first(i) > second(i))
      return false;

    if(first(i) < second(i))
      flag = 1;
  }
  if (flag)
    return true;

  return false;
}

//! No objectives to evaluate.
template<std::size_t I,
  typename MatType,
  typename ...ArbitraryFunctionType>
  typename std::enable_if<I == sizeof...(ArbitraryFunctionType), void>::type
MOEAD::EvaluateObjectives(
    std::vector<MatType>&,
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
    std::vector<MatType>& population,
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

