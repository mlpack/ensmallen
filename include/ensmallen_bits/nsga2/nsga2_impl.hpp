/**
 * @file nsga2_impl.hpp
 * @author Sayan Goswami
 *
 * Implementation of the NSGA-II algorithm. Used for multi-objective
 * optimization problems on arbitrary functions.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more Information.
 */

#ifndef ENSMALLEN_NSGA2_NSGA2_IMPL_HPP
#define ENSMALLEN_NSGA2_NSGA2_IMPL_HPP

#include "nsga2.hpp"
#include <assert.h>

namespace ens {

inline NSGA2::NSGA2(const size_t populationSize,
                    const size_t maxGenerations,
                    const double crossoverProb,
                    const double mutationProb,
                    const double mutationStrength,
                    const double epsilon,
                    const arma::vec& lowerBound,
                    const arma::vec& upperBound) :
    populationSize(populationSize),
    maxGenerations(maxGenerations),
    crossoverProb(crossoverProb),
    mutationProb(mutationProb),
    mutationStrength(mutationStrength),
    epsilon(epsilon),
    lowerBound(lowerBound),
    upperBound(upperBound)
{ /* Nothing to do here. */ }

inline NSGA2::NSGA2(const size_t populationSize,
                    const size_t maxGenerations,
                    const double crossoverProb,
                    const double mutationProb,
                    const double mutationStrength,
                    const double epsilon,
                    const double lowerBound,
                    const double upperBound) :
    populationSize(populationSize),
    maxGenerations(maxGenerations),
    crossoverProb(crossoverProb),
    mutationProb(mutationProb),
    mutationStrength(mutationStrength),
    epsilon(epsilon),
    lowerBound(lowerBound * arma::ones(1, 1)),
    upperBound(upperBound * arma::ones(1, 1))
{ /* Nothing to do here. */ }

//! Optimize the function.
template<typename MatType,
         typename... ArbitraryFunctionType,
         typename... CallbackTypes>
typename MatType::elem_type NSGA2::Optimize(
    std::tuple<ArbitraryFunctionType...>& objectives,
    MatType& iterate,
    CallbackTypes&&... callbacks)
{
  // Make sure for evolution to work at least four candidates are present.
  if (populationSize < 4 && populationSize % 4 != 0)
  {
    throw std::logic_error("NSGA2::Optimize(): population size should be at"
        " least 4, and, a multiple of 4!");
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

  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;

  numObjectives = sizeof...(ArbitraryFunctionType);
  numVariables = iterate.n_rows;

  // Cache calculated objectives.
  std::vector<arma::Col<ElemType> > calculatedObjectives;
  // Pre-allocate space for the calculated objectives.
  calculatedObjectives.resize(populationSize);

  // Population size reserved to 2 * populationSize + 1 to accommodate
  // for the size of intermediate candidate population.
  std::vector<MatType> population;
  population.reserve(2 * populationSize + 1);

  // Pareto fronts, initialized during non-dominated sorting.
  std::vector<std::vector<size_t> > fronts;
  // Initialised in CrowdingDistanceAssignment.
  std::vector<double> crowdingDistance;
  // Initialised during non-dominated sorting.
  std::vector<size_t> ranks;

  // Controls early termination of the optimization process.
  bool terminate = false;

  // Generate the population based on a uniform distribution around the given
  // starting point.
  for (size_t i = 0; i < populationSize; i++)
  {
    population.push_back(arma::randu<MatType>(iterate.n_rows,
        iterate.n_cols) - 0.5 + iterate);
  }

  Info << "NSGA2 initialized successfully. Optimization started." << std::endl;

  // Evaluate the fitness before optimization.
  for (size_t i = 0; i < population.size(); i++)
    calculatedObjectives[i] = arma::Col<ElemType>(numObjectives, arma::fill::zeros);
  EvaluateObjectives(population, objectives, calculatedObjectives);

  // Iterate until maximum number of generations is obtained.
  terminate |= Callback::BeginOptimization(*this, objectives, iterate, callbacks...);

  for (size_t generation = 1; generation <= maxGenerations && !terminate; generation++)
  {
    Info << "NSGA2: iteration " << generation << "." << std::endl;
    terminate |= Callback::StepTaken(*this, objectives, iterate, callbacks...);

    // Create new population of candidate from the present elite population.
    // Have P_t, generate G_t using P_t.
    BinaryTournamentSelection(population, lowerBound, upperBound);

    // Evaluate the objectives for the new population.
    calculatedObjectives.resize(population.size());
    for (size_t i = 0; i < population.size(); i++)
      calculatedObjectives[i] = arma::Col<ElemType>(numObjectives, arma::fill::zeros);
    EvaluateObjectives(population, objectives, calculatedObjectives);

    // Perform fast non dominated sort on P_t ∪ G_t.
    ranks.resize(population.size());
    FastNonDominatedSort<MatType>(fronts, ranks, calculatedObjectives);

    // Perform crowding distance assignment.
    crowdingDistance.resize(population.size());

    for (size_t fNum = 0; fNum < fronts.size(); fNum++)
    {
      CrowdingDistanceAssignment(fronts[fNum], crowdingDistance);
    }

    // Sort based on crowding distance.
    std::sort(population.begin(), population.end(),
              [this, ranks, crowdingDistance, population](MatType candidateP,
                                                          MatType candidateQ)
              {
                size_t idxP, idxQ;
                for (size_t i = 0; i < population.size(); i++)
                {
                  if (arma::approx_equal(population[i], candidateP, "absdiff", epsilon))
                    idxP = i;

                  if (arma::approx_equal(population[i], candidateQ, "absdiff", epsilon))
                    idxQ = i;
                }

                return CrowdingOperator(idxP, idxQ, ranks, crowdingDistance);
              }
    );

    // Yield a new population P_{t+1} of size populationSize.
    population.resize(populationSize);
  }

  // Set the candidates from the best front as the output.
  std::vector<MatType> front;

  for (size_t f: fronts[0])
    front.push_back(population[f]);

  // bestFront is stored, can be obtained by the Front() getter.
  bestFront = front;

  // Assign iterate to first element of the best front.
  iterate = bestFront[0];

  Callback::EndOptimization(*this, objectives, iterate, callbacks...);

  ElemType performance = std::numeric_limits<ElemType>::max();

  for(arma::Col<ElemType> objective: calculatedObjectives)
    if (arma::accu(objective) < performance)
      performance = arma::accu(objective);

  return performance;
}

//! No objectives to evaluate.
template<std::size_t I,
         typename MatType,
         typename ...ArbitraryFunctionType>
typename std::enable_if<I == sizeof...(ArbitraryFunctionType), void>::type
NSGA2::EvaluateObjectives(
    std::vector<MatType>&,
    std::tuple<ArbitraryFunctionType...>&,
    std::vector<arma::Col<double> >&)
{
  // Nothing to do here.
}

//! Evaluate the objectives for the entire population.
template<std::size_t I,
         typename MatType,
         typename ...ArbitraryFunctionType>
typename std::enable_if<I < sizeof...(ArbitraryFunctionType), void>::type
NSGA2::EvaluateObjectives(
    std::vector<MatType>& population,
    std::tuple<ArbitraryFunctionType...>& objectives,
    std::vector<arma::Col<double> >& calculatedObjectives)
{
  for (size_t i = 0; i < populationSize; i++)
  {
    calculatedObjectives[i](I) = std::get<I>(objectives).Evaluate(population[i]);
    EvaluateObjectives<I+1, MatType, ArbitraryFunctionType...>(population, objectives,
                                                               calculatedObjectives);
  }
}

//! Reproduce and generate new candidates.
template<typename MatType>
inline void NSGA2::BinaryTournamentSelection(std::vector<MatType>& population,
                                             const arma::vec& lowerBound,
                                             const arma::vec& upperBound)
{
  std::vector<MatType> children;

  while (children.size() < population.size())
  {
    // Choose two random parents for reproduction from the elite population.
    size_t indexA = arma::randi<size_t>(arma::distr_param(0, populationSize - 1));
    size_t indexB = arma::randi<size_t>(arma::distr_param(0, populationSize - 1));

    // Make sure that the parents differ.
    if (indexA == indexB)
    {
      if (indexB < populationSize - 1)
        indexB++;
      else
        indexB--;
    }

    // Initialize the children to the respective parents.
    MatType childA = population[indexA], childB = population[indexB];

    Crossover(childA, childB, population[indexA], population[indexB]);

    Mutate(childA, lowerBound, upperBound);
    Mutate(childB, lowerBound, upperBound);

    // Add the children to the candidate population.
    children.push_back(childA);
    children.push_back(childB);
  }

  // Add the candidates to the elite population.
  population.insert(std::end(population), std::begin(children), std::end(children));
}

//! Perform crossover of genes for the children.
template<typename MatType>
inline void NSGA2::Crossover(MatType& childA,
                             MatType& childB,
                             const MatType& parentA,
                             const MatType& parentB)
{
  // Indices at which crossover is to occur.
  const arma::umat idx = arma::randu<MatType>(childA.n_rows, childA.n_cols) < crossoverProb;

  // Use traits from parentA for indices where idx is 1 and parentB otherwise.
  childA = parentA % idx + parentB % (1 - idx);
  // Use traits from parentB for indices where idx is 1 and parentA otherwise.
  childB = parentA % (1 - idx) + parentA % idx;
}

//! Perform mutation of the candidates weights with some noise.
template<typename MatType>
inline void NSGA2::Mutate(MatType& child,
                          const arma::vec& lowerBound,
                          const arma::vec& upperBound)
{
  child += (arma::randu<MatType>(child.n_rows, child.n_cols) < mutationProb) %
      (mutationStrength * arma::randn<MatType>(child.n_rows, child.n_cols));

  // Constrain all genes to be between bounds.
  for (size_t idx = 0; idx < numVariables; idx++)
  {
    if (child[idx] < lowerBound(idx))
      child[idx] = lowerBound(idx);
    else if (child[idx] > upperBound(idx))
      child[idx] = upperBound(idx);
  }
}

//! Sort population into Pareto fronts.
template<typename MatType>
inline void NSGA2::FastNonDominatedSort(
    std::vector<std::vector<size_t> >& fronts,
    std::vector<size_t>& ranks,
    std::vector<arma::Col<typename MatType::elem_type> >& calculatedObjectives)
{
  std::map<size_t, size_t> dominationCount;
  std::map<size_t, std::set<size_t> > dominated;

  // Reset and initialize fronts.
  fronts.clear();
  fronts.push_back(std::vector<size_t>());

  for (size_t p = 0; p < populationSize; p++)
  {
    dominated[p] = std::set<size_t>();
    dominationCount[p] = 0;

    for (size_t q = 0; q < populationSize; q++)
    {
      if (Dominates<MatType>(calculatedObjectives, p, q))
        dominated[p].insert(q);
      else if (Dominates<MatType>(calculatedObjectives, q, p))
        dominationCount[p] += 1;
    }

    if (dominationCount[p] == 0)
    {
      ranks[p] = 0;
      fronts[0].push_back(p);
    }
  }

  size_t i = 0;

  while (fronts[i].size() > 0)
  {
    std::vector<size_t> nextFront;

    for (size_t p: fronts[i])
    {
      for (size_t q: dominated[p])
      {
        dominationCount[q]--;

        if (dominationCount[q] == 0)
        {
          ranks[q] = i + 1;
          nextFront.push_back(q);
        }
      }
    }

    i++;
    fronts.push_back(nextFront);
  }
}

//! Check if a candidate Pareto dominates another candidate.
template<typename MatType>
inline bool NSGA2::Dominates(
    std::vector<arma::Col<typename MatType::elem_type> >& calculatedObjectives,
    size_t candidateP,
    size_t candidateQ)
{
  bool allBetterOrEqual = true;
  bool atleastOneBetter = false;
  size_t n_objectives = calculatedObjectives[0].n_elem;

  for (size_t i = 0; i < n_objectives; i++)
  {
    // P is worse than Q for the i-th objective function.
    if (calculatedObjectives[candidateP](i) > calculatedObjectives[candidateQ](i))
      allBetterOrEqual = false;

    // P is better than Q for the i-th objective function.
    else if (calculatedObjectives[candidateP](i) < calculatedObjectives[candidateQ](i))
      atleastOneBetter = true;
  }

  return allBetterOrEqual && atleastOneBetter;
}

//! Assign crowding distance to the population.
inline void NSGA2::CrowdingDistanceAssignment(const std::vector<size_t>& front,
                                              std::vector<double>& crowdingDistance)
{
  if (front.size() > 0)
  {
    for (size_t elem: front)
      crowdingDistance[elem] = 0;

    size_t fSize = front.size();

    for (size_t m = 0; m < numObjectives; m++)
    {
      crowdingDistance[front[0]] = std::numeric_limits<double>::max();
      crowdingDistance[front[fSize - 1]] = std::numeric_limits<double>::max();

      for (size_t i = 1; i < fSize - 1 ; i++)
      {
        crowdingDistance[front[i]] += (crowdingDistance[front[i - 1]] -
            crowdingDistance[front[i + 1]]) /
            (std::numeric_limits<double>::max() -
            std::numeric_limits<double>::min());
      }
    }
  }
}

//! Comparator for crowding distance based sorting.
inline bool NSGA2::CrowdingOperator(size_t idxP,
                                    size_t idxQ,
                                    const std::vector<size_t>& ranks,
                                    const std::vector<double>& crowdingDistance)
{
  if (ranks[idxP] < ranks[idxQ])
    return true;
  else if (ranks[idxP] == ranks[idxQ] && crowdingDistance[idxP] > crowdingDistance[idxQ])
    return true;

  return false;
}

} // namespace ens

#endif
