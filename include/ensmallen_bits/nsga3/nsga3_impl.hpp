/**
 * @file nsga3_impl.hpp
 * @author Satyam Shukla
 *
 * Implementation of the NSGA3 algorithm. Used for multi-objective
 * optimization problems on arbitrary functions.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more Information.
 */

#ifndef ENSMALLEN_NSGA3_NSGA3_IMPL_HPP
#define ENSMALLEN_NSGA3_NSGA3_IMPL_HPP

#include "nsga3.hpp"
#include <assert.h>
#include "normalization.hpp"

namespace ens {

template <typename ElementType>
inline NSGA3<ElementType>::NSGA3(
    const arma::Mat<ElementType>& referencePoints,
    const size_t populationSize,
    const size_t maxGenerations,
    const double crossoverProb,
    const double distributionIndex,
    const double eta,
    const arma::vec& lowerBound,
    const arma::vec& upperBound):
    referencePoints(referencePoints),    
    numObjectives(0),
    numVariables(0),
    populationSize(populationSize),
    maxGenerations(maxGenerations),
    crossoverProb(crossoverProb),
    distributionIndex(distributionIndex),
    eta(eta),
    lowerBound(lowerBound),
    upperBound(upperBound)
{ /* Nothing to do here. */ }

template <typename ElementType>
inline NSGA3<ElementType>::NSGA3(
    const arma::Mat<ElementType>& referencePoints,
    const size_t populationSize,
    const size_t maxGenerations,
    const double crossoverProb,
    const double distributionIndex,
    const double eta,
    const double lowerBound,
    const double upperBound):
    referencePoints(referencePoints),
    numObjectives(0),
    numVariables(0),
    populationSize(populationSize),
    maxGenerations(maxGenerations),
    crossoverProb(crossoverProb),
    distributionIndex(distributionIndex),
    eta(eta),
    lowerBound(lowerBound * arma::ones(1, 1)),
    upperBound(upperBound * arma::ones(1, 1))
{ /* Nothing to do here. */ }

//! Optimize the function.
template<typename ElementType>
template<typename MatType,
         typename... ArbitraryFunctionType,
         typename... CallbackTypes>
typename MatType::elem_type NSGA3<ElementType>::Optimize(
    std::tuple<ArbitraryFunctionType...>& objectives,
    MatType& iterateIn,
    CallbackTypes&&... callbacks)
{
  // Make sure for evolution to work at least four candidates are present.
  if (populationSize < 4 && populationSize % 4 != 0)
  {
    throw std::logic_error("NSGA3::Optimize(): population size should be at"
        " least 4, and, a multiple of 4!");
  }

  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;

  BaseMatType& iterate = (BaseMatType&) iterateIn;

  // Make sure that we have the methods that we need.  Long name...
  traits::CheckArbitraryFunctionTypeAPI<ArbitraryFunctionType...,
      BaseMatType>();
  RequireDenseFloatingPointType<BaseMatType>();

  // Check if lower bound is a vector of a single dimension.
  if (lowerBound.n_rows == 1)
    lowerBound = lowerBound(0, 0) * arma::ones(iterate.n_rows, iterate.n_cols);

  // Check if upper bound is a vector of a single dimension.
  if (upperBound.n_rows == 1)
    upperBound = upperBound(0, 0) * arma::ones(iterate.n_rows, iterate.n_cols);

  // Check the dimensions of lowerBound and upperBound.
  assert(lowerBound.n_rows == iterate.n_rows && "The dimensions of "
      "lowerBound are not the same as the dimensions of iterate.");
  assert(upperBound.n_rows == iterate.n_rows && "The dimensions of "
      "upperBound are not the same as the dimensions of iterate.");

  numObjectives = sizeof...(ArbitraryFunctionType);
  numVariables = iterate.n_rows;

  assert(numObjectives == referencePoints.n_rows && "The dimensions of "
      "reference points do not match the number of functions.");

  // Cache calculated objectives.
  std::vector<arma::Col<ElementType> > calculatedObjectives(populationSize);

  // Population size reserved to 2 * populationSize + 1 to accommodate
  // for the size of intermediate candidate population.
  std::vector<BaseMatType> population;
  std::vector<BaseMatType> tempPopulation;
  population.reserve(2 * populationSize + 1);
  tempPopulation.reserve(populationSize);

  // Pareto fronts, initialized during non-dominated sorting.
  // Stores indices of population belonging to a certain front.
  std::vector<std::vector<size_t> > fronts;
  // Initialised during non-dominated sorting.
  std::vector<size_t> ranks;

  //! Useful temporaries for float-like comparisons.
  const BaseMatType castedLowerBound = arma::conv_to<BaseMatType>::from(lowerBound);
  const BaseMatType castedUpperBound = arma::conv_to<BaseMatType>::from(upperBound);

  // Controls early termination of the optimization process.
  bool terminate = false;

  // Generate the population based on a uniform distribution around the given
  // starting point.
  for (size_t i = 0; i < populationSize; i++)
  {
    population.push_back(arma::randu<BaseMatType>(iterate.n_rows,
        iterate.n_cols) - 0.5 + iterate);

    // Constrain all genes to be within bounds.
    population[i] = arma::min(arma::max(population[i], castedLowerBound), castedUpperBound);
  }
  Normalization<BaseMatType> hpn(numObjectives);

  Info << "NSGA3 initialized successfully. Optimization started." << std::endl;

  // Iterate until maximum number of generations is obtained.
  Callback::BeginOptimization(*this, objectives, iterate, callbacks...);

  for (size_t generation = 1; generation <= maxGenerations && !terminate; generation++)
  {
    // Create new population of candidate from the present elite population.
    // Have P_t, generate G_t using P_t.
    BinaryTournamentSelection(population, castedLowerBound, castedUpperBound);

    // Evaluate the objectives for the new population.
    calculatedObjectives.resize(population.size());
    std::fill(calculatedObjectives.begin(), calculatedObjectives.end(),
        arma::Col<ElementType>(numObjectives, arma::fill::zeros));

    EvaluateObjectives(population, objectives, calculatedObjectives);

    // Perform fast non dominated sort on P_t ∪ G_t.
    ranks.resize(population.size());
    FastNonDominatedSort<arma::Col<ElementType>>(fronts, ranks, calculatedObjectives);

    hpn.update(calculatedObjectives, fronts[0]);
    arma::Col<ElementType> denom = hpn.NadirPoint() - hpn.IdealPoint();
    
    // S_t and P_t+1 declared. 
    std::vector<size_t> selectedPoints;
    std::vector<size_t> nextPopulation;

    size_t index = 0;
    while (nextPopulation.size() + fronts[index].size() < populationSize)
    {
      selectedPoints.insert(selectedPoints.end(), fronts[index].begin(), fronts[index].end());
      nextPopulation.insert(nextPopulation.end(), fronts[index].begin(), fronts[index].end());   
      index++;
    }

    if(nextPopulation.size() != populationSize)
    {
      selectedPoints.insert(selectedPoints.end(), fronts[index].begin(), fronts[index].end());

      size_t lastFront = index;

      for (index = 0; index < selectedPoints.size(); index++)
      {
        calculatedObjectives[selectedPoints[index]] = 
            calculatedObjectives[selectedPoints[index]] - hpn.IdealPoint();
        calculatedObjectives[selectedPoints[index]] = 
            calculatedObjectives[selectedPoints[index]] / denom;
      }

      // Find the associated reference directions to the selected points.
      arma::urowvec refIndex(selectedPoints.size());
      arma::Row<ElementType> dists(selectedPoints.size());

      Associate<arma::Col<ElementType>>(refIndex, dists, calculatedObjectives,
          selectedPoints);

      // Calculate the niche count of S_t and performing the niching operation.
      arma::Row<size_t> count(referencePoints.n_cols, arma::fill::zeros);

      NicheCount(count, refIndex, nextPopulation);
      Niching(populationSize - nextPopulation.size(), count, refIndex,
          dists, fronts[lastFront], nextPopulation);
    }
    for (size_t i : nextPopulation)
    {
      tempPopulation.push_back(population[i]);
    }
    population = tempPopulation;
    tempPopulation.erase(tempPopulation.begin(), tempPopulation.end());

    terminate |= Callback::GenerationalStepTaken(*this, objectives, iterate,
        calculatedObjectives, fronts, callbacks...);
  }
  EvaluateObjectives(population, objectives, calculatedObjectives);
  // Set the candidates from the Pareto Set as the output.
  paretoSet.set_size(population[0].n_rows, population[0].n_cols, fronts[0].size());
  // The Pareto Set is stored, can be obtained via ParetoSet() getter.
  for (size_t solutionIdx = 0; solutionIdx < fronts[0].size(); ++solutionIdx)
  {
    paretoSet.slice(solutionIdx) =
      arma::conv_to<arma::mat>::from(population[fronts[0][solutionIdx]]);
  }

  // Set the candidates from the Pareto Front as the output.
  paretoFront.set_size(calculatedObjectives[0].n_rows, calculatedObjectives[0].n_cols,
      fronts[0].size());
  // The Pareto Front is stored, can be obtained via ParetoFront() getter.
  for (size_t solutionIdx = 0; solutionIdx < fronts[0].size(); ++solutionIdx)
  {
    paretoFront.slice(solutionIdx) =
      arma::conv_to<arma::mat>::from(calculatedObjectives[fronts[0][solutionIdx]]);
  }

  // Clear rcFront, in case it is later requested by the user for reverse
  // compatibility reasons.
  rcFront.clear();

  // Assign iterate to first element of the Pareto Set.
  iterate = population[fronts[0][0]];

  Callback::EndOptimization(*this, objectives, iterate, callbacks...);

  ElemType performance = std::numeric_limits<ElemType>::max();

  for (const arma::Col<ElementType>& objective: calculatedObjectives)
    if (arma::accu(objective) < performance)
      performance = arma::accu(objective);

  return performance;
}

//! No objectives to evaluate.
template<typename ElementType>
template<std::size_t I,
         typename MatType,
         typename ...ArbitraryFunctionType>
typename std::enable_if<I == sizeof...(ArbitraryFunctionType), void>::type
NSGA3<ElementType>::EvaluateObjectives(
    std::vector<MatType>&,
    std::tuple<ArbitraryFunctionType...>&,
    std::vector<arma::Col<ElementType> >&)
{
  // Nothing to do here.
}

//! Evaluate the objectives for the entire population.
template<typename ElementType>
template<std::size_t I,
         typename MatType,
         typename ...ArbitraryFunctionType>
typename std::enable_if<I < sizeof...(ArbitraryFunctionType), void>::type
NSGA3<ElementType>::EvaluateObjectives(
    std::vector<MatType>& population,
    std::tuple<ArbitraryFunctionType...>& objectives,
    std::vector<arma::Col<ElementType> >& calculatedObjectives)
{
  for (size_t i = 0; i < population.size(); i++)
  {
    calculatedObjectives[i](I) = std::get<I>(objectives).Evaluate(population[i]);
    EvaluateObjectives<I+1, MatType, ArbitraryFunctionType...>(population, objectives,
                                                               calculatedObjectives);
  }
}

//! Reproduce and generate new candidates.
template<typename ElementType>
template<typename MatType>
inline void NSGA3<ElementType>::BinaryTournamentSelection(
    std::vector<MatType>& population,
    const MatType& lowerBound,
    const MatType& upperBound)
{
  std::vector<MatType> children;

  while (children.size() < populationSize)
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

    if(arma::randu() <= crossoverProb)
      Crossover(childA, childB, population[indexA], population[indexB], 
                lowerBound, upperBound);

    Mutate(childA, 1.0 / static_cast<double>(numVariables),
          lowerBound, upperBound);
    Mutate(childB, 1.0 / static_cast<double>(numVariables),
          lowerBound, upperBound);

    // Add the children to the candidate population.
    children.push_back(childA);
    children.push_back(childB);
  }

  // Add the candidates to the elite population.
  population.insert(std::end(population), std::begin(children), std::end(children));
}

//! Perform simulated binary crossover (SBX) of genes for the children.
template<typename ElementType>
template<typename MatType> 
inline void NSGA3<ElementType>::Crossover(MatType& childA,
                                          MatType& childB,
                                          const MatType& parentA,
                                          const MatType& parentB,
                                          const MatType& lowerBound,
                                          const MatType& upperBound)
{
    //! Generates a child from two parent individuals
    // according to the polynomial probability distribution.
    arma::Cube<typename MatType::elem_type> parents(parentA.n_rows, parentA.n_cols, 2);
    parents.slice(0) = parentA;
    parents.slice(1) = parentB;
    MatType current_min =  arma::min(parents, 2);
    MatType current_max =  arma::max(parents, 2);

    if (arma::accu(parentA - parentB < 1e-14))
    {
      childA = parentA;
      childB = parentB;
      return;
    }
    MatType current_diff = current_max - current_min;
    current_diff.transform( [](typename MatType::elem_type val) 
      { return (val < 1e-10 ? 1e-10:val); } );

    // Calculating beta used for the final crossover.
    MatType beta1 = 1 + 2.0 * (current_min - lowerBound) / current_diff;
    MatType beta2 = 1 + 2.0 * (upperBound - current_max) / current_diff;
    MatType alpha1 = 2 - arma::pow(beta1, -(eta + 1));
    MatType alpha2 = 2 - arma::pow(beta2, -(eta + 1));

    MatType us(arma::size(alpha1), arma::fill::randu);
    arma::umat mask1 = us > (1.0 / alpha1); 
    MatType betaq1 = arma::pow(us % alpha1, 1. / (eta + 1));
    betaq1 = betaq1 % (mask1 != 1.0) + arma::pow((1.0 / (2.0 - us % alpha1)), 1.0 / (eta + 1)) % mask1;
    arma::umat mask2 = us > (1.0 / alpha2);
    MatType betaq2 = arma::pow(us % alpha2, 1 / (eta + 1));
    betaq2 = betaq2 % (mask1 != 1.0) + arma::pow((1.0 / (2.0 - us % alpha2)), 1.0 / (eta + 1)) % mask2;

    // Variables after the cross over for all of them.
    MatType c1 = 0.5 * ((current_min + current_max) - betaq1 % current_diff);
    MatType c2 = 0.5 * ((current_min + current_max) + betaq2 % current_diff);
    c1 = arma::min(arma::max(c1, lowerBound), upperBound);
    c2 = arma::min(arma::max(c2, lowerBound), upperBound);
    
    // Decision for the crossover between the two parents for each variable.
    us.randu();
    childA = parentA % (us <= 0.5);
    childB = parentB % (us <= 0.5);
    us.randu();
    childA = childA + c1 % ((us <= 0.5) % (childA == 0));
    childA = childA + c2 % ((us > 0.5) % (childA == 0));
    childB = childB + c2 % ((us <= 0.5) % (childB == 0));
    childB = childB + c1 % ((us > 0.5) % (childB == 0));
}

//! Perform Polynomial mutation of the candidate.
template<typename ElementType>
template<typename MatType>
inline void NSGA3<ElementType>::Mutate(MatType& candidate,
                                       double mutationRate,
                                       const MatType& lowerBound,
                                       const MatType& upperBound)
{
    const size_t numVariables = candidate.n_rows;
    for (size_t geneIdx = 0; geneIdx < numVariables; ++geneIdx)
    {
      // Should this gene be mutated?
      if (arma::randu() > mutationRate)
        continue;

      const double geneRange = upperBound(geneIdx) - lowerBound(geneIdx);
      // Normalised distance from the bounds.
      const double lowerDelta = (candidate(geneIdx) - lowerBound(geneIdx)) / geneRange;
      const double upperDelta = (upperBound(geneIdx) - candidate(geneIdx)) / geneRange;
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
    candidate = arma::min(arma::max(candidate, lowerBound), upperBound);
}

//! Find the distance of a front from a line formed by two points.
template <typename ElementType>
template <typename ColType>
inline void NSGA3<ElementType>::PointToLineDistance(
    arma::Row<ElementType>& distances,
    const std::vector<ColType>& calculatedObjectives,
    const std::vector<size_t>& front,
    const ColType& pointA,
    const ColType& pointB)
{
  arma::Row<ElementType> distancesTemp(front.size());
  ColType ba = pointB - pointA; 
  ColType pa;

  for (size_t i = 0; i < front.size(); i++)
  {
    size_t ind = front[i];
 
    pa = (calculatedObjectives[ind] - pointA);
    double t = arma::dot(pa, ba) / arma::dot(ba, ba);
    distancesTemp[i] = std::pow(arma::accu(arma::pow((pa - t * ba), 2)), 0.5);
  }
  distances = distancesTemp;
}

//! Sort population into Pareto fronts.
template<typename ElementType>
template<typename ColType>
inline void NSGA3<ElementType>::FastNonDominatedSort(
    std::vector<std::vector<size_t> >& fronts,
    std::vector<size_t>& ranks,
    std::vector<ColType>& calculatedObjectives)
{
  std::map<size_t, size_t> dominationCount;
  std::map<size_t, std::set<size_t> > dominated;

  // Reset and initialize fronts.
  fronts.clear();
  fronts.push_back(std::vector<size_t>());

  for (size_t p = 0; p < calculatedObjectives.size(); p++)
  {
    dominated[p] = std::set<size_t>();
    dominationCount[p] = 0;

    for (size_t q = 0; q < calculatedObjectives.size(); q++)
    {
      if (Dominates<arma::Col<ElementType>>(calculatedObjectives, p, q))
        dominated[p].insert(q);
      else if (Dominates<arma::Col<ElementType>>(calculatedObjectives, q, p))
        dominationCount[p] += 1;
    }

    if (dominationCount[p] == 0)
    {
      ranks[p] = 0;
      fronts[0].push_back(p);
    }
  }

  size_t i = 0;

  while (!fronts[i].empty())
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
  // Remove the empty final set.
  fronts.pop_back();
}

template<typename ElementType>
inline void NSGA3<ElementType>::Niching(size_t K,
                                 arma::Row<size_t>& nicheCount,
                                 const arma::urowvec& refIndex,
                                 const arma::Row<ElementType>& dists,
                                 const std::vector<size_t>& front,
                                 std::vector<size_t>& population)
{
  arma::Row<double> popMask(front.size(), arma::fill::zeros);
  int nextPopSize = population.size();
  size_t k = 0;
  while (k < K)
  {
   size_t jMin = arma::index_min(nicheCount);
   std::vector<unsigned int> I;
   for (size_t i = 0; i < front.size(); i++)
   {
    if(refIndex[nextPopSize + i] == jMin && !popMask[i])
      I.push_back(i);
   }
   if (I.size() != 0)
   {
      size_t min = 0;
      if(nicheCount[jMin] == 0)
      {
        for (size_t i = 0; i < I.size(); i++)
        {
          if(dists[nextPopSize + I[i]] < dists[nextPopSize + I[min]])
          {
            min = i;
          }
        }
      }
      population.push_back(front[I[min]]);

      nicheCount[jMin] += 1;
      popMask[I[min]] = 1;
      k++;
   }
   else
   {
    nicheCount[jMin] = 100000;
   }
  }
}


template <typename ElementType>
template <typename ColType>
inline void NSGA3<ElementType>::Associate(
    arma::urowvec& refIndex,
    arma::Row<ElementType>& dists,
    const std::vector<ColType>& calculatedObjectives,
    const std::vector<size_t>& St)
{
  arma::Mat<ElementType> d(referencePoints.n_cols, St.size());
  ColType zero(arma::size(calculatedObjectives[0]),arma::fill::zeros);
  arma::Row<ElementType> temp;
  for (size_t i = 0; i < referencePoints.n_cols; i++)
  {
    PointToLineDistance<ColType>(temp, calculatedObjectives, St, 
        zero, referencePoints.col(i));
    d.row(i) = temp;
  }
  refIndex = arma::index_min(d, 0);
  dists = arma::min(d, 0);
}

template <typename ElementType>
inline void NSGA3<ElementType>::NicheCount(arma::Row<size_t>& count,
                                    const arma::urowvec& refIndex,
                                    const std::vector<size_t>& nextPopulation)
{
  for (size_t i = 0; i < nextPopulation.size(); i++)
  {
    count[refIndex[i]] += 1;
  }
}

//! Check if a candidate Pareto dominates another candidate.
template<typename ElementType>
template<typename ColType>
inline bool NSGA3<ElementType>::Dominates(
    std::vector<ColType>& calculatedObjectives,
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

} // namespace ens

#endif