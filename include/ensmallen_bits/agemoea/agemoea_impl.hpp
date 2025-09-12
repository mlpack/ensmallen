/**
 * @file agemoea_impl.hpp
 * @author Satyam Shukla
 *
 * Implementation of the AGEMOEA algorithm. Used for multi-objective
 * optimization problems on arbitrary functions.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more Information.
 */

#ifndef ENSMALLEN_AGEMOEA_AGEMOEA_IMPL_HPP
#define ENSMALLEN_AGEMOEA_AGEMOEA_IMPL_HPP

#include "agemoea.hpp"
#include <assert.h>

namespace ens {

inline AGEMOEA::AGEMOEA(const size_t populationSize,
                        const size_t maxGenerations,
                        const double crossoverProb,
                        const double distributionIndex,
                        const double epsilon,
                        const double eta,
                        const arma::vec& lowerBound,
                        const arma::vec& upperBound) :
    numObjectives(0),
    numVariables(0),
    populationSize(populationSize),
    maxGenerations(maxGenerations),
    crossoverProb(crossoverProb),
    distributionIndex(distributionIndex),
    epsilon(epsilon),
    eta(eta),
    lowerBound(lowerBound),
    upperBound(upperBound)
{ /* Nothing to do here. */ }

inline AGEMOEA::AGEMOEA(const size_t populationSize,
                        const size_t maxGenerations,
                        const double crossoverProb,
                        const double distributionIndex,
                        const double epsilon,
                        const double eta,
                        const double lowerBound,
                        const double upperBound) :
    numObjectives(0),
    numVariables(0),
    populationSize(populationSize),
    maxGenerations(maxGenerations),
    crossoverProb(crossoverProb),
    distributionIndex(distributionIndex),
    epsilon(epsilon),
    eta(eta),
    lowerBound(lowerBound * arma::ones(1, 1)),
    upperBound(upperBound * arma::ones(1, 1))
{ /* Nothing to do here. */ }

//! Optimize the function.
template<typename MatType,
         typename... ArbitraryFunctionType,
         typename... CallbackTypes>
typename MatType::elem_type AGEMOEA::Optimize(
    std::tuple<ArbitraryFunctionType...>& objectives,
    MatType& iterateIn,
    CallbackTypes&&... callbacks)
{
  typedef typename ForwardType<MatType>::bcube CubeType;
  CubeType paretoFront, paretoSet;
  return Optimize(objectives, iterateIn, paretoFront, paretoSet,
      std::forward<CallbackTypes>(callbacks)...);
}

//! Optimize the function.
template<typename MatType,
         typename CubeType,
         typename... ArbitraryFunctionType,
         typename... CallbackTypes>
typename MatType::elem_type AGEMOEA::Optimize(
    std::tuple<ArbitraryFunctionType...>& objectives,
    MatType& iterateIn,
    CubeType& paretoFrontIn,
    CubeType& paretoSetIn,
    CallbackTypes&&... callbacks)
{
  // Make sure for evolution to work at least four candidates are present.
  if (populationSize < 4 && populationSize % 4 != 0)
  {
    throw std::logic_error("AGEMOEA::Optimize(): population size should be at"
        " least 4, and, a multiple of 4!");
  }

  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;
  typedef typename ForwardType<MatType>::bcol BaseColType;
  typedef typename ForwardType<CubeType>::bmat CubeBaseMatType;

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

  // Cache calculated objectives.
  std::vector<BaseColType> calculatedObjectives(populationSize);

  // Population size reserved to 2 * populationSize + 1 to accommodate
  // for the size of intermediate candidate population.
  std::vector<BaseMatType> population;
  population.reserve(2 * populationSize + 1);

  // Pareto fronts, initialized during non-dominated sorting.
  // Stores indices of population belonging to a certain front.
  std::vector<std::vector<size_t> > fronts;
  // Initialised in SurvivalScoreAssignment.
  std::vector<ElemType> survivalScore;
  // Initialised during non-dominated sorting.
  std::vector<size_t> ranks;

  //! Useful temporaries for float-like comparisons.
  const BaseMatType castedLowerBound = conv_to<BaseMatType>::from(lowerBound);
  const BaseMatType castedUpperBound = conv_to<BaseMatType>::from(upperBound);

  // Controls early termination of the optimization process.
  bool terminate = false;

  // Generate the population based on a uniform distribution around the given
  // starting point.
  for (size_t i = 0; i < populationSize; i++)
  {
    population.push_back(arma::randu<BaseMatType>(iterate.n_rows,
        iterate.n_cols) - ElemType(0.5) + iterate);

    // Constrain all genes to be within bounds.
    population[i] = min(max(population[i], castedLowerBound),
        castedUpperBound);
  }

  Info << "AGEMOEA initialized successfully. Optimization started." << std::endl;

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
        BaseColType(numObjectives, GetFillType<MatType>::zeros));
    EvaluateObjectives(population, objectives, calculatedObjectives);

    // Perform fast non dominated sort on P_t ∪ G_t.
    ranks.resize(population.size());
    FastNonDominatedSort<BaseMatType>(fronts, ranks, calculatedObjectives);

    arma::Col<ElemType> idealPoint(calculatedObjectives[fronts[0][0]]);
    for (size_t index = 1; index < fronts[0].size(); index++)
    {
      idealPoint = min(idealPoint, calculatedObjectives[fronts[0][index]]);
    }

    // Perform survival score assignment.
    survivalScore.resize(population.size());
    std::fill(survivalScore.begin(), survivalScore.end(), 0.);
    ElemType dimension;
    BaseColType normalize(numObjectives, GetFillType<MatType>::zeros);
    for (size_t fNum = 0; fNum < fronts.size(); fNum++)
    {
      SurvivalScoreAssignment<BaseMatType>(fronts[fNum], idealPoint,
      calculatedObjectives, survivalScore, normalize, dimension, fNum);
    }

    // Sort based on survival score.
    std::sort(population.begin(), population.end(),
      [this, ranks, survivalScore, population]
        (BaseMatType candidateP, BaseMatType candidateQ)
          {
            size_t idxP{}, idxQ{};
            for (size_t i = 0; i < population.size(); i++)
            {
              if (approx_equal(population[i], candidateP, "absdiff",
                  ElemType(epsilon)))
                idxP = i;

              if (approx_equal(population[i], candidateQ, "absdiff",
                  ElemType(epsilon)))
                idxQ = i;
            }

            return SurvivalScoreOperator<BaseMatType>(idxP, idxQ, ranks,
                survivalScore);
          }
    );

    // Yield a new population P_{t+1} of size populationSize.
    // Discards unfit population from the R_{t} to yield P_{t+1}.
    population.resize(populationSize);

    terminate |= Callback::GenerationalStepTaken(*this, objectives, iterate,
        calculatedObjectives, fronts, callbacks...);
  }
  EvaluateObjectives(population, objectives, calculatedObjectives);
  // Set the candidates from the Pareto Set as the output.
  paretoSetIn.set_size(population[0].n_rows, population[0].n_cols,
      population.size());
  // The Pareto Set is stored, can be obtained via ParetoSet() getter.
  for (size_t solutionIdx = 0; solutionIdx < population.size(); ++solutionIdx)
  {
    paretoSetIn.slice(solutionIdx) =
      conv_to<CubeBaseMatType>::from(population[solutionIdx]);
  }

  // Set the candidates from the Pareto Front as the output.
  paretoFrontIn.set_size(calculatedObjectives[0].n_rows,
      calculatedObjectives[0].n_cols, population.size());
  for (size_t solutionIdx = 0; solutionIdx < population.size(); ++solutionIdx)
  {
    paretoFrontIn.slice(solutionIdx) =
      conv_to<CubeBaseMatType>::from(calculatedObjectives[solutionIdx]);
  }

  // Assign iterate to first element of the Pareto Set.
  iterate = population[fronts[0][0]];

  Callback::EndOptimization(*this, objectives, iterate, callbacks...);

  ElemType performance = std::numeric_limits<ElemType>::max();

  for (const BaseColType& objective: calculatedObjectives)
    if (accu(objective) < performance)
      performance = accu(objective);

  return performance;
}

//! No objectives to evaluate.
template<std::size_t I,
         typename InputMatType,
         typename ObjectiveMatType,
         typename ...ArbitraryFunctionType>
typename std::enable_if<I == sizeof...(ArbitraryFunctionType), void>::type
AGEMOEA::EvaluateObjectives(
    std::vector<InputMatType>&,
    std::tuple<ArbitraryFunctionType...>&,
    std::vector<ObjectiveMatType>&)
{
  // Nothing to do here.
}

//! Evaluate the objectives for the entire population.
template<std::size_t I,
         typename InputMatType,
         typename ObjectiveMatType,
         typename ...ArbitraryFunctionType>
typename std::enable_if<I < sizeof...(ArbitraryFunctionType), void>::type
AGEMOEA::EvaluateObjectives(
    std::vector<InputMatType>& population,
    std::tuple<ArbitraryFunctionType...>& objectives,
    std::vector<ObjectiveMatType>& calculatedObjectives)
{
  for (size_t i = 0; i < population.size(); i++)
  {
    calculatedObjectives[i](I) = std::get<I>(objectives).Evaluate(population[i]);
    EvaluateObjectives<I+1, InputMatType, ObjectiveMatType,
        ArbitraryFunctionType...>(population, objectives,
        calculatedObjectives);
  }
}

//! Reproduce and generate new candidates.
template<typename InputMatType>
inline void AGEMOEA::BinaryTournamentSelection(std::vector<InputMatType>& population,
                                               const InputMatType& lowerBound,
                                               const InputMatType& upperBound)
{
  std::vector<InputMatType> children;

  while (children.size() < population.size())
  {
    // Choose two random parents for reproduction from the elite population.
    size_t indexA = arma::randi<size_t>(
        arma::distr_param(0, populationSize - 1));
    size_t indexB = arma::randi<size_t>(
        arma::distr_param(0, populationSize - 1));

    // Make sure that the parents differ.
    if (indexA == indexB)
    {
      if (indexB < populationSize - 1)
        indexB++;
      else
        indexB--;
    }

    // Initialize the children to the respective parents.
    InputMatType childA = population[indexA], childB = population[indexB];

    if (arma::randu() <= crossoverProb)
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
  population.insert(std::end(population), std::begin(children),
      std::end(children));
}

//! Perform simulated binary crossover (SBX) of genes for the children.
template<typename InputMatType>
inline void AGEMOEA::Crossover(InputMatType& childA,
                               InputMatType& childB,
                               const InputMatType& parentA,
                               const InputMatType& parentB,
                               const InputMatType& lowerBound,
                               const InputMatType& upperBound)
{
  typedef typename InputMatType::elem_type ElemType;
  typedef typename ForwardType<InputMatType>::bcube BaseCubeType;
  typedef typename ForwardType<InputMatType>::umat UMatType;

  // Generates a child from two parent individuals
  // according to the polynomial probability distribution.
  BaseCubeType parents(parentA.n_rows,
    parentA.n_cols, 2);
  parents.slice(0) = parentA;
  parents.slice(1) = parentB;
  InputMatType current_min = min(parents, 2);
  InputMatType current_max = max(parents, 2);

  if (accu(parentA - parentB < ElemType(1e-14)))
  {
    childA = parentA;
    childB = parentB;
    return;
  }
  InputMatType current_diff = current_max - current_min;
  current_diff.transform( [](ElemType val)
      { return (val < ElemType(1e-10) ? ElemType(1e-10) : val); } );

  // Calculating beta used for the final crossover.
  InputMatType beta1 = 1 + 2 * (current_min - lowerBound) / current_diff;
  InputMatType beta2 = 1 + 2 * (upperBound - current_max) / current_diff;
  InputMatType alpha1 = 2 - pow(beta1, -(eta + 1));
  InputMatType alpha2 = 2 - pow(beta2, -(eta + 1));

  InputMatType us(size(alpha1), GetFillType<InputMatType>::randu);

  UMatType mask1 = us > (1 / alpha1);
  InputMatType betaq1 = pow(us % alpha1, 1. / (eta + 1));
  betaq1 = betaq1 % (mask1 != 1) + pow((1 / (2 - us % alpha1)),
      1 / (eta + 1)) % mask1;
  UMatType mask2 = us > (1 / alpha2);
  InputMatType betaq2 = pow(us % alpha2, 1 / (eta + 1));
  betaq2 = betaq2 % (mask1 != 1) + pow((1 / (2 - us % alpha2)),
      1 / (eta + 1)) % mask2;

  // Variables after the cross over for all of them.
  InputMatType c1 = ((current_min + current_max) - betaq1 % current_diff) / 2;
  InputMatType c2 = ((current_min + current_max) + betaq2 % current_diff) / 2;
  c1 = min(max(c1, lowerBound), upperBound);
  c2 = min(max(c2, lowerBound), upperBound);

  // Decision for the crossover between the two parents for each variable.
  us.randu();
  childA = parentA % (us <= ElemType(0.5));
  childB = parentB % (us <= ElemType(0.5));
  us.randu();
  childA = childA + c1 % ((us <= ElemType(0.5)) % (childA == 0));
  childA = childA + c2 % ((us >  ElemType(0.5)) % (childA == 0));
  childB = childB + c2 % ((us <= ElemType(0.5)) % (childB == 0));
  childB = childB + c1 % ((us >  ElemType(0.5)) % (childB == 0));
}

//! Perform Polynomial mutation of the candidate.
template<typename MatType>
inline void AGEMOEA::Mutate(MatType& candidate,
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
    const double lowerDelta = (candidate(geneIdx)
        - lowerBound(geneIdx)) / geneRange;
    const double upperDelta = (upperBound(geneIdx)
        - candidate(geneIdx)) / geneRange;
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

    candidate(geneIdx) +=
        typename MatType::elem_type(perturbationFactor * geneRange);
  }
  //! Enforce bounds.
  candidate = min(max(candidate, lowerBound), upperBound);
}

template <typename MatType>
inline void AGEMOEA::NormalizeFront(
      std::vector<arma::Col<typename MatType::elem_type> >& calculatedObjectives,
      arma::Col<typename MatType::elem_type>& normalization,
      const std::vector<size_t>& front,
      const arma::Row<size_t>& extreme)
{
  arma::Mat<typename MatType::elem_type> vectorizedObjectives(numObjectives,
      front.size());
  arma::Mat<typename MatType::elem_type> vectorizedExtremes(numObjectives,
      extreme.n_elem);
  for (size_t i = 0; i < front.size(); i++)
  {
    vectorizedObjectives.col(i) = calculatedObjectives[front[i]];
  }
  for (size_t i = 0; i < extreme.n_elem; i++)
  {
    vectorizedExtremes.col(i) = calculatedObjectives[front[extreme[i]]];
  }

  if (front.size() < numObjectives)
  {
    normalization = arma::max(vectorizedObjectives, 1);
    return;
  }
  arma::Col<typename MatType::elem_type> temp;
  arma::uvec unique = arma::find_unique(extreme);
  if (extreme.n_elem != unique.n_elem)
  {
    normalization = arma::max(vectorizedObjectives, 1);
    return;
  }
  arma::Col<typename MatType::elem_type> one(extreme.n_elem, arma::fill::ones);
  arma::Col<typename MatType::elem_type> hyperplane(numObjectives, arma::fill::zeros);
  try{
    hyperplane = arma::solve(
    vectorizedExtremes.t(), one);
  }
  catch(...)
  {
    normalization = arma::max(vectorizedObjectives, 1);
    normalization = normalization + (normalization == 0);
    return;
  }
  if (hyperplane.has_inf() || hyperplane.has_nan() || (arma::accu(hyperplane < 0.0) > 0))
  {
    normalization = arma::max(vectorizedObjectives, 1);
  }
  else
  {
    normalization = 1. / hyperplane;
    if (normalization.has_inf() || normalization.has_nan())
    {
      normalization = arma::max(vectorizedObjectives, 1);
    }
  }
  normalization = normalization + (normalization == 0);
}

template <typename MatType>
inline typename MatType::elem_type AGEMOEA::GetGeometry(
    std::vector<arma::Col<typename MatType::elem_type> >& calculatedObjectives,
    const std::vector<size_t>& front,
    const arma::Row<size_t>& extreme)
{
  typedef typename MatType::elem_type ElemType;

  arma::Row<ElemType> d;
  arma::Col<ElemType> zero(numObjectives, arma::fill::zeros);
  arma::Col<ElemType> one(numObjectives, arma::fill::ones);

  PointToLineDistance<MatType>(d, calculatedObjectives, front, zero, one);

  for (size_t i = 0; i < extreme.size(); i++)
  {
    d[extreme[i]] = arma::Datum<ElemType>::inf;
  }

  size_t index = arma::index_min(d);
  ElemType avg = accu(calculatedObjectives[front[index]]) / numObjectives;
  ElemType p = std::log(ElemType(numObjectives)) / std::log(1 / avg);
  if (p <= ElemType(0.1) || std::isnan(p))
    p = 1;

  return p;
}

//! Pairwise distance for each point in the given front.
template <typename MatType>
inline void AGEMOEA::PairwiseDistance(
    MatType& f,
    std::vector<arma::Col<typename MatType::elem_type> >& calculatedObjectives,
    const std::vector<size_t>& front,
    const typename MatType::elem_type dimension)
{
  for (size_t i = 0; i < front.size(); i++)
  {
    for (size_t j = i + 1; j < front.size(); j++)
    {
      f(i, j) = std::pow(accu(pow(abs(
          calculatedObjectives[front[i]] - calculatedObjectives[front[j]]),
          dimension)), 1 / dimension);
      f(j, i) = f(i, j);
    }
  }
}

//! Find the index of the of the extreme points in the given front.
template <typename MatType>
void AGEMOEA::FindExtremePoints(
    arma::Row<size_t>& indexes,
    std::vector<arma::Col<typename MatType::elem_type> >& calculatedObjectives,
    const std::vector<size_t>& front)
{
  typedef typename MatType::elem_type ElemType;

  if (numObjectives >= front.size())
  {
    indexes = arma::linspace<arma::Row<size_t>>(0, front.size() - 1, front.size());
    return;
  }

  arma::Mat<ElemType> W(numObjectives, numObjectives, arma::fill::eye);
  W = W + 1e-6;
  std::vector<bool> selected(front.size());
  arma::Col<ElemType> z(numObjectives, arma::fill::zeros);
  arma::Row<ElemType> dists;
  for (size_t i = 0; i < numObjectives; i++)
  {
    PointToLineDistance<MatType>(dists, calculatedObjectives, front, z, W.col(i));
    for (size_t j = 0; j < front.size(); j++)
      if (selected[j]){dists[j] = arma::datum::inf;}
    indexes[i] = dists.index_min();
    selected[dists.index_min()] = true;
  }
}

//! Find the distance of a front from a line formed by two points.
template <typename MatType>
void AGEMOEA::PointToLineDistance(
    arma::Row<typename MatType::elem_type>& distances,
    std::vector<arma::Col<typename MatType::elem_type> >& calculatedObjectives,
    const std::vector<size_t>& front,
    const arma::Col<typename MatType::elem_type>& pointA,
    const arma::Col<typename MatType::elem_type>& pointB)
{
  typedef typename MatType::elem_type ElemType;
  arma::Row<ElemType> distancesTemp(front.size());
  arma::Col<ElemType> ba = pointB - pointA;
  arma::Col<ElemType> pa;

  for (size_t i = 0; i < front.size(); i++)
  {
    size_t ind = front[i];

    pa = (calculatedObjectives[ind] - pointA);
    double t = arma::dot(pa, ba) / arma::dot(ba, ba);
    distancesTemp[i] = arma::accu(arma::pow((pa - t * ba), 2));
  }
  distances = distancesTemp;
}

//! Sort population into Pareto fronts.
template<typename MatType>
inline void AGEMOEA::FastNonDominatedSort(
    std::vector<std::vector<size_t> >& fronts,
    std::vector<size_t>& ranks,
    std::vector<arma::Col<typename MatType::elem_type> >& calculatedObjectives)
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

//! Check if a candidate Pareto dominates another candidate.
template<typename MatType>
inline bool AGEMOEA::Dominates(
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
    else if (calculatedObjectives[candidateP](i) <
        calculatedObjectives[candidateQ](i))
      atleastOneBetter = true;
  }

  return allBetterOrEqual && atleastOneBetter;
}

//! Assign diversity score for a given point and the selected set.
template <typename MatType>
inline typename MatType::elem_type AGEMOEA::DiversityScore(
    std::set<size_t>& selected,
    const MatType& pairwiseDistance,
    size_t S)
{
  typedef typename MatType::elem_type ElemType;
  ElemType m = arma::datum::inf;
  ElemType m1 = arma::datum::inf;
  std::set<size_t>::iterator it;
  for (it = selected.begin(); it != selected.end(); it++)
  {
    if (*it == S){ continue; }
    if (pairwiseDistance(S, *it) < m)
    {
      m1 = m;
      m = pairwiseDistance(S, *it);
    }
    else if (pairwiseDistance(S, *it) < m1)
    {
      m1 = pairwiseDistance(S, *it);
    }
  }
  m1 = (m1 == arma::datum::inf) ? 0 : m1;
  m = (m == arma::datum::inf) ? 0 : m;
  return m + m1;
}

//! Assign survival score for a front of the population.
template <typename MatType>
inline void AGEMOEA::SurvivalScoreAssignment(
    const std::vector<size_t>& front,
    const arma::Col<typename MatType::elem_type>& idealPoint,
    std::vector<arma::Col<typename MatType::elem_type>>& calculatedObjectives,
    std::vector<typename MatType::elem_type>& survivalScore,
    arma::Col<typename MatType::elem_type>& normalize,
    typename MatType::elem_type& dimension,
    size_t fNum)
{
  typedef typename MatType::elem_type ElemType;

  // Calculations for the first front.
  if (fNum == 0)
  {
    if (front.size() < numObjectives)
    {
      dimension = 1;
      arma::Row<size_t> extreme(numObjectives, arma::fill::zeros);
      NormalizeFront<MatType>(calculatedObjectives, normalize, front, extreme);
      return;
    }

    for (size_t index = 0; index < front.size(); index++)
    {
      calculatedObjectives[front[index]] = calculatedObjectives[front[index]]
          - idealPoint;
    }

    arma::Row<size_t> extreme(numObjectives, arma::fill::zeros);
    FindExtremePoints<MatType>(extreme, calculatedObjectives, front);
    NormalizeFront<MatType>(calculatedObjectives, normalize, front, extreme);

    for (size_t index = 0; index < front.size(); index++)
    {
      calculatedObjectives[front[index]] = calculatedObjectives[front[index]]
          / normalize;
    }

    std::set<size_t> selected;
    std::set<size_t> remaining;

    // Create the selected and remaining sets.
    for (size_t index: extreme)
    {
      selected.insert(index);
      survivalScore[front[index]] = arma::Datum<ElemType>::inf;
    }

    dimension = GetGeometry<MatType>(calculatedObjectives, front, extreme);
    for (size_t i = 0; i < front.size(); i++)
    {
      if (selected.count(i) == 0)
      {
        remaining.insert(i);
      }
    }

    arma::Mat<ElemType> pairwise(front.size(), front.size(), arma::fill::zeros);
    PairwiseDistance<MatType>(pairwise, calculatedObjectives, front, dimension);
    arma::Row<typename MatType::elem_type> value(front.size(),
        arma::fill::zeros);

    // Calculate the diversity and proximity score.
    for (size_t i = 0; i < front.size(); i++)
    {
      pairwise.col(i) = pairwise.col(i) / std::pow(accu(pow(
        arma::abs(calculatedObjectives[front[i]]), dimension)), 1 / dimension);
    }

    while (remaining.size() > 0)
    {
      std::set<size_t>::iterator it;
      value = value.fill(-1);
      for (it = remaining.begin(); it != remaining.end(); it++)
      {
        value[*it] = DiversityScore<MatType>(selected, pairwise, *it);
      }
      size_t index = arma::index_max(value);
      survivalScore[front[index]] = value[index];
      selected.insert(index);
      remaining.erase(index);
    }
  }

  // Calculations for the other fronts.
  else
  {
    for (size_t i = 0; i < front.size(); i++)
    {
      calculatedObjectives[front[i]] =
          (calculatedObjectives[front[i]]) / normalize;
      survivalScore[front[i]] =  1 / std::pow(accu(pow(abs(
          calculatedObjectives[front[i]] - idealPoint), dimension)),
              1 / dimension);
    }
  }
}

//! Comparator for survival score based sorting.
template<typename MatType>
inline bool AGEMOEA::SurvivalScoreOperator(
    size_t idxP,
    size_t idxQ,
    const std::vector<size_t>& ranks,
    const std::vector<typename MatType::elem_type>& survivalScore)
{
  if (ranks[idxP] < ranks[idxQ])
    return true;
  else if (ranks[idxP] == ranks[idxQ] && survivalScore[idxP] > survivalScore[idxQ])
    return true;

  return false;
}

} // namespace ens

#endif
