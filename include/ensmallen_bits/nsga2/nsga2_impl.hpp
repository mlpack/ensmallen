/**
 * @file nsga2_impl.hpp
 * @author Sayan Goswami
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */


#ifndef ENSMALLEN_NSGA2_NSGA2_IMPL_HPP
#define ENSMALLEN_NSGA2_NSGA2_IMPL_HPP

#include "nsga2.hpp"
#include <set>
#include <vector>
#include <map>

namespace ens {

inline NSGA2::NSGA2(const size_t populationSize,
                    const size_t maxGenerations,
                    const double crossoverProb,
                    const double mutationProb,
                    const double mutationStrength,
                    const double epsilon) :
    populationSize(populationSize),
    maxGenerations(maxGenerations),
    crossoverProb(crossoverProb),
    mutationProb(mutationProb),
    mutationStrength(mutationStrength),
    epsilon(epsilon)
{ /* Nothing to do here. */ }

template<typename MultiobjectiveFunctionType,
         typename MatType,
         typename... CallbackTypes>
std::vector<MatType> NSGA2::Optimize(MultiobjectiveFunctionType& objectives,
                        MatType& iterate,
                        CallbackTypes&&... callbacks)
{
  std::vector<std::vector<double> > calculatedObjectives;
  calculatedObjectives.resize(populationSize);

  std::vector<MatType> population;
  std::vector<std::vector<size_t> > fronts;
  std::vector<double> crowdingDistance;
  std::vector<size_t> ranks;

  bool terminate = false;

  for (size_t i = 0; i < populationSize; i++) {
    population.push_back(arma::randu<MatType>(iterate.n_rows,
        iterate.n_cols) + iterate);
  }

  EvaluateObjectives(population, objectives, calculatedObjectives);

  terminate |= Callback::BeginOptimization(*this, objectives, iterate, callbacks...);

  for (size_t generation = 1; generation <= maxGenerations && !terminate; generation++) {
    terminate |= Callback::StepTaken(*this, objectives, iterate, callbacks...);

    Info << "NSGA2::Optimize() Generation: " << generation << std::endl;
    // have P_t, generate G_t using P_t
    Info << "NSGA2::Optimize() BinaryTournamentSelection" << std::endl;
    BinaryTournamentSelection(population);

    // evaluate objectives
    Info << "NSGA2::Optimize() EvaluateObjectives" << std::endl;
    calculatedObjectives.resize(population.size());
    EvaluateObjectives(population, objectives, calculatedObjectives);

    // perform fast non dominated sort on $$ P_t \cup G_t $$
    Info << "NSGA2::Optimize() FastNonDominatedSort" << std::endl;
    ranks.resize(population.size());
    FastNonDominatedSort(fronts, ranks, calculatedObjectives);

    // perform crowding distance assignment
    Info << "NSGA2::Optimize() CrowdingDistanceAssignment" << std::endl;
    crowdingDistance.resize(population.size());
    for (size_t fNum = 0; fNum < fronts.size(); fNum++) {
      CrowdingDistanceAssignment(fronts[fNum], objectives, crowdingDistance);
    }

    // sort based on crowding distance
    Info << "NSGA2::Optimize() Sort(Crowding Distance)" << std::endl;
    std::sort(population.begin(),
              population.end(),
              [this, ranks, crowdingDistance, population](MatType candidateP,
                                                          MatType candidateQ){
                size_t idxP, idxQ;
                for(int i = 0; i < population.size(); i++) {
                  if (arma::approx_equal(population[i], candidateP, "absdiff", epsilon)) {
                    idxP = i;
                  }
                  if (arma::approx_equal(population[i], candidateQ, "absdiff", epsilon)) {
                    idxQ = i;
                  }
                }

                return CrowdingOperator(idxP,
                                        idxQ,
                                        ranks,
                                        crowdingDistance);
              }
    );

    // yeild new population P_{t+1}
    Info << "NSGA2::Optimize() Get P(t+1)" << std::endl;
    population.resize(populationSize);
  }

  std::vector<MatType> bestFront;

  for(size_t f: fronts[0]) {
    bestFront.push_back(population[f]);
  }

  Callback::EndOptimization(*this, objectives, iterate, callbacks...);
  return bestFront;
}

template<typename MultiobjectiveFunctionType,
         typename MatType>
inline void NSGA2::EvaluateObjectives(std::vector<MatType> population,
                                      MultiobjectiveFunctionType objectives,
                                      std::vector<std::vector<double> >& calculatedObjectives)
{
  for (size_t i = 0; i < populationSize; i++) {
    calculatedObjectives[i] = objectives.Evaluate(population[i]);
  }
}

template<typename MatType>
inline void NSGA2::BinaryTournamentSelection(std::vector<MatType>& population)
{
  std::vector<MatType> children;

  while (children.size() < population.size()) {
    size_t indexA = arma::randi<size_t>(arma::distr_param(0, populationSize - 1));
    size_t indexB = arma::randi<size_t>(arma::distr_param(0, populationSize - 1));

    if (indexA == indexB) {
      if (indexB < populationSize - 1) {
        indexB++;
      }
      else {
        indexB--;
      }
    }

    MatType parentA = population[indexA];
    MatType parentB = population[indexB];

    MatType childA = parentA, childB = parentB;

    Info << "NSGA2::BinaryTournamentSelection() Crossover" << std::endl;
    Crossover(childA, childB, parentA, parentB);

    Info << "NSGA2::BinaryTournamentSelection() Mutate(A)" << std::endl;
    Mutate(childA);
    Info << "NSGA2::BinaryTournamentSelection() Mutate(B)" << std::endl;
    Mutate(childB);

    children.push_back(childA);
    children.push_back(childB);
  }

  population.reserve(population.size() + children.size());
  population.insert(std::end(population), std::begin(children), std::end(children));
}

template<typename MatType>
inline void NSGA2::Crossover(MatType& childA,
                MatType& childB,
                MatType parentA,
                MatType parentB)
{
  // crossover indices
  auto idx = arma::randu<MatType>(childA.n_rows, childA.n_cols) < crossoverProb;

  childA = parentA % idx + parentB % (1 - idx);
  childB = parentA % (1 - idx) + parentA % idx;
}

template<typename MatType>
inline void NSGA2::Mutate(MatType& child)
{
  child += (arma::randu<MatType>(child.n_rows, child.n_cols) < mutationProb) %
           (mutationStrength * arma::randn<MatType>(child.n_rows, child.n_cols));
}

inline void NSGA2::FastNonDominatedSort(std::vector<std::vector<size_t> >& fronts,
                                        std::vector<size_t>& ranks,
                                        std::vector<std::vector<double> > calculatedObjectives)
{
  std::map<size_t, size_t> dominationCount;
  std::map<size_t, std::set<size_t> > dominated;

  // reset and intialize fronts
  fronts.clear();
  fronts.push_back(std::vector<size_t>());

  for (size_t p = 0; p < populationSize; p++) {
    dominated[p] = std::set<size_t>();
    dominationCount[p] = 0;

    for(size_t q=0; q < populationSize; q++) {
      if (Dominates(calculatedObjectives, p, q)) {
        dominated[p].insert(q);
      }
      else if (Dominates(calculatedObjectives, q, p)) {
        dominationCount[p] += 1;
      }
    }

    if (dominationCount[p] == 0) {
      ranks[p] = 0;
      fronts[0].push_back(p);
    }
  }

  size_t i = 0;

  while (fronts[i].size() > 0) {
    std::vector<size_t> nextFront;

    for (size_t p: fronts[i]) {
      for (size_t q: dominated[p]) {
        dominationCount[q]--;

        if (dominationCount[q] == 0) {
          ranks[q] = i + 1;
          nextFront.push_back(q);
        }
      }
    }

    i++;
    fronts.push_back(nextFront);
  }
}

inline bool NSGA2::Dominates(std::vector<std::vector<double> > calculatedObjectives,
               size_t candidateP,
               size_t candidateQ)
{
  bool all_better_or_equal = true;
  bool atleast_one_better = false;
  size_t n_objectives = calculatedObjectives.size();

  for (size_t i = 0; i < n_objectives; i++) {
    if (calculatedObjectives[candidateP][i] > calculatedObjectives[candidateQ][i]) {
      // p.i is worse than q.i for the i-th objective function
      all_better_or_equal = false;
    }

    if (calculatedObjectives[candidateP][i] < calculatedObjectives[candidateQ][i]) {
      // p.i is better than q.i for the i-th objective function
      atleast_one_better = true;
    }
  }

  return all_better_or_equal and atleast_one_better;
}

template<typename MultiobjectiveFunctionType>
inline void NSGA2::CrowdingDistanceAssignment(std::vector<size_t> front,
                                       MultiobjectiveFunctionType objectives,
                                       std::vector<double>& crowdingDistance)
{
  if (front.size() > 0) {
    for(size_t elem: front) {
      crowdingDistance[elem] = 0;
    }

    size_t fSize = front.size();

    for(size_t m = 0; m < objectives.NumObjectives(); m++) {
      crowdingDistance[front[0]] = objectives.GetMaximum(m);
      crowdingDistance[front[fSize - 1]] = objectives.GetMaximum(m);

      for(size_t i = 1; i < fSize - 1 ; i++) {
        crowdingDistance[front[i]] += (crowdingDistance[front[i-1]] - crowdingDistance[front[i+1]])/(objectives.GetMaximum(m) - objectives.GetMinimum(m));
      }
    }
  }
}

inline bool NSGA2::CrowdingOperator(size_t idxP,
                             size_t idxQ,
                             std::vector<size_t> ranks,
                             std::vector<double> crowdingDistance)
{
  if (ranks[idxP] < ranks[idxQ]) {
    return true;
  }
  else if (ranks[idxP] < ranks[idxQ] && crowdingDistance[idxP] > crowdingDistance[idxQ]) {
    return true;
  }

  return false;
}

}

#endif
