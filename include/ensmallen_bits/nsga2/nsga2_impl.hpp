/**
 * @file nsga2_impl.hpp
 * @author Sayan Goswami
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */


#ifndef ENSMALLEN_NSGA2_NSGA2_HPP
#define ENSMALLEN_NSGA2_NSGA2_HPP

#include "nsga2.hpp"
#include <set>
#include <vector>
#include <map>

namespace ens {

inline NSGA2::NSGA2(const size_t populationSize,
                    const size_t maxGenerations,
                    const double crossoverProb,
                    const double epsilon) :
    populationSize(populationSize),
    maxGenerations(maxGenerations),
    crossoverProb(crossoverProb),
    epsilon(epsilon)
{ /* Nothing to do here. */ }

template<typename ArbitraryFunctionType,
         typename MatType,
         typename... CallbackTypes>
typename MatType::elem_type NSGA2::Optimize(std::vector<ArbitraryFunctionType>& objectives,
                                            MatType& iterate,
                                            CallbackTypes&&... callbacks)
{
// TODO: Implement
}

template<typename MatType>
inline void NSGA2::FastNonDominatedSort(std::vector<MatType>& population,
                                        std::vector<std::vector<int> >& fronts,
                                        std::vector<int>& ranks,
                                        std::vector<std::vector<double> > calculatedObjectives)
{
  std::map<int, int> dominationCount;
  std::map<int, std::set<int> > dominated;

  for (int p = 0; p < populationSize; p++) {
    dominated[p] = std::set<int>();
    dominationCount[p] = 0;

    for(int q=0; q < populationSize; q++) {
      if (Dominates(calculatedObjectives, p, q)) {
        dominated[p].insert(q);
      }
      else if (Dominates(calculatedObjectives, q, p)) {
        dominationCount[p] += 1
      }
    }

    if (dominationCount[p] == 0) {
      rank[p] = 0;
      fronts[0].append(p)
    }
  }

  int i = 0;

  while (fronts[i].size() > 0) {
    std::vector<int> nextFront;

    for (int p: fronts[i]) {
      for (int q: dominated[p]) {
        dominationCount[q]--;

        if (dominationCount[q] == 0) {
          rank[q] = i + 1;
          nextFront.push_back(q);
        }
      }
    }

    i++;
    fronts[i].push_back(nextFront);
  }
}

bool Dominates(std::vector<std::vector<double> > calculatedObjectives,
               int candidateP,
               int candidateQ)
{
  bool all_better_or_equal = true;
  bool atleast_one_better = false;
  int n_objectives = calculatedObjectives.size();

  for (int i = 0; i < n_objectives; i++) {
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

#endif
