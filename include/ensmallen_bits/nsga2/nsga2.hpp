/**
 * @file nsga2.hpp
 * @author Sayan Goswami
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_NSGA2_NSGA2_HPP
#define ENSMALLEN_NSGA2_NSGA2_HPP

namespace ens {

class NSGA2 {
 public:
  NSGA2(const size_t populationSize = 100,
        const size_t maxGenerations = 2000,
        const double crossoverProb = 0.6,
        const double mutationProb = 0.3,
        const double mutationStrength = 1e-3,
        const double epsilon = 1e-6);

  template<typename MultiobjectiveFunctionType,
           typename MatType,
           typename... CallbackTypes>
  std::vector<MatType> Optimize(MultiobjectiveFunctionType& objectives,
                                MatType& iterate,
                                CallbackTypes&&... callbacks);

  size_t PopulationSize() const { return populationSize; }
  size_t& PopulationSize() { return populationSize; }

  size_t MaxGenerations() const { return maxGenerations; }
  size_t& MaxGenerations() { return maxGenerations; }

  double CrossoverRate() const { return crossoverProb; }
  double& CrossoverRate() { return crossoverProb; }

  double MutationProbability() const { return mutationProb; }
  double& MutationProbability() { return mutationProb; }

  double MutationSize() const { return mutationStrength; }
  double& MutationSize() { return mutationStrength; }

  double Epsilon() const { return epsilon; }
  double& Epsilon() { return epsilon; }


 private:
  template<typename MultiobjectiveFunctionType,
           typename MatType>
  void EvaluateObjectives(std::vector<MatType> population,
                          MultiobjectiveFunctionType objectives,
                          std::vector<arma::vec>& calculatedObjectives);

  template<typename MatType>
  void BinaryTournamentSelection(std::vector<MatType>& population);

  template<typename MatType>
  void Crossover(MatType& childA,
                 MatType& childB,
                 MatType parentA,
                 MatType parentB);

  template<typename MatType>
  void Mutate(MatType& child);

  void FastNonDominatedSort(std::vector<std::vector<size_t> >& fronts,
                            std::vector<size_t>& ranks,
                            std::vector<arma::vec> calculatedObjectives);

  bool Dominates(std::vector<arma::vec> calculatedObjectives,
                 size_t candidateP,
                 size_t candidateQ);

  template<typename MultiobjectiveFunctionType>
  void CrowdingDistanceAssignment(std::vector<size_t> front,
                                  MultiobjectiveFunctionType objectives,
                                  std::vector<double>& crowdingDistance);

  bool CrowdingOperator(size_t idxP,
                        size_t idxQ,
                        std::vector<size_t> ranks,
                        std::vector<double> crowdingDistance);

  size_t populationSize;
  size_t maxGenerations;
  double crossoverProb;
  double mutationProb;
  double mutationStrength;
  double epsilon;
};

} // namespace ens

// Include implementation.
#include "nsga2_impl.hpp"

#endif
