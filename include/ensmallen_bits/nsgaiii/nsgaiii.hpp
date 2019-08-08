/**
 * @file nsgaiii.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Non-dominated Sorting Genetic Algorithm - III (NSGA-III)
 * A multi-objective optimizer. \temp
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_NSGAIII_NSGAIII_HPP
#define ENSMALLEN_NSGAIII_NSGAIII_HPP

namespace ens {

class NSGAIII
{
 public:
  NSGAIII(const size_t populationSize,
          const size_t maxGenerations,
          const double mutationProb,
          const double mutationSize,
          const double selectPercent);

  template<typename MultiObjectiveFunctionType>
  double Optimize(MultiObjectiveFunctionType& function, arma::mat& iterate);

  //! Get the population size.
  size_t PopulationSize() const { return populationSize; }
  //! Modify the population size.
  size_t& PopulationSize() { return populationSize; }

  //! Get maximum number of generations.
  size_t MaxGenerations() const { return maxGenerations; }
  //! Modify maximum number of generations.
  size_t& MaxGenerations() { return maxGenerations; }

  //! Get the mutation probability.
  double MutationProbability() const { return mutationProb; }
  //! Modify the mutation probability.
  double& MutationProbability() { return mutationProb; }

  //! Get the mutation size.
  double MutationSize() const { return mutationSize; }
  //! Modify the mutation size.
  double& MutationSize() { return mutationSize; }

  //! Get the selection percentage.
  double SelectionPercentage() const { return selectPercent; }
  //! Modify the selection percentage.
  double& SelectionPercentage() { return selectPercent; }

 private:
  std::vector<std::vector<size_t>> NonDominatedSorting(const arma::mat& fitnessValues);

  arma::cube Mate(arma::cube& population);

  size_t populationSize;

  size_t maxGenerations;

  double mutationProb;

  double mutationSize;

  double selectPercent;

};

} // namespace ens

// Include implementation.
#include "nsgaiii_impl.hpp"

#endif
