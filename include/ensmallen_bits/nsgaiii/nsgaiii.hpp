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
          const double crossoverProb);

  template<typename MultiObjectiveFunctionType>
  arma::cube Optimize(MultiObjectiveFunctionType& function, arma::mat& iterate);

  //! Get the population size.
  size_t PopulationSize() const { return populationSize; }
  //! Modify the population size.
  size_t& PopulationSize() { return populationSize; }

  //! Get maximum number of generations.
  size_t MaxGenerations() const { return maxGenerations; }
  //! Modify maximum number of generations.
  size_t& MaxGenerations() { return maxGenerations; }

  //! Get the probability of crossover.
  double CrossoverProb() const { return crossoverProb; }
  //! Set the probability of crossover.
  double& CrossoverProb() { return crossoverProb; }

 private:
  void NonDominatedSorting(const arma::mat& fitnessValues,
                           std::vector<std::vector<size_t>>& fronts;);

  arma::cube Mate(arma::cube& population);

  // The number of members in a population.
  size_t populationSize;

  // The maximum number of generations.
  size_t maxGenerations;

  // The probability of crossover.
  double crossoverProb;
};

} // namespace ens

// Include implementation.
#include "nsgaiii_impl.hpp"

#endif
