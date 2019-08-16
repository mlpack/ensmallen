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
          const double crossoverProb,
          const double distrIndex,
          const size_t numPartitions = 0);

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

  //! Get the distribution index.
  double DistIndex() const { return distrIndex; }
  //! Set the distribution index.
  double& DistrIndex() { return distrIndex; }

  //! Get the referemce set.
  arma::cube ReferenceSet() const { return referenceSet; }
  // Load reference set.
  arma::cube& ReferenceSet();

 private:
  // Sort the population into non-dominated fronts.
  void NonDominatedSorting(const arma::mat& fitnessValues,
                           std::vector<std::vector<size_t>>& fronts);

  // Mate to create a new population.
  arma::cube Mate(arma::cube& population);

  // Find reference points.
  void FindReferencePoints(std::vector<arma::mat>& referenceVec,
                           arma::mat& refPoint,
                           size_t numPartitions,
                           size_t beta,
                           size_t depth);

  //! The number of members in a population.
  size_t populationSize;

  //! The maximum number of generations.
  size_t maxGenerations;

  //! The probability of crossover.
  double crossoverProb;

  //! The distribution index for SBX crossover.
  double distrIndex;

  //! The number of partitions.
  size_t numPartitions;

  //! Reference set.
  arma::cube referenceSet;

  //! Boolean denoting whether or not the reference set is user defined.
  bool userDefinedSet;
};

} // namespace ens

// Include implementation.
#include "nsgaiii_impl.hpp"

#endif
