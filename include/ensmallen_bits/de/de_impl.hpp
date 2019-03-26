/**
 * @file de_impl.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Implementation of Differential Evolution an evolutionary algorithm used for
 * global optimization of arbitrary functions.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_DE_DE_IMPL_HPP
#define ENSMALLEN_DE_DE_IMPL_HPP

#include "de.hpp"

namespace ens {

inline DE::DE(const size_t populationSize ,
              const size_t maxGenerations,
              const double crossoverRate,
              const double differentialWeight,
              const double tolerance):
    populationSize(populationSize),
    maxGenerations(maxGenerations),
    crossoverRate(crossoverRate),
    differentialWeight(differentialWeight),
    tolerance(tolerance)
{ /* Nothing to do here. */ }

//!Optimize the function
template<typename DecomposableFunctionType>
inline double DE::Optimize(DecomposableFunctionType& function,
                           arma::mat& iterate)
{
  // Population Size must be atleast 3 for DE to work.
  if (populationSize < 3)
  {
    throw std::logic_error("CNE::Optimize(): population size should be at least"
        " 3!");
  }

  // Initialize helper variables.
  fitnessValues.set_size(populationSize);
  double lastBestFitness = DBL_MAX;
  arma::mat bestElement;

  // Generate a population based on a Gaussian distribution around the given
  // starting point. Also finds the best element of the population.
  population = arma::randn(iterate.n_rows, iterate.n_cols, populationSize);
  for (size_t i = 0; i < populationSize; i++)
  {
    population.slice(i) = population.slice(i) + iterate;
    fitnessValues[i] = function.Evaluate(population.slice(i));
    if(fitnessValues[i] < lastBestFitness)
    {
      lastBestFitness = fitnessValues[i];
      bestElement = population.slice(i);
    }
  }

  // Iterate until maximum number of generations are completed.
  for (size_t gen = 0; gen < maxGenerations; gen++)
  {
    // Generate new population based on /best/1/bin strategy.
    for (size_t member = 0; member < populationSize; member++)
    {
      iterate = population.slice(member);

      // Generate two different random numbers to choose two random members.
      size_t l = 0, m = 0;
      do
      {
        l = arma::as_scalar(arma::randi<arma::uvec>(
            1, arma::distr_param(0, populationSize - 1)));
      }
      while(l == member);

      do
      {
        m = arma::as_scalar(arma::randi<arma::uvec>(
            1, arma::distr_param(0, populationSize - 1)));
      }
      while(m == member && m == l);

      // Generate new "mutant" from two randomly chosen members.
      arma::mat mutant = bestElement + differentialWeight *
          (population.slice(l) - population.slice(m));

      // Perform crossover.
      const arma::mat cr = arma::randu(iterate.n_rows);
      for (size_t it = 0; it < iterate.n_rows; it++)
      {
        if (cr[it] >= crossoverRate)
        {
          mutant[it] = iterate[it];
        }
      }

      double iterateValue = function.Evaluate(iterate);
      const double mutantValue = function.Evaluate(mutant);

      // Replace the current member if mutant is better.
      if (mutantValue < iterateValue)
      {
        iterate = mutant;
        iterateValue = mutantValue;
      }

      fitnessValues[member] = iterateValue;
      population.slice(member) = iterate;
    }

    // Check for termination criteria.
    if (std::abs(lastBestFitness - fitnessValues.min()) < tolerance)
    {
      Info << "DE: minimized within tolerance " << tolerance << "; "
            << "terminating optimization." << std::endl;
      break;
    }

    // Update helper variables.
    lastBestFitness = fitnessValues.min();
    for (size_t it = 0; it < populationSize; it++)
    {
      if (fitnessValues[it] == lastBestFitness)
      {
        bestElement = population.slice(it);
        break;
      }
    }
  }

  iterate = bestElement;
  return lastBestFitness;
}

} // namespace ens

#endif
