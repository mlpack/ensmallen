/**
 * @file de.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Differential Evolution
 * An evolutionary algorithm used for global optimization of arbitrary functions
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_DE_DE_IMPL_HPP
#define ENSMALLEN_DE_DE_IMPL_HPP

#include "de.hpp"
#include <random>
#include <tuple>
#include <iostream>

namespace ens {

inline DE::DE(const size_t populationSize ,
			  const size_t maxGenerations,
			  const double crossoverRate,
			  const double differentialWeight):
    populationSize(populationSize),
    maxGenerations(maxGenerations),
    crossoverRate(crossoverRate),
    differentialWeight(differentialWeight)
{ /*Nothing to do here*/ }

//!Optimize the function
template<typename DecomposableFunctionType>
inline double DE::Optimize(DecomposableFunctionType& function, arma::mat& iterate)
{
  // Population Size must be atleast 3 for DE to work
  if (populationSize < 3)
    {
      throw std::logic_error("CNE::Optimize(): population size should be at least"
          " 3!");
    }

  // Initialize helper variables
  fitnessValues.set_size(populationSize);
  double lastBestFitness = DBL_MAX;
  arma::mat bestElement;

  // Generate a population based on a Gaussian distribution around the given starting point.
  // Also finds the best element of the population
  population = arma::randn(iterate.n_rows, iterate.n_cols, populationSize);
  for(int i = 0; i < populationSize; i++)
  {
    population.slice(i) = population.slice(i) + iterate;
    fitnessValues[i] = function.Evaluate(population.slice(i));
    if(fitnessValues[i] < lastBestFitness)
    {
      lastBestFitness = fitnessValues[i];
      bestElement = population.slice(i);
    }
  }

  // Initialize random number generator
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_int_distribution<std::mt19937::result_type> dist(0,populationSize-1);

  Info << "DE initialized successfully. Optimization started."
  << std::endl;

  // Iterate until maximum number of generations are completed
  for(size_t gen = 0; gen < maxGenerations; gen++)
  {

    Info << "Generation: " << gen<< "\nBest Fitness: " << lastBestFitness << std::endl;

    // Generate new populationbased on /best/1/bin strategy
    for(size_t member = 0; member < populationSize; member++)
    {
	  iterate = population.slice(member);

	  int l=0, m=0;
	  do
	  {
	    l = dist(rng);
	  }
	  while(l == member);
	  do
	  {
	    m = dist(rng);
	  }
	  while(m == member && m == l);

	  // Generate new "mutant" from two randomly chosen members
	  arma::mat a = population.slice(l);
	  arma::mat b = population.slice(m);
	  arma::mat mutant = bestElement + differentialWeight*(a - b);

	  // Perform crossover
	  arma::mat cr = arma::randu(iterate.n_rows);
	  for(int it = 0; it<iterate.n_rows; it++)
	  {
	    if(cr[it] >= crossoverRate)
	    {
	      mutant[it] = iterate[it];
	    }
	  }

	  double iterateValue = function.Evaluate(iterate);
	  double mutantValue = function.Evaluate(mutant);

	  // Replace the current member if mutant is better
	  if(mutantValue < iterateValue)
	    {
	      iterate = mutant;
	      iterateValue = mutantValue;
	    }

	  fitnessValues[member] = iterateValue;

	  population.slice(member) = iterate;			
    }

    // Update helper variables 
    lastBestFitness = fitnessValues.min();
    for(int it = 0; it < populationSize; it++)
      if(fitnessValues[it] == lastBestFitness)
        bestElement = population.slice(it);
  }
  iterate = bestElement;
  return lastBestFitness;
}

}

#endif