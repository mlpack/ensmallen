/**
 * @file cne_impl.hpp
 * @author Marcus Edel
 * @author Kartik Nighania
 *
 * Conventional Neural Evolution
 * An optimizer that works like biological evolution which selects best
 * candidates based on their fitness scores and creates new generation by
 * mutation and crossover of population.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CNE_CNE_IMPL_HPP
#define ENSMALLEN_CNE_CNE_IMPL_HPP

#include "cne.hpp"

namespace ens {

inline CNE::CNE(const size_t populationSize,
                const size_t maxGenerations,
                const double mutationProb,
                const double mutationSize,
                const double selectPercent,
                const double tolerance,
                const double objectiveChange) :
    populationSize(populationSize),
    maxGenerations(maxGenerations),
    mutationProb(mutationProb),
    mutationSize(mutationSize),
    selectPercent(selectPercent),
    tolerance(tolerance),
    objectiveChange(objectiveChange),
    numElite(0),
    elements(0)
{ /* Nothing to do here. */ }

//! Optimize the function.
template<typename DecomposableFunctionType, typename MatType>
typename MatType::elem_type CNE::Optimize(DecomposableFunctionType& function,
                                          MatType& iterate)
{
  // TODO: check function type
  // TODO: disable sp_mat

  typedef typename MatType::elem_type ElemType;

  // Vector of fitness values corresponding to each candidate.
  MatType fitnessValues;
  //! Index of sorted fitness values.
  arma::uvec index;

  // Make sure for evolution to work at least four candidates are present.
  if (populationSize < 4)
  {
    throw std::logic_error("CNE::Optimize(): population size should be at least"
        " 4!");
  }

  // Find the number of elite canditates from population.
  numElite = floor(selectPercent * populationSize);

  // Making sure we have even number of candidates to remove and create.
  if ((populationSize - numElite) % 2 != 0)
    numElite--;

  // Terminate if two parents can not be created.
  if (numElite < 2)
  {
    throw std::logic_error("CNE::Optimize(): unable to select two parents. "
        "Increase selection percentage.");
  }

  // Terminate if at least two childs are not possible.
  if ((populationSize - numElite) < 2)
  {
    throw std::logic_error("CNE::Optimize(): no space to accomodate even 2 "
        "children. Increase population size.");
  }

  // Set the population size and fill random values [0,1].
  std::vector<MatType> population;
  for (size_t i = 0 ; i < populationSize; ++i)
    population.push_back(arma::randu<MatType>(iterate.n_rows, iterate.n_cols));

  // Store the number of elements in the objective matrix.
  elements = iterate.n_rows * iterate.n_cols;

  // Initialize helper variables.
  fitnessValues.set_size(populationSize);

  Info << "CNE initialized successfully. Optimization started."
      << std::endl;

  // Find the fitness before optimization using given iterate parameters.
  size_t lastBestFitness = function.Evaluate(iterate);

  // Iterate until maximum number of generations is obtained.
  for (size_t gen = 1; gen <= maxGenerations; gen++)
  {
    // Calculating fitness values of all candidates.
    for (size_t i = 0; i < populationSize; i++)
    {
       // Select a candidate and insert the parameters in the function.
       iterate = population[i];

       // Find fitness of candidate.
       fitnessValues[i] = function.Evaluate(iterate);
    }

    Info << "Generation number: " << gen << " best fitness = "
        << fitnessValues.min() << std::endl;

    // Create next generation of species.
    Reproduce(population, fitnessValues, index);

    // Check for termination criteria.
    if (tolerance >= fitnessValues.min())
    {
      Info << "CNE::Optimize(): terminating. Given fitness criteria "
          << tolerance << " > " << fitnessValues.min() << "." << std::endl;
      break;
    }

    // Check for termination criteria.
    if (lastBestFitness - fitnessValues.min() < objectiveChange)
    {
      Info << "CNE::Optimize(): terminating. Fitness history change "
          << (lastBestFitness - fitnessValues.min())
          << " < " << objectiveChange << "." << std::endl;
      break;
    }

    // Store the best fitness of present generation.
    lastBestFitness = fitnessValues.min();
  }

  // Set the best candidate into the network parameters.
  iterate = population[index(0)];

  return function.Evaluate(iterate);
}

//! Reproduce candidates to create the next generation.
template<typename MatType>
inline void CNE::Reproduce(std::vector<MatType>& population,
                           const MatType& fitnessValues,
                           arma::uvec& index)
{
  // Sort fitness values. Smaller fitness value means better performance.
  index = arma::sort_index(fitnessValues);

  // First parent.
  size_t mom;

  // Second parent.
  size_t dad;

  for (size_t i = numElite; i < populationSize - 1; i++)
  {
    // Select 2 different parents from elite group randomly [0, numElite).
    mom = arma::as_scalar(arma::randi<arma::uvec>(
          1, arma::distr_param(0, numElite - 1)));

    dad = arma::as_scalar(arma::randi<arma::uvec>(
          1, arma::distr_param(0, numElite - 1)));

    // Making sure both parents are not the same.
    if (mom == dad)
    {
      if (dad != numElite - 1)
      {
        dad++;
      }
      else
      {
        dad--;
      }
    }

    // Parents generate 2 children replacing the dropped-out candidates.
    // Also finding the index of these candidates in the population matrix.
    Crossover(population, index[mom], index[dad], index[i], index[i + 1]);
  }

  // Mutating the weights with small noise values.
  // This is done to bring change in the next generation.
  Mutate(population, index);
}

//! Crossover parents to create new children.
template<typename MatType>
inline void CNE::Crossover(std::vector<MatType>& population,
                           const size_t mom,
                           const size_t dad,
                           const size_t child1,
                           const size_t child2)
{
  // Replace the candidates with parents at their place.
  population[child1] = population[mom];
  population[child2] = population[dad];

  // Randomly alter mom and dad genome weights to get two different children.
  for (size_t i = 0; i < elements; i++)
  {
    // Using it to alter the weights of the children.
    const double random = arma::randu<typename MatType::elem_type>();
    if (random > 0.5)
    {
      population[child1](i) = population[mom](i);
      population[child2](i) = population[dad](i);
    }
    else
    {
      population[child1](i) = population[dad](i);
      population[child2](i) = population[mom](i);
    }
  }
}

//! Modify weights with some noise for the evolution of next generation.
template<typename MatType>
inline void CNE::Mutate(std::vector<MatType>& population, arma::uvec& index)
{
  // Mutate the whole matrix with the given rate and probability.
  // The best candidate is not altered.
  for (size_t i = 1; i < populationSize; i++)
  {
    population[index(i)] += (arma::randu<MatType>(
        population[index(i)].n_rows, population[index(i)].n_cols) <
            mutationProb) %
        (mutationSize * arma::randn<MatType>(population[index(i)].n_rows,
                                             population[index(i)].n_cols));
  }
}

} // namespace ens

#endif
