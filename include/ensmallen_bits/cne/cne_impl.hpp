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
                const double tolerance) :
    populationSize(populationSize),
    maxGenerations(maxGenerations),
    mutationProb(mutationProb),
    mutationSize(mutationSize),
    selectPercent(selectPercent),
    tolerance(tolerance),
    numElite(0),
    elements(0)
{ /* Nothing to do here. */ }

//! Optimize the function.
template<typename ArbitraryFunctionType,
         typename MatType,
         typename... CallbackTypes>
typename MatType::elem_type CNE::Optimize(ArbitraryFunctionType& function,
                                          MatType& iterateIn,
                                          CallbackTypes&&... callbacks)
{
  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;

  // Make sure that we have the methods that we need.  Long name...
  traits::CheckArbitraryFunctionTypeAPI<ArbitraryFunctionType,
      BaseMatType>();
  RequireDenseFloatingPointType<BaseMatType>();

  // Vector of fitness values corresponding to each candidate.
  BaseMatType fitnessValues;
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

  BaseMatType& iterate = (BaseMatType&) iterateIn;

  // Generate the population based on a Gaussian distribution around the given
  // starting point.
  std::vector<BaseMatType> population;
  for (size_t i = 0 ; i < populationSize; ++i)
  {
    population.push_back(arma::randn<BaseMatType>(iterate.n_rows,
        iterate.n_cols) + iterate);
  }

  // Store the number of elements in the objective matrix.
  elements = iterate.n_rows * iterate.n_cols;

  // Initialize helper variables.
  fitnessValues.set_size(populationSize);

  // Controls early termination of the optimization process.
  bool terminate = false;

  Info << "CNE initialized successfully. Optimization started."
      << std::endl;

  // Find the fitness before optimization using given iterate parameters.
  ElemType lastBestFitness = function.Evaluate(iterate);
  terminate |= Callback::Evaluate(*this, function, iterate, lastBestFitness,
      callbacks...);

  // Iterate until maximum number of generations is obtained.
  Callback::BeginOptimization(*this, function, iterate, callbacks...);
  for (size_t gen = 1; gen <= maxGenerations && !terminate; gen++)
  {
    // Calculating fitness values of all candidates.
    for (size_t i = 0; i < populationSize; i++)
    {
        // Select a candidate and insert the parameters in the function.
        iterate = population[i];
        terminate |= Callback::StepTaken(*this, function, iterate,
            callbacks...);

        // Find fitness of candidate.
        fitnessValues[i] = function.Evaluate(iterate);

        terminate |= Callback::Evaluate(*this, function, iterate,
            fitnessValues[i], callbacks...);
    }

    Info << "Generation number: " << gen << " best fitness = "
        << fitnessValues.min() << std::endl;

    // Create next generation of species.
    Reproduce(population, fitnessValues, index);

    // Check for termination criteria.
    if (std::abs(lastBestFitness - fitnessValues.min()) < tolerance)
    {
      Info << "CNE: minimized within tolerance " << tolerance << "; "
            << "terminating optimization." << std::endl;
      break;
    }

    // Store the best fitness of present generation.
    lastBestFitness = fitnessValues.min();
  }

  // Set the best candidate into the network parameters.
  iterateIn = population[index(0)];

  // The output of the callback doesn't matter because the optimization is
  // finished.
  const ElemType objective = function.Evaluate(iterate);
  (void) Callback::Evaluate(*this, function, iterate, objective, callbacks...);

  Callback::EndOptimization(*this, function, iterate, callbacks...);
  return objective;
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
    population[index(i)] += (arma::randu<MatType>(population[index(i)].n_rows,
        population[index(i)].n_cols) < mutationProb) %
        (mutationSize * arma::randn<MatType>(population[index(i)].n_rows,
        population[index(i)].n_cols));
  }
}

} // namespace ens

#endif
