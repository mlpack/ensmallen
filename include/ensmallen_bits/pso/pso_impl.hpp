/**
 * @file pso_impl.hpp
 * @author Chintan Soni
 * @author Suryoday Basak
 *
 * Implementation of the particle swarm optimization algorithm.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PSO_PSO_IMPL_HPP
#define ENSMALLEN_PSO_PSO_IMPL_HPP

#include "pso.hpp"
#include <ensmallen_bits/function.hpp>
#include <queue>

namespace ens {
/* After the velocity of each particle is updated at the end of each iteration
 * in PSO, the position of particle i (in iteration j) is updated as:
 * 
 * \f[
 *     p_{i, j + 1} = p_{i, j} + v_{i, j}
 * \f]
 *
 * pso_impl.hpp implements the position-updating procedure. The velocity update
 * may be done using either the lbest or gbest methods, by using the
 * appropriate templates.
 */

//! Optimize the function (minimize).
template<typename VelocityUpdatePolicy,
         typename InitPolicy>
template<typename FunctionType>
double PSOType<VelocityUpdatePolicy, InitPolicy>::Optimize(
    FunctionType& function, arma::mat& iterate)
{
  /* The following cast is made to make sure that PSO can run on arbitrary
   * continuous functions and to ensure that we have all the necessary functions.
   */
  typedef Function<FunctionType> ArbitraryFunctionType;
  ArbitraryFunctionType& f(static_cast<ArbitraryFunctionType&>(function));
  
  // Make sure we have the methods that we need.
  traits::CheckEvaluate<ArbitraryFunctionType>();

  // Initialize particles using the init policy.
  initPolicy.Initialize(iterate,
      numParticles,
      lowerBound,
      upperBound,
      particlePositions,
      particleVelocities,
      particleFitnesses,
      particleBestPositions,
      particleBestFitnesses);

  // Initialize the update policy.
  velocityUpdatePolicy.Initialize(exploitationFactor,
      explorationFactor,
      numParticles,
      iterate);

  // Calculate initial fitness of population.
  for (size_t i = 0; i < numParticles; i++)
  {
    // Calculate fitness value.
    particleFitnesses(i) = f.Evaluate(particlePositions.slice(i));
    particleBestFitnesses(i) = particleFitnesses(i);
  }
  
  // Declare queue to keep track of improvements over a number of iterations.
  queue <double> performanceHorizon;
  // Variable to store the position of the best particle.
  size_t bestParticle = 0;
  // Find the best fitness.
  double bestFitness = std::numeric_limits<double>::max();
  
  // Run PSO for horizonSize number of iterations.
  // This will allow the performanceHorizon to be updated.
  // With some initial values in this, we may proceed with the remaining steps
  // in the PSO method.
  // The performanceHorizon will be updated with the best particle
  // in a FIFO manner.
  for (size_t i = 0; i < horizonSize; i++)
  {
    // Calculate fitness and evaluate personal best.
    for (size_t j = 0; j < numParticles; j++)
    {
      particleFitnesses(j) = f.Evaluate(particlePositions.slice(j));
      // Compare and copy fitness and position to particle best.
      if (particleFitnesses(j) < particleBestFitnesses(j))
      {
        particleBestFitnesses(j) = particleFitnesses(j);
        particleBestPositions.slice(j) = particlePositions.slice(j);
      }
    }

    // Evaluate local best and update velocity.
    velocityUpdatePolicy.Update(particlePositions,
        particleVelocities,
        particleBestPositions,
        particleBestFitnesses);

    // In-place update of particle positions.
    particlePositions += particleVelocities;

    // Find the best particle.
    for (size_t j = 0; j < numParticles; j++)
    {
      if (particleBestFitnesses(j) < bestFitness)
      {
        bestParticle = j;
        bestFitness = particleBestFitnesses(bestParticle);
      }
    }

    // Append bestFitness to performanceHorizon.
    performanceHorizon.push(bestFitness);
  }

  // Run the remaining iterations of PSO.
  for (size_t i = 0; i < maxIterations - horizonSize; i++)
  {
    // Check if there is any improvement over the horizon.
    // If there is no significant improvement, terminate.
    if (performanceHorizon.front() - performanceHorizon.back() < impTolerance)
      break;
    
    // Calculate fitness and evaluate personal best.
    for (size_t j = 0; j < numParticles; j++)
    {
      particleFitnesses(j) = f.Evaluate(particlePositions.slice(j));
      // Compare and copy fitness and position to particle best.
      if (particleFitnesses(j) < particleBestFitnesses(j))
      {
        particleBestFitnesses(j) = particleFitnesses(j);
        particleBestPositions.slice(j) = particlePositions.slice(j);
      }
    }

    // Evaluate local best and update velocity.
    velocityUpdatePolicy.Update(particlePositions,
        particleVelocities,
        particleBestPositions,
        particleBestFitnesses);

    // In-place update of particle positions.
    particlePositions += particleVelocities;

    //Find the best particle.
    for (size_t j = 0; j < numParticles; j++)
    {
      if (particleBestFitnesses(j) < bestFitness)
      {
        bestParticle = j;
        bestFitness = particleBestFitnesses(bestParticle);
      }
    }

    // Pop the oldest value from performanceHorizon.
    performanceHorizon.pop();
    // Push most recent bestFitness to performanceHorizon.
    performanceHorizon.push(bestFitness);
  }

  // Copy results back.
  iterate = particleBestPositions.slice(bestParticle);

  return bestFitness;
}

} // ens

#endif
