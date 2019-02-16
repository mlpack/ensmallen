/**
 * @file pso_impl.hpp
 * @author Chintan Soni
 * @author Suryoday Basak
 *
 * Implementation of the particle swarm optimization algorithm.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PSO_PSO_IMPL_HPP
#define ENSMALLEN_PSO_PSO_IMPL_HPP

#include "pso.hpp"

#include <ensmallen_bits/function.hpp>

namespace ens {

//! Optimize the function (minimize).
template<typename VelocityUpdatePolicy,
         typename InitPolicy>
template<typename FunctionType>
double PSOType<VelocityUpdatePolicy, InitPolicy>::Optimize(
  FunctionType& function, arma::mat& iterate)
{
  // The following cast is made to make sure that PSO can run on arbitrary continuous functions
  // and to ensure that we have all the necessary functions.
  typedef Function<FunctionType> ArbitraryFunctionType;
  ArbitraryFunctionType& f(static_cast<ArbitraryFunctionType&>(function));
  
  // Make sure we have the methods that we need.
  traits::CheckFunctionTypeAPI<ArbitraryFunctionType>();

  // Initialize particles using the init policy.
  initPolicy.InitializeParticles(
      iterate,
      numParticles,
      particlePositions,
      particleVelocities,
      particleFitnesses,
      particleBestPositions,
      particleBestFitnesses);

  // Initialize the update policy.
  velocityUpdatePolicy.Initialize(
      exploitationFactor, explorationFactor, numParticles, iterate);

  // User provided weights replacement performed here.
  for (size_t i = 0; i < numParticles; i++)
  {
    // Calculate fitness value.
    particleFitnesses(i) = f.Evaluate(particlePositions.slice(i));
    // Compare and copy fitness and position to particle best.
    if (particleFitnesses(i) < particleBestFitnesses(i))
    {
      particleBestFitnesses(i) = particleFitnesses(i);
      particleBestPositions.slice(i) = particlePositions.slice(i);
    }
  }
  //std::cout<<particleFitnesses;

  // Find the worst particle.
  size_t worstParticle = 0;
  double worstFitness = particleBestFitnesses(worstParticle);
  for (size_t i = 1; i < numParticles; i++)
  {
    if (particleBestFitnesses(i) < worstFitness)
    {
      worstParticle = i;
      worstFitness = particleBestFitnesses(worstParticle);
    }
  }

  // Replace worst particle with iterate if iterate is better.
  double iterateFitness = f.Evaluate(iterate);
  if (iterateFitness < worstFitness)
  {
    particleBestFitnesses(worstParticle) = iterateFitness;
    particlePositions.slice(worstParticle) = iterate;
    particleBestPositions.slice(worstParticle) = iterate;
  }

  size_t psoIterations = maxIterations;
  // Run PSO.
  for (size_t i = 0; i < psoIterations; i++)
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
    velocityUpdatePolicy.Update(
        particlePositions,
        particleVelocities,
        particleBestPositions,
        particleBestFitnesses);

    // In-place update of particle positions.
    particlePositions += particleVelocities;
  }

  // Find best particle.
  size_t bestParticle = 0;
  double bestFitness = particleBestFitnesses(bestParticle);
  for (size_t i = 1; i < numParticles; i++)
  {
    if (particleBestFitnesses(i) < bestFitness)
    {
      bestParticle = i;
      bestFitness = particleBestFitnesses(bestParticle);
    }
  }

  // Copy results back.
  iterate = particleBestPositions.slice(bestParticle);

  return bestFitness;
}

} // ens

#endif
