/**
 * @file pso_impl.hpp
 * @author Chintan Soni
 * @author Suryoday Basak
 *
 * Implementation of the particle swarm optimization algorithm.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PSO_PSO_IMPL_HPP
#define ENSMALLEN_PSO_PSO_IMPL_HPP

#include "pso.hpp"
#include <ensmallen_bits/function.hpp>
#include <queue>

namespace ens {

/**
 * After the velocity of each particle is updated at the end of each iteration
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
template<typename ArbitraryFunctionType,
         typename MatType,
         typename... CallbackTypes>
typename MatType::elem_type PSOType<VelocityUpdatePolicy, InitPolicy>::Optimize(
    ArbitraryFunctionType& function,
    MatType& iterateIn,
    CallbackTypes&&... callbacks)
{
  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;

  // The update policy internally use a templated class so that
  // we can know MatType only when Optimize() is called.
  typedef typename VelocityUpdatePolicy::template Policy<BaseMatType>
      InstUpdatePolicyType;

  // Make sure that we have the methods that we need.  Long name...
  traits::CheckArbitraryFunctionTypeAPI<ArbitraryFunctionType,
      BaseMatType>();
  RequireDenseFloatingPointType<BaseMatType>();

  // The number of iterations must be greater than the horizon size.
  if (maxIterations < horizonSize)
  {
    std::ostringstream oss;
    oss << "PSO::Optimize(): maxIterations (" << maxIterations << ") must be "
        << "greater than or equal to horizonSize (" << horizonSize << ")!";
    throw std::runtime_error(oss.str());
  }

  BaseMatType& iterate = (BaseMatType&) iterateIn;

  // Controls early termination of the optimization process.
  bool terminate = false;

  if (!instUpdatePolicy.Has<InstUpdatePolicyType>())
  {
    instUpdatePolicy.Clean();
    instUpdatePolicy.Set<InstUpdatePolicyType>(
        new InstUpdatePolicyType(velocityUpdatePolicy));
  }

  // Initialize helper variables.
  arma::Cube<ElemType> particlePositions;
  arma::Cube<ElemType> particleVelocities;
  arma::Col<ElemType> particleFitnesses;
  arma::Col<ElemType> particleBestFitnesses;
  arma::Cube<ElemType> particleBestPositions;

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
  instUpdatePolicy.As<InstUpdatePolicyType>().Initialize(exploitationFactor,
      explorationFactor, numParticles, iterate);

  Callback::BeginOptimization(*this, function, iterate, callbacks...);

  // Calculate initial fitness of population.
  for (size_t i = 0; (i < numParticles) && !terminate; i++)
  {
    // Calculate fitness value.
    particleFitnesses(i) = function.Evaluate(particlePositions.slice(i));
    terminate |= Callback::Evaluate(*this, function,
        particlePositions.slice(i), particleFitnesses(i), callbacks...);
    particleBestFitnesses(i) = particleFitnesses(i);
  }

  // Declare queue to keep track of improvements over a number of iterations.
  std::queue<ElemType> performanceHorizon;
  // Variable to store the position of the best particle.
  size_t bestParticle = 0;
  // Find the best fitness.
  ElemType bestFitness = std::numeric_limits<ElemType>::max();

  // Run PSO for horizonSize number of iterations.
  // This will allow the performanceHorizon to be updated.
  // With some initial values in this, we may proceed with the remaining steps
  // in the PSO method.
  // The performanceHorizon will be updated with the best particle
  // in a FIFO manner.
  for (size_t i = 0; (i < horizonSize) && !terminate; i++)
  {
    // Calculate fitness and evaluate personal best.
    for (size_t j = 0; (j < numParticles) && !terminate; j++)
    {
      particleFitnesses(j) = function.Evaluate(particlePositions.slice(j));
      terminate |= Callback::Evaluate(*this, function,
          particlePositions.slice(j), particleFitnesses(j), callbacks...);
      if (terminate)
        break;

      // Compare and copy fitness and position to particle best.
      if (particleFitnesses(j) < particleBestFitnesses(j))
      {
        particleBestFitnesses(j) = particleFitnesses(j);
        particleBestPositions.slice(j) = particlePositions.slice(j);
      }
    }

    // Evaluate local best and update velocity.
    instUpdatePolicy.As<InstUpdatePolicyType>().Update(
        particlePositions, particleVelocities, particleBestPositions,
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

    terminate |= Callback::StepTaken(*this, function,
        particleBestPositions.slice(bestParticle), callbacks...);

    // Append bestFitness to performanceHorizon.
    performanceHorizon.push(bestFitness);
  }

  // Run the remaining iterations of PSO.
  for (size_t i = 0; (i < maxIterations - horizonSize) && !terminate; i++)
  {
    // Check if there is any improvement over the horizon.
    // If there is no significant improvement, terminate.
    if (performanceHorizon.front() - performanceHorizon.back() < impTolerance)
      break;

    // Calculate fitness and evaluate personal best.
    for (size_t j = 0; (j < numParticles) && !terminate; j++)
    {
      particleFitnesses(j) = function.Evaluate(particlePositions.slice(j));
      terminate |= Callback::Evaluate(*this, function,
          particlePositions.slice(j), particleFitnesses(j), callbacks...);

      // Compare and copy fitness and position to particle best.
      if (particleFitnesses(j) < particleBestFitnesses(j))
      {
        particleBestFitnesses(j) = particleFitnesses(j);
        particleBestPositions.slice(j) = particlePositions.slice(j);
      }
    }

    // Evaluate local best and update velocity.
    instUpdatePolicy.As<InstUpdatePolicyType>().Update(
        particlePositions, particleVelocities, particleBestPositions,
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

    terminate |= Callback::StepTaken(*this, function,
        particleBestPositions.slice(bestParticle), callbacks...);

    // Pop the oldest value from performanceHorizon.
    performanceHorizon.pop();
    // Push most recent bestFitness to performanceHorizon.
    performanceHorizon.push(bestFitness);
  }

  // Copy results back.
  iterate = particleBestPositions.slice(bestParticle);

  Callback::EndOptimization(*this, function, iterate, callbacks...);
  return bestFitness;
}

} // ens

#endif
