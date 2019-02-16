/**
 * @file pso.hpp
 * @author Chintan Soni
 * @author Suryoday Basak
 *
 * Particle swarm optimization.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PSO_PSO_HPP
#define ENSMALLEN_PSO_PSO_HPP

#include "update_policies/lbest_update.hpp"
#include "init_policies/default_init.hpp"
#include <iostream>

namespace ens {

/**
 * EXTREMELY DETAILED DESCRIPTION OF THE WORKING OF PSO.
 *
 * For Particle Swarm Optimization to work, a FunctionType template parameter is
 * required.
 * This class must implement the following function:
 *
 *   double Evaluate(const arma::mat& coordinates);
 *   void Gradient(const arma::mat& coordinates,
 *                 arma::mat& gradient);
 */
template<typename VelocityUpdatePolicy = LBestUpdate,
         typename InitPolicy = DefaultInit>
class PSOType
{
 public:
  /**
   * Construct the particle swarm optimizer with the given function and
   * parameters. The defaults here are not necessarily good for the given
   * problem, so it is suggested that the values used be tailored to the task
   * at hand.
   *
   * @param numParticles Number of particles in the swarm.
   * @param maxIterations Number of iterations allowed.
   * @param exploitationFactor Influence of the personal best of the particle.
   * @param explorationFactor Influence of the neighbours of the particle.
   * @param enableGradientDescent Enable the use of gradient descent optimizer.
   * @param psoIterationsRatio Portion of maxIterations which will be run using
   *    the PSO optimizer. The rest of the iterations will use gradient descent.
   * @param stepSize The step size for the gradient descent optimizer.
   */
  PSOType(
    const size_t numParticles = 16,
    const size_t maxIterations = 3000,
    const double exploitationFactor = 2.05,
    const double explorationFactor = 2.05,
    const VelocityUpdatePolicy& velocityUpdatePolicy = VelocityUpdatePolicy(),
    const InitPolicy& initPolicy = InitPolicy()) :
    numParticles(numParticles),
    maxIterations(maxIterations),
    exploitationFactor(exploitationFactor),
    explorationFactor(explorationFactor),
    velocityUpdatePolicy(velocityUpdatePolicy),
    initPolicy(initPolicy) { /* Nothing to do */ }

  /**
   * ADD PSO OPTIMIZATION DESCRIPTION.
   *
   * Optimize the given function using particle swarm optimization. The given
   * starting point will be modified to store the finishing point of the
   * algorithm, and the final objective value is returned.
   *
   * @tparam FunctionType Type of the function to optimize.
   * @param function Function to optimize.
   * @param iterate Initial point (will be modified).
   * @return Objective value of the final point.
   */
  template<typename FunctionType>
  double Optimize(FunctionType& function, arma::mat& iterate);

  //! Retrieve value of numParticles.
  size_t NumParticles() const { return numParticles; }

  //! Modify value of numParticles.
  size_t& NumParticles() { return numParticles; }

  //! Retrieve value of maxIterations.
  size_t MaxIterations() const { return maxIterations; }

  //! Modify value of maxIterations.
  size_t& MaxIterations() { return maxIterations; }

  //! Retrieve value of exploitationFactor.
  double ExploitationFactor() const { return exploitationFactor; }

  //! Modify value of exploitationFactor.
  double& ExploitationFactor() { return exploitationFactor; }

  //! Retrieve value of explorationFactor.
  double ExplorationFactor() const { return explorationFactor; }

  //! Modify value of explorationFactor.
  double& ExplorationFactor() { return explorationFactor; }

 private:
  //! Number of particles in the swarm.
  size_t numParticles;
  //! Maximum number of iterations for which the optimizer will run.
  size_t maxIterations;
  //! Exploitation factor for lbest version.
  double exploitationFactor;
  //! Exploration factor for lbest version.
  double explorationFactor;
  //! Particle positions.
  arma::cube particlePositions;
  //! Particle velocities.
  arma::cube particleVelocities;
  //! Particle fitness values.
  arma::vec particleFitnesses;
  //! Best fitness attained by particle so far.
  arma::vec particleBestFitnesses;
  //! Position corresponding to the best fitness of particle.
  arma::cube particleBestPositions;
  //! Velocity update policy used.
  VelocityUpdatePolicy velocityUpdatePolicy;
  //! Particle initialization policy used.
  InitPolicy initPolicy;
};

using LBestPSO = PSOType<LBestUpdate>;

} // ens

#include "pso_impl.hpp"

#endif
