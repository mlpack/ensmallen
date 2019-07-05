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

namespace ens {

/**
 * Particle Swarm Optimization (PSO) is an evolutionary approach to optimization
 * that is inspired by flocks or birds or fishes. The fundamental analogy is
 * that every creature (particle in a swarm) is at a measurable position of
 * `goodness' (in the context of PSO, called `fitness') in terms of being able
 * to find food, and this information can be shared amongst the creatures in the
 * flock, so that iteratively, the entire flock can get close to the optimal
 * food source. In a more technical respect, the means by which the fitness
 * information is shared determines the way in which the global optimum is
 * approached.
 *
 * When this information is shared among particles whose fitness is close to
 * each other (in a sense, the `nearest' neighbors in the fitness space), the
 * variant of the approach is called the `local-best' or `lbest' PSO
 * (consequently, it follows a ring-topology in an information-communication
 * sense); and when this information is globally shared, the variant is called
 * the `global-best' or `gbest' PSO (consequently, it follows a star-topology in
 * an information-communication sense).
 *
 * For more information, refer to:
 *
 * @inproceedings{Kennedy,
 *    doi = {10.1109/icnn.1995.488968},
 *    url = {https://doi.org/10.1109/icnn.1995.488968},
 *    publisher = {{IEEE}},
 *    author = {J. Kennedy and R. Eberhart},
 *    title = {Particle swarm optimization},
 *    booktitle = {Proceedings of {ICNN}{\textquotesingle}95 - 
 *                 International Conference on Neural Networks}
 * }
 *
 * PSO can optimize arbitrary functions. For more details, see the documentation
 * on function types included with this distribution or on the ensmallen
 * website.
 * 
 * For PSO to work, the function being optimized must implement an
 * ArbitraryFunctionType template parameter. The respective class must implement
 * the following function:
 *
 *    double Evaluate(const arma::mat& x);
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
   */
  PSOType(const size_t numParticles = 64,
          arma::vec lowerBound = arma::zeros<arma::vec>(1),
	  arma::vec upperBound = arma::zeros<arma::vec>(1),
          const size_t maxIterations = 3000,
          const double exploitationFactor = 2.05,
          const double explorationFactor = 2.05,
          const VelocityUpdatePolicy& velocityUpdatePolicy =
              VelocityUpdatePolicy(),
          const InitPolicy& initPolicy = InitPolicy()) :
          numParticles(numParticles),
          lowerBound(lowerBound),
          upperBound(upperBound),
          maxIterations(maxIterations),
          exploitationFactor(exploitationFactor),
          explorationFactor(explorationFactor),
          velocityUpdatePolicy(velocityUpdatePolicy),
          initPolicy(initPolicy) { /* Nothing to do */ }

  /**
   * Optimize the input function using PSO. The given variable that holds the
   * initial point will be modified to store the value of the optimum, or the
   * point where the PSO method stops, and the final objective value is
   * returned.
   *
   * @param FunctionType Type of the function to be optimized.
   * @param function Function to be optimized.
   * @param iterate Initial point (will be modified).
   * @return Objective value of the final point.
   */
  template<typename FunctionType>
  double Optimize(FunctionType& function, arma::mat& iterate);

  //! Retrieve value of numParticles.
  size_t NumParticles() const { return numParticles; }

  //! Modify value of numParticles.
  size_t& NumParticles() { return numParticles; }

  //! Retrieve value of lowerBound.
  size_t LowerBound() const { return lowerBound; }

  //! Modify value of lowerBound.
  size_t& LowerBound() { return lowerBound; }

  //! Retrieve value of upperBound.
  size_t UpperBound() const { return upperBound; }

  //! Modify value of upperBound.
  size_t& UpperBound() { return upperBound; }

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
  //! Lower bound of the initual swarm.
  arma::vec lowerBound;
  //! Upper bound of the initual swarm.
  arma::vec upperBound;
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
