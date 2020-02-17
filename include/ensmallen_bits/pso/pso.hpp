/**
 * @file pso.hpp
 * @author Chintan Soni
 * @author Suryoday Basak
 *
 * Particle swarm optimization.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
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
 * @code
 * @inproceedings{Kennedy1995,
 *   author    = {Kennedy, James and Eberhart, Russell C.},
 *   booktitle = {Proceedings of the IEEE International Conference on
 *                Neural Networks},
 *   pages     = {1942--1948},
 *   title     = {Particle swarm optimization},
 *   year      = 1995
 * }
 * @endcode
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
 *
 * @tparam VelocityUpdatePolicy Velocity update policy. By default LBest update
 *     policy (see ens::LBestUpdate) is used.
 * @tparam InitPolicy Particle initialization policy. By default DefaultInit
 *     policy (see ens::DefaultInit) is used.
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
   * @param lowerBound Lower bound of the coordinates of the initial population.
   * @param upperBound Upper bound of the coordinates of the initial population.
   * @param maxIterations Number of iterations allowed.
   * @param horizonSize Size of the lookback-horizon for computing improvement.
   * @param impTolerance Improvement threshold for termination.
   * @param exploitationFactor Influence of the personal best of the particle.
   * @param explorationFactor Influence of the neighbours of the particle.
   * @param velocityUpdatePolicy Velocity update policy.
   * @param initPolicy Particle initialization policy.
   */
  PSOType(const size_t numParticles = 64,
          const arma::mat& lowerBound = arma::ones(1, 1),
          const arma::mat& upperBound = arma::ones(1, 1),
          const size_t maxIterations = 3000,
          const size_t horizonSize = 350,
          const double impTolerance = 1e-10,
          const double exploitationFactor = 2.05,
          const double explorationFactor = 2.05,
          const VelocityUpdatePolicy& velocityUpdatePolicy =
              VelocityUpdatePolicy(),
          const InitPolicy& initPolicy = InitPolicy()) :
          numParticles(numParticles),
          lowerBound(lowerBound),
          upperBound(upperBound),
          maxIterations(maxIterations),
          horizonSize(horizonSize),
          impTolerance(impTolerance),
          exploitationFactor(exploitationFactor),
          explorationFactor(explorationFactor),
          velocityUpdatePolicy(velocityUpdatePolicy),
          initPolicy(initPolicy)
  { /* Nothing to do. */ }

  /**
   * Clean memory associated with the PSO object.
   */
  ~PSOType()
  {
    instUpdatePolicy.Clean();
  }

  /**
   * Construct the particle swarm optimizer with the given function and
   * parameters. The defaults here are not necessarily good for the given
   * problem, so it is suggested that the values used be tailored to the task
   * at hand.
   *
   * @param numParticles Number of particles in the swarm.
   * @param lowerBound Lower bound of the coordinates of the initial population.
   * @param upperBound Upper bound of the coordinates of the initial population.
   * @param maxIterations Number of iterations allowed.
   * @param horizonSize Size of the lookback-horizon for computing improvement.
   * @param impTolerance Improvement threshold for termination.
   * @param exploitationFactor Influence of the personal best of the particle.
   * @param explorationFactor Influence of the neighbours of the particle.
   */
  PSOType(const size_t numParticles,
          const double lowerBound,
          const double upperBound,
          const size_t maxIterations = 3000,
          const size_t horizonSize = 350,
          const double impTolerance = 1e-10,
          const double exploitationFactor = 2.05,
          const double explorationFactor = 2.05,
          const VelocityUpdatePolicy& velocityUpdatePolicy =
              VelocityUpdatePolicy(),
          const InitPolicy& initPolicy = InitPolicy()) :
          numParticles(numParticles),
          lowerBound(lowerBound * arma::ones(1, 1)),
          upperBound(upperBound * arma::ones(1, 1)),
          maxIterations(maxIterations),
          horizonSize(horizonSize),
          impTolerance(impTolerance),
          exploitationFactor(exploitationFactor),
          explorationFactor(explorationFactor),
          velocityUpdatePolicy(velocityUpdatePolicy),
          initPolicy(initPolicy)
  { /* Nothing to do. */ }

  /**
   * Optimize the input function using PSO. The given variable that holds the
   * initial point will be modified to store the value of the optimum, or the
   * point where the PSO method stops, and the final objective value is
   * returned.
   *
   * @tparam ArbitraryFunctionType Type of the function to be optimized.
   * @tparam MatType Type of matrix to optimize.
   * @tparam CallbackTypes Types of callback functions.
   * @param function Function to be optimized.
   * @param iterate Initial point (will be modified).
   * @param callbacks Callback functions.
   * @return Objective value of the final point.
   */
  template<typename ArbitraryFunctionType,
           typename MatType,
           typename... CallbackTypes>
  typename MatType::elem_type Optimize(ArbitraryFunctionType& function,
                                       MatType& iterate,
                                       CallbackTypes&&... callbacks);

  //! Retrieve value of numParticles.
  size_t NumParticles() const { return numParticles; }
  //! Modify value of numParticles.
  size_t& NumParticles() { return numParticles; }

  //! Retrieve value of lowerBound.
  const arma::mat& LowerBound() const { return lowerBound; }
  //! Modify value of lowerBound.
  arma::mat& LowerBound() { return lowerBound; }

  //! Retrieve value of upperBound.
  const arma::mat& UpperBound() const { return upperBound; }
  //! Modify value of upperBound.
  arma::mat& UpperBound() { return upperBound; }

  //! Retrieve value of maxIterations.
  size_t MaxIterations() const { return maxIterations; }
  //! Modify value of maxIterations.
  size_t& MaxIterations() { return maxIterations; }

  //! Retrieve value of horizonSize.
  size_t HorizonSize() const { return horizonSize; }
  //! Modify value of horizonSize.
  size_t& HorizonSize() { return horizonSize; }

  //! Retrieve value of impTolerance.
  double ImpTolerance() const { return impTolerance; }
  //! Modify value of impTolerance.
  double& ImpTolerance() { return impTolerance; }

  //! Retrieve value of exploitationFactor.
  double ExploitationFactor() const { return exploitationFactor; }
  //! Modify value of exploitationFactor.
  double& ExploitationFactor() { return exploitationFactor; }

  //! Retrieve value of explorationFactor.
  double ExplorationFactor() const { return explorationFactor; }
  //! Modify value of explorationFactor.
  double& ExplorationFactor() { return explorationFactor; }

  //! Get the update policy.
  const VelocityUpdatePolicy& UpdatePolicy() const
  {
    return velocityUpdatePolicy;
  }
  //! Modify the update policy.
  VelocityUpdatePolicy& UpdatePolicy() { return velocityUpdatePolicy; }

  //! Get the instantiated update policy type.  Be sure to check its type with
  //! Has() before using!
  const Any& InstUpdatePolicy() const { return instUpdatePolicy; }
  //! Modify the instantiated update policy type.  Be sure to check its type
  //! with Has() before using!
  Any& InstUpdatePolicy() { return instUpdatePolicy; }

 private:
  //! Number of particles in the swarm.
  size_t numParticles;

  //! Lower bound of the initial swarm.
  arma::mat lowerBound;

  //! Upper bound of the initial swarm.
  arma::mat upperBound;

  //! Maximum number of iterations for which the optimizer will run.
  size_t maxIterations;

  //! The number of iterations looked back at for improvement analysis.
  size_t horizonSize;

  //! The tolerance for improvement over the horizon.
  double impTolerance;

  //! Exploitation factor for lbest version.
  double exploitationFactor;

  //! Exploration factor for lbest version.
  double explorationFactor;

  //! Velocity update policy used.
  VelocityUpdatePolicy velocityUpdatePolicy;
  //! Particle initialization policy used.
  InitPolicy initPolicy;

  //! The initialized update policy.
  Any instUpdatePolicy;
};

using LBestPSO = PSOType<LBestUpdate>;
} // ens

#include "pso_impl.hpp"

#endif
