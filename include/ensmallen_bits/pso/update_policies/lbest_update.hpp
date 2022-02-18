/**
 * @file lbest_update.hpp
 * @author Chintan Soni
 * @author Suryoday Basak
 *
 * Implementation of the lbest update policy for particle swarm optimization.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PSO_UPDATE_POLICIES_LBEST_UPDATE_HPP
#define ENSMALLEN_PSO_UPDATE_POLICIES_LBEST_UPDATE_HPP
#include <assert.h>

namespace ens {

/**
 * The local best version (lbest) of PSO in which particles communicate with
 * only two neighbours each, thus forming a ring topology amongst them. This
 * approach allows PSO to converge at the global minimum, but takes
 * significantly more iterations to do so.
 *
 * The lbest update scheme is described as follows:
 *
 * \f{eqation}{
 * v_{i+1} = \phi (v_i + c_1 * r_1 * (p_{best} - p_{current}) +
 *           c_1 * r_1 * (l_{best} - p_{current}))
 * \f}
 *
 * where \f$ v_i \f$ is the velocity of a particle in iteration \f$ i \f$,
 *       \f$ p_{best} \f$ is the best position of an individual particle,
 *       \f$ p_{current} \f$ is the current position of the particle,
 *       \f$ l_{best} \f$ is the local best position,
 *       \f$ r_1 \f$ and \f$ r_2 \f$  are standard uniform random variables,
 *       \f$ c_1 \f$ is the exploitation factor,
 *       \f$ c_2 \f$ is the exploration factor, and
 *       \f$ \phi \f$ is the constriction factor.
 *
 * For more information, refer the following:
 *
 * @code
 * @article{Poli2007,
 *   author    = {Riccardo Poli and James Kennedy and Tim Blackwell},
 *   title     = {Particle swarm optimization},
 *   year      = {2007},
 *   month     = aug,
 *   publisher = {Springer},
 *   volume    = {1},
 *   number    = {1},
 *   pages     = {33--57},
 *   journal   = {Swarm Intelligence}
 * }
 * @endcod
 */
class LBestUpdate
{
 public:
  /**
   * The UpdatePolicyType policy classes must contain an internal 'Policy'
   * template class with two template arguments: MatType.  This is
   * instantiated at the start of the optimization, and holds parameters
   * specific to an individual optimization.
   */
  template<typename MatType>
  class Policy
  {
    public:
    /**
     * This is called by the optimizer method before the start of the iteration
     * update process.
     *
     * @param parent Instantiated parent class.
     */
     Policy(const LBestUpdate& /* parent */) : n(0)
     { /* Do nothing. */ }

     /**
      * The Initialize method is called by PSO Optimizer method before the
      * start of the iteration process. It calculates the value of the
      * constriction coefficent, initializes the local best indices of each
      * particle to itself, and sets the shape of the r1 and r2 vectors.
      *
      * @param exploitationFactor Influence of personal best achieved.
      * @param explorationFactor Influence of neighbouring particles.
      * @param numParticles The number of particles in the swarm.
      * @param iterate The user input, used for shaping intermediate vectors.
      */
     void Initialize(const double exploitationFactor,
                     const double explorationFactor,
                     const size_t numParticles,
                     MatType& iterate)
     {
       // Copy values to aliases.
       n = numParticles;
       c1 = exploitationFactor;
       c2 = explorationFactor;

       // Calculate the constriction factor
       static double phi = c1 + c2;
       assert(phi > 4.0 && "The sum of the exploitation and exploration "
           "factors must be greater than 4.");

       chi = 2.0 / std::abs(2.0 - phi - std::sqrt((phi - 4.0) * phi));

       // Initialize local best indices to self indices of particles.
       localBestIndices = arma::linspace<
           arma::Col<typename MatType::elem_type> >(0, n-1, n);

       // Set sizes r1 and r2.
       r1.set_size(iterate.n_rows, iterate.n_cols);
       r2.set_size(iterate.n_rows, iterate.n_cols);
     }

     /**
      * Update step for LBestPSO. Compares personal best of each particle with
      * that of its neighbours, and sets the best of the 3 as the lobal best.
      * This particle is then used for calculating the velocity for the update
      * step.
      *
      * @param particlePositions The current coordinates of particles.
      * @param particleVelocities The current velocities (will be modified).
      * @param particleFitnesses The current fitness values or particles.
      * @param particleBestPositions The personal best coordinates of particles.
      * @param particleBestFitnesses The personal best fitness values of
      *     particles.
      */
     void Update(arma::Cube<typename MatType::elem_type>& particlePositions,
                 arma::Cube<typename MatType::elem_type>& particleVelocities,
                 arma::Cube<typename MatType::elem_type>& particleBestPositions,
                 arma::Col<typename MatType::elem_type>& particleBestFitnesses)
     {
       // Velocity update logic.
       for (size_t i = 0; i < n; i++)
       {
         localBestIndices(i) =
             particleBestFitnesses(left(i)) < particleBestFitnesses(i) ?
             left(i) : i;
         localBestIndices(i) =
             particleBestFitnesses(right(i)) < particleBestFitnesses(i) ?
             right(i) : i;
       }

       for (size_t i = 0; i < n; i++)
       {
         // Generate random numbers for current particle.
         r1.randu();
         r2.randu();
         particleVelocities.slice(i) = chi * (particleVelocities.slice(i) +
             c1 * r1 % (particleBestPositions.slice(i) -
             particlePositions.slice(i)) + c2 * r2 %
             (particleBestPositions.slice(localBestIndices(i)) -
             particlePositions.slice(i)));
       }
     }

    private:
     //! Number of particles.
     size_t n;

     //! Exploitation factor.
     typename MatType::elem_type c1;

     //! Exploration factor.
     typename MatType::elem_type c2;

     //! Constriction factor chi.
     typename MatType::elem_type chi;

     //! Vectors of random numbers.
     MatType r1, r2;

     //! Indices of each particle's best neighbour.
     arma::Col<typename MatType::elem_type> localBestIndices;

     // Helper functions for calculating neighbours.
    inline size_t left(size_t index) { return (index + n - 1) % n; }
    inline size_t right(size_t index) { return (index + 1) % n; }
  };
};

} // ens

#endif
