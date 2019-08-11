/**
 * @file default_init.hpp
 * @author Chintan Soni
 * @author Suryoday Basak
 *
 * The default initialization policy used by the PSO optimizer.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PSO_INIT_POLICIES_DEFAULT_INIT_HPP
#define ENSMALLEN_PSO_INIT_POLICIES_DEFAULT_INIT_HPP
#include <assert.h>

namespace ens {

/**
 * The default initialization policy used by the PSO optimizer. It initializes
 * particle positions uniformly in [-1, 1], the velocities in [0, 1] personal
 * bests of the particles to the initial positions, and all fitness values to
 * std::numeric_limits<double>::max().
 */
class DefaultInit
{
 public:
  /**
   * Constructor for the DefaultInit policy. The policy initializes particle
   * posiitons in the range [lowerBound, upperBound]. Defaults to [-1, 1].
   */
  DefaultInit()
  {
    /* Nothing to do.*/
  }

  /**
   * The InitializeParticles method of the init policy. Any class that is used
   * in place of this default must implement this method which is used by the
   * optimizer.
   *
   * @param iterate Coordinates of the initial point for training.
   * @param numParticles The number of particles in the swarm.
   * @param lowerBound Lower bound of the position initialization range.
   * @param upperBound Upper bound of the position initialization range.
   * @param particlePositions Current positions of particles.
   * @param particleVelocities Current velocities of particles.
   * @param particleFitnesses Current fitness values of particles.
   * @param particleBestPositions Best positions attained by each particle.
   * @param particleBestFitnesses Best fitness values attained by each particle.
   */
  template<typename MatType,
           typename BoundMatType,
           typename VecType,
           typename CubeType>
  void Initialize(const MatType& iterate,
                  const size_t numParticles,
                  BoundMatType& lowerBound,
                  BoundMatType& upperBound,
                  CubeType& particlePositions,
                  CubeType& particleVelocities,
                  VecType& particleFitnesses,
                  CubeType& particleBestPositions,
                  VecType& particleBestFitnesses)
  {
    // Convenience typedef.
    typedef typename MatType::elem_type ElemType;
    typedef typename CubeType::elem_type CubeElemType;

    // Randomly initialize the particle positions.
    particlePositions.randu(iterate.n_rows, iterate.n_cols, numParticles);

    // Check if lowerBound is equal to upperBound. If equal, reinitialize.
    arma::umat lbEquality = (lowerBound == upperBound);
    if (lbEquality.n_rows == 1 && lbEquality(0, 0) == 1)
    {
      lowerBound.set_size(iterate.n_rows, iterate.n_cols);
      lowerBound.fill(-1.0);

      upperBound.set_size(iterate.n_rows, iterate.n_cols);
      upperBound.fill(1.0);
    }
    // Check if lowerBound and upperBound are vectors of a single dimension.
    else if (lbEquality.n_rows == 1 && lbEquality(0, 0) == 0)
    {
      lowerBound = -lowerBound(0) * arma::ones(iterate.n_rows, iterate.n_cols);
      upperBound = upperBound(0) * arma::ones(iterate.n_rows, iterate.n_cols);
    }

    // Check the dimensions of lowerBound and upperBound.
    assert(lowerBound.n_rows == iterate.n_rows && "The dimensions of "
        "lowerBound are not the same as the dimensions of iterate.");
    assert(upperBound.n_rows == iterate.n_rows && "The dimensions of "
        "upperBound are not the same as the dimensions of iterate.");

    // Distribute particles in [lowerBound, upperBound].
    for (size_t i = 0; i < numParticles; i++)
    {
      particlePositions.slice(i) = particlePositions.slice(i) %
          arma::conv_to<arma::Mat<CubeElemType> >::from(upperBound - lowerBound)
          + arma::conv_to<arma::Mat<CubeElemType> >::from(lowerBound);
    }

    // Randomly initialize particle velocities.
    particleVelocities.randu(iterate.n_rows, iterate.n_cols, numParticles);

    // Initialize current fitness values to infinity.
    particleFitnesses.set_size(numParticles);
    particleFitnesses.fill(std::numeric_limits<ElemType>::max());

    // Copy to personal best values for first iteration.
    particleBestPositions = particlePositions;
    // Initialize personal best fitness values to infinity.
    particleBestFitnesses.set_size(numParticles);
    particleBestFitnesses.fill(std::numeric_limits<ElemType>::max());
  }

};

} // ens

#endif
