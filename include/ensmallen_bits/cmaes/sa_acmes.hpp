/**
 * @file sa_acmes.hpp
 * @author Suvarsha Chennareddy
 *
 * Definition of the Self-Adaptive Surrogate-Assisted Covariance Matrix
 * Adaptation Evolution Strategy as proposed by Ilya Loshchilov, Marc Schoenauer,
 * and Michèle Sebag in "Self-Adaptive Surrogate-Assisted Covariance Matrix
 * Adaptation Evolution Strategy".
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CMAES_SA_ACMES_HPP
#define ENSMALLEN_CMAES_SA_ACMES_HPP

#include "cmaes.hpp"
#include "active_cmaes.hpp"

namespace ens {

/**
 * saACM-ES is an extension of the surrogated-assisted variant of CMA-ES called 
 * ACM-ES. saACM-ES adjusts the lifelength of the current surrogate model (the
 * number of CMA-ES generations before learning a new surrogate) and the 
 * surrogate hyper-parameters.
 *
 * For more information, please refer to:
 *
 * @code
 * @inproceedings{10.1145/2330163.2330210,
 *   author = {Loshchilov, Ilya and Schoenauer, Marc and Sebag, Michele},
 *   title = {Self-Adaptive Surrogate-Assisted Covariance Matrix Adaptation Evolution Strategy},
 *   year = {2012},
 *   isbn = {9781450311779},
 *   publisher = {Association for Computing Machinery},
 *   address = {New York, NY, USA},
 *   url = {https://doi.org/10.1145/2330163.2330210},
 *   doi = {10.1145/2330163.2330210},
 *   booktitle = {Proceedings of the 14th Annual Conference on Genetic and Evolutionary Computation},
 *   pages = {321–328},
 *   numpages = {8},
 *   location = {Philadelphia, Pennsylvania, USA},
 *   series = {GECCO '12}}
 * @endcode
 * 
 * saACM-ES can optimize separable functions.  For more details, see the
 * documentation on function types included with this distribution or on the
 * ensmallen website.
 *
 * @tparam OptCMAESType The CMA-ES type used for the optimization of the 
 *       objective function. Currently, either CMAES or ActiveCMAES can be used.
 * @tparam SurrCMAESType The CMA-ES type used for the optimization of the 
         surrogate model hyper-parameters. Currently, either CMAES or 
         ActiveCMAES can be used.
 */
  template<typename OptCMAESType = CMAES<>, typename SurrCMAESType = CMAES<>>
class saACMES
{
 public:
  /**
   * Construct the saACM-ES optimizer with the given OptCMAES object,
   * surrCMAES object, and other parameters.
   *
   * @param optCMAES The CMAES object which will be used for the optimization
          of the objective function.
   * @param surrCMAES The CMAES object which will be used for the optimization
          of the surrogate model hyper-parameters.
   * @param maxIterations Maximum number of iterations allowed.
   * @param gStart Number of iterations optCMAES is allowed to run
          with true objective function initially.
   * @param errorThreshold Maximum error the surrogate model is allowed
   *      to have.
   * @param maxSurrLifeLength Maximum number of iterations optCMAES is allowed 
          to run with surrogate model.
   * @param maxNumTrainingPoints Maximum number of training points
          the surrogate model can train with.
   * @param numTestingPoints Number of testing points to test the surrogate model.
   * @param errorRelaxationConstant Relaxation constant used to update the 
          surrogate error with additive relaxation.
   */
  saACMES(const OptCMAESType& optCMAES,
          const SurrCMAESType& surrCMAES,
          const size_t maxIterations,
          const size_t gStart,
          const double errorThreshold,
          const size_t maxSurrLifeLength,
          const size_t maxNumTrainingPoints,
          const size_t numTestingPoints,
          const double errorRelaxationConstant);

  /**
   * Optimize the given (expensive) function using saACM CMA-ES. The given 
   * starting point will be modified to store the finishing point of the 
   * algorithm, and the final objective value is returned.
   *
   * @tparam SeparableFunctionType Type of the function to be optimized.
   * @tparam MatType Type of matrix to optimize.
   * @tparam CallbackTypes Types of callback functions.
   * @param function Function to optimize.
   * @param iterate Starting point (will be modified).
   * @param callbacks Callback functions.
   * @return Objective value of the final point.
   */
  template<typename SeparableFunctionType,
      typename MatType,
      typename... CallbackTypes>
      typename MatType::elem_type Optimize(SeparableFunctionType& function,
                                           MatType& iterate,
                                           CallbackTypes&&... callbacks);

  //! Get the optCMAES object.
  OptCMAESType OptCMAES() const { return optCMAES; }
  //! Modify the optCMAES object.
  OptCMAESType& OptCMAES() { return optCMAES; }

  //! Get the surrCMAES object.
  SurrCMAESType SurrCMAES() const { return SurrCMAES; }
  //! Modify the surrCMAES object.
  SurrCMAESType& SurrCMAES() { return SurrCMAES; }

  //! Get the maximum number of iterations.
  size_t MaxIterations() const { return maxIterations; }
  //! Get the maximum number of iterations.
  size_t& MaxIterations() { return maxIterations; }

  //! Get the number of generations that optCMAES is allowed to run
  //! with true objective function initially.
  size_t NumStartGenerations() const { return gStart; }
  //! Modify the number of generations that optCMAES is allowed to run
  //! with true objective function initially.
  size_t& NumStartGenerations() { return gStart; }

  //! Get the surrogate error threshold.
  double ErrorThreshold() const { return errorThreshold; }
  //! Modify the surrogate error threshold.
  double& ErrorThreshold() { return errorThreshold; }

  //! Get the maximum surrogate lifelength.
  size_t MaxSurrogateLifeLength() const { return maxSurrLifeLength; }
  //! Modify the maximum surrogate lifelength..
  size_t& MaxSurrogateLifeLength() { return maxSurrLifeLength; }

  //! Get the maximum number of surrogate training points.
  size_t MaxNumSurrogateTrainingPoints() const
  { return maxNumTrainingPoints; }
  //! Modify the maximum number of surrogate training points.
  size_t& MaxNumSurrogateTrainingPoints()
  { return maxNumTrainingPoints; }

  //! Get the surrogate error relaxation constant.
  double ErrorRelaxationConstant() const { return beta; }
  //! Modify the surrogate error relaxation constant.
  double& ErrorRelaxationConstant() { return beta; }

 private:

   //! CMAES object for the optimization of the objective function.
   OptCMAESType optCMAES;

   //! CMAES object for the optimization of the surrogate hyper-parameters.
   SurrCMAESType surrCMAES;

   //! Maximum number of iterations.
   size_t maxIterations;

   //! The number of generations that optCMAES is allowed to run
   //! with true objective function initially.
   size_t gStart;

   //! The surrogate error threshold
   double errorThreshold;

   //! The maximum number of training points to train the surrogate model.
   size_t maxNumTrainingPoints;

   //! The maximum number of testing points points to test the surrogate model.
   size_t numTestingPoints;

   //! The maximum surrogate lifelength.
   size_t maxSurrLifeLength;

   //! The surrogate error relaxation constant.
   double beta;
};

} // namespace ens

// Include implementation.
#include "sa_acmes_impl.hpp"

#endif
