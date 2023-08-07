/**
 * @file ipop_cmaes.hpp
 * @author Marcus Edel
 * @author Suvarsha Chennareddy
 *
 * Definition of the IPOP Covariance Matrix Adaptation Evolution Strategy 
 * as proposed by A. Auger and N. Hansen in "A Restart CMA Evolution 
 * Strategy With Increasing Population Size".
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CMAES_IPOP_CMAES_HPP
#define ENSMALLEN_CMAES_IPOP_CMAES_HPP

#include "cmaes.hpp"
#include "active_cmaes.hpp"

namespace ens {

/**
 * IPOP CMA-ES is a variant of the stochastic search algorithm
 * CMA-ES - Covariance Matrix Adaptation Evolution Strategy.
 * IPOP CMA-ES, also known as CMAES with increasing population size, 
 * incorporates a restart strategy that involves gradually increasing
 * the population size. This approach is specifically designed to 
 * enhance the performance of CMA-ES on multi-modal functions.
 *
 * For more information, please refer to:
 *
 * @code
 * @INPROCEEDINGS{1554902,
 *   author={Auger, A. and Hansen, N.},
 *   booktitle={2005 IEEE Congress on Evolutionary Computation},
 *   title={A restart CMA evolution strategy with increasing population size},
 *   year={2005},
 *   volume={2},
 *   number={},
 *   pages={1769-1776 Vol. 2},
 *   doi={10.1109/CEC.2005.1554902}}
 * @endcode
 * 
 * IPOP CMA-ES can optimize separable functions.  For more details, see the
 * documentation on function types included with this distribution or on the
 * ensmallen website.
 *
 * @tparam CMAESType The CMA-ES type used for the optimization. Currently, 
 *       either CMAES or ActiveCMAES can be used.
 */
template<typename CMAESType = CMAES<>>
class IPOPCMAES
{
 public:
  /**
   * Construct the IPOP CMA-ES optimizer with the given CMAES object,
   * population factor, and maximum number of restarts. The defaults 
   * here are not necessarily good for the given problem, so it is
   * suggested that the values used be tailored to the task at hand.  The
   * maximum number of iterations refers to the maximum number of points that
   * are processed (i.e., one iteration equals one point; one iteration does not
   * equal one pass over the dataset).
   *
   * @param CMAES The CMAES object.
   * @param populationFactor The factor by which population increases
   *      after each restart.
   * @param maxRestarts Maximum number of restarts.
   */
  IPOPCMAES(const CMAESType& CMAES = CMAESType(),
            const double populationFactor = 1.5,
            const size_t maxRestarts = 5);

  /**
   * Construct the IPOP CMA-ES optimizer with the given function and parameters. 
   * The defaults here are not necessarily good for the given problem, so it 
   * is suggested that the values used be tailored to the task at hand.  The 
   * maximum number of iterations refers to the maximum number of points that 
   * are processed (i.e., one iteration equals one point; one iteration does 
   * not equal one pass over the dataset). 
   * 
   * @param lambda The initial population size (0 use the default size).
   * @param transformationPolicy Instantiated transformation policy used to
   *    map the coordinates to the desired domain.
   * @param batchSize Batch size to use for the objective calculation.
   * @param maxIterations Maximum number of iterations allowed(0 means no
         limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param selectionPolicy Instantiated selection policy used to calculate the
   *     objective.
   * @param stepSize Starting sigma/step size (will be modified).
   * @param populationFactor The factor by which population increases
   *     after each restart.
   * @param maxRestarts Maximum number of restarts.
   */
IPOPCMAES(const size_t lambda = 0,
          const typename CMAESType::TransformationPolicyType&
                transformationPolicy = typename CMAESType::TransformationPolicyType(),
          const size_t batchSize = 32,
          const size_t maxIterations = 1000,
          const double tolerance = 1e-5,
          const typename CMAESType::SelectionPolicyType&
                selectionPolicy = typename CMAESType::SelectionPolicyType(),
          double stepSize = 0,
          const double populationFactor = 1.5,
          const size_t maxRestarts = 5);

  /**
   * Optimize the given function using IPOP CMA-ES. The given starting point will be
   * modified to store the finishing point of the algorithm, and the final
   * objective value is returned.
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

  //! Get the population factor.
  double PopulationFactor() const { return populationFactor; }
  //! Modify the population factor.
  double& PopulationFactor() { return populationFactor; }

  //! Get the maximum number of restarts.
  double MaxRestarts() const { return maxRestarts; }
  //! Modify the maximum number of restarts.
  double& MaxRestarts() { return maxRestarts; }

  //! Get the CMAES object.
  const CMAESType& CMAES() const { return cmaes; }
  //! Modify the CMAES object.
  CMAESType& CMAES() { return cmaes; }

 private:

  //! Population factor.
  double populationFactor;
  
  //! Maximum number of restarts.
  size_t maxRestarts;

  //! The CMAES object used for optimization.
  CMAESType cmaes;
};

} // namespace ens

// Include implementation.
#include "ipop_cmaes_impl.hpp"

#endif
