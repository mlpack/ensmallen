/**
 * @file bipop_cmaes.hpp
 * @author Marcus Edel
 * @author Benjami Parellada
 * 
 * Definition of the BIPOP Covariance Matrix Adaptation Evolution Strategy 
 * as proposed by N. Hansen in "Benchmarking a BI-population CMA-ES on the
 * BBOB-2009 function testbed".
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CMAES_BIPOP_CMAES_HPP
#define ENSMALLEN_CMAES_BIPOP_CMAES_HPP

#include "cmaes.hpp"

namespace ens {

/**
 * BI-Population CMA-ES is a variant of the stochastic search algorithm
 * CMA-ES - Covariance Matrix Adaptation Evolution Strategy.
 * It implements a dual restart strategy with varying population sizes: one 
 * increasing and one with smaller, varied sizes. This BI-population approach
 * is designed to optimize performance on multi-modal function testbeds by 
 * leveraging different exploration and exploitation dynamics.
 *
 * For more information, please refer to:
 *
 * @code
 * @inproceedings{hansen2009benchmarking,
 *   title={Benchmarking a BI-population CMA-ES on the BBOB-2009 function testbed},
 *   author={Hansen, Nikolaus},
 *   booktitle={Proceedings of the 11th annual conference companion on genetic and evolutionary computation conference: late breaking papers},
 *   pages={2389--2396},
 *   year={2009}}
 * @endcode
 *
 * BI-Population CMA-ES can efficiently handle separable, multimodal, and weak
 * structure functions across various dimensions, as demonstrated in the 
 * comprehensive results of the BBOB-2009 function testbed. The optimizer
 * utilizes an interlaced multistart strategy to balance between broad 
 * exploration and intensive exploitation, adjusting population sizes and 
 * step-sizes dynamically.
 *
 * @tparam CMAESType The type of CMA-ES used for optimization. Options include standard CMAES and variants like ActiveCMAES.
 */
template<typename CMAESType = CMAES<>>
class BIPOPCMAES
{
 public:
  /**
   * Construct the BIPOP CMA-ES optimizer with the given CMAES object,
   * and maximum number of restarts. The defaults 
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
  BIPOPCMAES(const CMAESType& CMAES = CMAESType(),
             const size_t maxRestarts = 9);

  /**
   * Construct the BIPOP CMA-ES optimizer with the given function and parameters. 
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
BIPOPCMAES(const size_t lambda = 0,
          const typename CMAESType::transformationPolicyType&
                transformationPolicy = typename CMAESType::transformationPolicyType(),
          const size_t batchSize = 32,
          const size_t maxIterations = 1000,
          const double tolerance = 1e-5,
          const typename CMAESType::selectionPolicyType&
                selectionPolicy = typename CMAESType::selectionPolicyType(),
          double stepSize = 0,
          const size_t maxRestarts = 9);

  /**
   * Optimize the given function using BIPOP CMA-ES. The given starting point will be
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

  //! Get the maximum number of restarts.
  size_t MaxRestarts() const { return maxRestarts; }
  //! Modify the maximum number of restarts.
  size_t& MaxRestarts() { return maxRestarts; }

  //! Get the CMAES object.
  const CMAESType& CMAES() const { return cmaes; }
  //! Modify the CMAES object.
  CMAESType& CMAES() { return cmaes; }

 private:
  //! Maximum number of restarts.
  size_t maxRestarts;

  //! The CMAES object used for optimization.
  CMAESType cmaes;
};

} // namespace ens

// Include implementation.
#include "bipop_cmaes_impl.hpp"

#endif