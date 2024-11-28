/**
 * @file ipop_cmaes_impl.hpp
 * @author Marcus Edel
 * @author Benjami Parellada
 *
 * Definition of the IPOP Covariance Matrix Adaptation Evolution Strategy
 * as proposed by A. Auger and N. Hansen in "A Restart CMA Evolution
 * Strategy With Increasing Population Size" and BIPOP Covariance Matrix
 * Adaptation Evolution Strategy as proposed by N. Hansen in "Benchmarking 
 * a BI-population CMA-ES on the BBOB-2009 function testbed".
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CMAES_POP_CMAES_HPP
#define ENSMALLEN_CMAES_POP_CMAES_HPP

#include "cmaes.hpp"

namespace ens {

/**
 * Population-based CMA-ES (POP-CMA-ES) that can operate as either IPOP-CMA-ES
 * or BIPOP-CMA-ES based on a flag.
 * 
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
 */
template<typename SelectionPolicyType = FullSelection,
         typename TransformationPolicyType = EmptyTransformation<>,
         bool UseBIPOPFlag = true>
class POP_CMAES : public CMAES<SelectionPolicyType, TransformationPolicyType>
{
 public:
  /**
   * Construct the POP-CMA-ES optimizer with the given parameters.
   * Other than the same CMA-ES parameters, it also adds the maximum number of 
   * restarts, the increase in population factor, the maximum number of 
   * evaluations, as well as a flag indicating to use BIPOP or not.
   * The suggested values are not necessarily good for the given problem, so it
   * is suggested that the values used be tailored to the task at hand. The
   * maximum number of iterations refers to the maximum number of points that
   * are processed (i.e., one iteration equals one point; one iteration does not
   * equal one pass over the dataset).
   * 
   * @param lambda The initial population size (0 use the default size).
   * @param transformationPolicy Instantiated transformation policy used to
   *    map the coordinates to the desired domain.
   * @param batchSize Batch size to use for the objective calculation.
   * @param maxIterations Maximum number of iterations allowed.
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param selectionPolicy Instantiated selection policy used to calculate the
   *     objective.
   * @param stepSize Starting sigma/step size (will be modified).
   * @param populationFactor The factor by which population increases
   *     after each restart.
   * @param maxRestarts Maximum number of restarts.
   * @param maxFunctionEvaluations Maximum number of function evaluations.
   */
  POP_CMAES(const size_t lambda = 0,
            const TransformationPolicyType& transformationPolicy = 
                 TransformationPolicyType(),
            const size_t batchSize = 32,
            const size_t maxIterations = 1000,
            const double tolerance = 1e-5,
            const SelectionPolicyType& selectionPolicy = SelectionPolicyType(),
            double stepSize = 0,
            const size_t maxRestarts = 9,
            const double populationFactor = 2,
            const size_t maxFunctionEvaluations = 1e9);

  /**
   * Set POP-CMA-ES specific parameters.
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
  size_t MaxRestarts() const { return maxRestarts; }
  //! Modify the maximum number of restarts.
  size_t& MaxRestarts() { return maxRestarts; }

  //! Get the maximum number of function evaluations.
  size_t MaxFunctionEvaluations() const { return maxFunctionEvaluations; }
  //! Modify the maximum number of function evaluations.
  size_t& MaxFunctionEvaluations() { return maxFunctionEvaluations; }

  //! Get the BIPOP mode flag.
  static constexpr bool UseBIPOP() { return UseBIPOPFlag; }

 private:
  //! Population factor
  double populationFactor;

  //! Maximum number of restarts.
  size_t maxRestarts;

  //! Maximum number of function evaluations.
  size_t maxFunctionEvaluations;

};

// Define IPOP_CMAES and BIPOP_CMAES using the POP_CMAES template
template<typename SelectionPolicyType = FullSelection,
         typename TransformationPolicyType = EmptyTransformation<>>
using IPOP_CMAES = POP_CMAES<SelectionPolicyType, TransformationPolicyType, false>;

template<typename SelectionPolicyType = FullSelection,
         typename TransformationPolicyType = EmptyTransformation<>>
using BIPOP_CMAES = POP_CMAES<SelectionPolicyType, TransformationPolicyType, true>;

} // namespace ens

// Include implementation.
#include "pop_cmaes_impl.hpp"

#endif