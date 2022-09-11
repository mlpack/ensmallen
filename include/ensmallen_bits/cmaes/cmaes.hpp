/**
 * @file cmaes.hpp
 * @author Marcus Edel
 * @author Kartik Nighania
 * @author John Hoang
 *
 * Definition of the Covariance Matrix Adaptation Evolution Strategy as proposed
 * by N. Hansen et al. in "Completely Derandomized Self-Adaptation in Evolution
 * Strategies".
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_CMAES_CMAES_HPP
#define ENSMALLEN_CMAES_CMAES_HPP

//! Selection Policy
#include "selection_policies/full_selection.hpp"
#include "selection_policies/random_selection.hpp"

//! Weight initialization policies.
#include "weight_init_policies/default_weight.hpp"
#include "weight_init_policies/negative_weight.hpp"

//! Update Policies
#include "update_policies/vanila_update.hpp"
#include "update_policies/sep_update.hpp"
#include "update_policies/vd_update.hpp"
namespace ens {

/**
 * CMA-ES - Covariance Matrix Adaptation Evolution Strategy is s a stochastic
 * search algorithm. CMA-ES is a second order approach estimating a positive
 * definite matrix within an iterative procedure using the covariance matrix.
 *
 * For more information, please refer to:
 *
 * @code
 * @article{Hansen2001
 *   author    = {Hansen, Nikolaus and Ostermeier, Andreas},
 *   title     = {Completely Derandomized Self-Adaptation in Evolution
 *                Strategies},
 *   journal   = {Evol. Comput.},
 *   volume    = {9},
 *   number    = {2},
 *   year      = {2001},
 *   pages     = {159--195},
 *   publisher = {MIT Press},
 * }
 * @endcode
 *
 * CMA-ES can optimize separable functions.  For more details, see the
 * documentation on function types included with this distribution or on the
 * ensmallen website.
 *
 * @tparam SelectionPolicy The selection strategy used for the evaluation step.
 * @tparam WeightPolicy The weight initialization strategy.
 * @tparam UpdatePolicyType The update parameters strategy.
 */
template<typename SelectionPolicyType = FullSelection,
         typename WeightPolicyType = DefaultWeight,
         typename UpdatePolicyType = VanilaUpdate>

class CMAES
{
 public:
  /**
   * Construct the CMA-ES optimizer with the given function and parameters. The
   * defaults here are not necessarily good for the given problem, so it is
   * suggested that the values used be tailored to the task at hand.  The
   * maximum number of iterations refers to the maximum number of points that
   * are processed (i.e., one iteration equals one point; one iteration does not
   * equal one pass over the dataset).
   *
   * @param lambda The population size (0 use the default size).
   * @param lowerBound Lower bound of decision variables.
   * @param upperBound Upper bound of decision variables.
   * @param batchSize Batch size to use for the objective calculation.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *     limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param selectionPolicy Instantiated selection policy used to calculate the
   *     objective.
   * @param weightPolicy Instantiated weight initialization policy applying on 
   * offsprings in each iteration.
   * @param updatePolicy Instantiated update parameters policy.
   */
  CMAES(const size_t lambda = 0,
        const double lowerBound = -10,
        const double upperBound = 10,
        const size_t batchSize = 32,
        const size_t maxIterations = 1000,
        const double tolerance = 1e-5,
        const SelectionPolicyType& selectionPolicy = SelectionPolicyType(),
        const WeightPolicyType& weightPolicy = WeightPolicyType(),
        const UpdatePolicyType& updatePolicy = UpdatePolicyType());

  /**
   * Optimize the given function using CMA-ES. The given starting point will be
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
  //! Get the population size.
  size_t PopulationSize() const { return lambda; }
  //! Modify the population size.
  size_t& PopulationSize() { return lambda; }

  //! Get the lower bound of decision variables.
  double LowerBound() const { return lowerBound; }
  //! Modify the lower bound of decision variables.
  double& LowerBound() { return lowerBound; }

  //! Get the upper bound of decision variables
  double UpperBound() const { return upperBound; }
  //! Modify the upper bound of decision variables
  double& UpperBound() { return upperBound; }

  //! Get the batch size.
  size_t BatchSize() const { return batchSize; }
  //! Modify the batch size.
  size_t& BatchSize() { return batchSize; }

  //! Get the maximum number of iterations (0 indicates no limit).
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations (0 indicates no limit).
  size_t& MaxIterations() { return maxIterations; }

  //! Get the tolerance for termination.
  double Tolerance() const { return tolerance; }
  //! Modify the tolerance for termination.
  double& Tolerance() { return tolerance; }

  //! Get the selection policy.
  const SelectionPolicyType& SelectionPolicy() const { return selectionPolicy; }
  //! Modify the selection policy.
  SelectionPolicyType& SelectionPolicy() { return selectionPolicy; }

  //! Get the weight policy.
  const WeightPolicyType& WeightPolicy() const { return weightPolicy; }
  //! Modify the weight policy.
  WeightPolicyType& WeightPolicy() { return weightPolicy; }

  //! Get the update policy.
  const UpdatePolicyType& UpdatePolicy() const { return updatePolicy; }
  //! Modify the update policy.
  UpdatePolicyType& UpdatePolicy() { return updatePolicy; }

 private:
  //! Initializing the parameters function.
  template<typename MatType>
  void Initialize(MatType& iterate);

  //! Update the algorithm's parameters.
  template<typename MatType, typename BaseMatType>
  void Update(MatType& iterate,
              BaseMatType& ps, 
              BaseMatType& pc, 
              double& sigma, 
              std::vector<BaseMatType>& z,
              std::vector<BaseMatType>& y,
              BaseMatType& B,
              BaseMatType& D,
              BaseMatType& C,
              BaseMatType& sepCovinv,
              BaseMatType& sepCov,  
              BaseMatType& v,
              arma::uvec& idx);

  //! Population size.
  size_t lambda;

  //! Lower bound of decision variables.
  double lowerBound;

  //! Upper bound of decision variables.
  double upperBound;

  //! The batch size for processing.
  size_t batchSize;

  //! The maximum number of allowed iterations.
  size_t maxIterations;

  //! The tolerance for termination.
  double tolerance;

  //! The selection policy used to calculate the objective.
  SelectionPolicyType selectionPolicy; 

  // The weight initialization policy.
  WeightPolicyType weightPolicy;

  // The update policy.
  UpdatePolicyType updatePolicy;
  
 private:
  size_t mu; // number of candidate solutions.
  // TODO: might need a more general type.
  arma::Row<double> weights; // offsprings weighting scheme.
  double csigma; // cumulation constant for step size.
  double c1; // learning rate of the rank one update using pc.
  double cmu; // covariance matrix learning reate for the rank mu update. 
  double cc; // cumulation constant for pc. 
  double mu_eff; // \sum^\mu _weights.
  double dsigma; // step size damping factor. 
  double alphamu;

  // computed once at init for speeding up operations.
  double chi; // norm of N(0,I).
  double hsigma;

  // active cma.
  double cm; //learning rate for the mean.
  double alphacov; // = 2 (active CMA only).

  // stopping criteria parameters.
  size_t countval;
  size_t eigenval;
  size_t niter;
};

/**
 * Convenient typedef for CMAES approximation.
 */

using ApproxCMAES = CMAES<RandomSelection, DefaultWeight, VanilaUpdate>;

using ActiveCMAES = CMAES<FullSelection, NegativeWeight, VanilaUpdate>;

using ActiveApproxCMAES = CMAES<RandomSelection, NegativeWeight, VanilaUpdate>;

using SepCMAES = CMAES<FullSelection, DefaultWeight, SepUpdate>;

using VDCMAES = CMAES<FullSelection, DefaultWeight, VDUpdate>;
} // namespace ens

#include "cmaes_impl.hpp"

#endif