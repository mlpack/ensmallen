/**
 * @file de.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Differential Evolution is a method used for global optimization of arbitrary
 * functions that optimizes a problem by iteratively trying to improve a
 * candidate solution.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_DE_DE_HPP
#define ENSMALLEN_DE_DE_HPP

namespace ens {

/**
 * Differential evolution is a stochastic evolutionary algorithm used for global
 * optimization. This class implements the best/1/bin strategy of differential
 * evolution to converge a given function to minima.
 *
 * The algorithm works by generating a fixed number of candidates from the
 * given starting point. At each pass through the population, the algorithm
 * mutates each candidate solution to create a trial solution. If the trial
 * solution is better than the candidate, it is replaced in the
 * population.
 *
 * The evolution takes place in two steps:
 * - Mutation
 * - Crossover
 *
 * Mutation is done by generating a new candidate solution from the best
 * candidate of the previous solution and two random other candidates.
 *
 * Crossover is done by mixing the parameters of the candidate solution and the
 * mutant solution. This is done only if a randomly generated number between 0
 * and 1 is greater than the crossover rate.
 *
 * The final value and the parameters are returned by the Optimize() method.
 *
 * For more information, see the following:
 *
 * @code
 * @techreport{storn1995,
 *   title    = {Differential Evolution—a simple and efficient adaptive scheme
 *               for global optimization over continuous spaces},
 *   author   = {Storn, Rainer and Price, Kenneth},
 *   year     = 1995
 * }
 * @endcode
 *
 * DE can optimize arbitrary functions.  For more details, see the
 * documentation on function types included with this distribution or on the
 * ensmallen website.
 */
class DE
{
 public:
  /**
   * Constructor for the DE optimizer
   *
   * The default values provided over here are not necessarily suitable for a
   * given function. Therefore it is highly recommended to adjust the
   * parameters according to the problem.
   *
   * @param populationSize The number of candidates in the population.
   *     This should be at least 3 in size.
   * @param maxGenerations The maximum number of generations allowed for CNE.
   * @param crossoverRate  The probability that a crossover will occur.
   * @param differentialWeight A parameter used in the mutation of candidate
   *     solutions controls amplification factor of the differentiation.
   * @param tolerance The final value of the objective function for termination.
   */
  DE(const size_t populationSize = 100,
     const size_t maxGenerations = 2000,
     const double crossoverRate = 0.6,
     const double differentialWeight = 0.8,
     const double tolerance = 1e-5);

  /**
   * Optimize the given function using DE. The given
   * starting point will be modified to store the finishing point of the
   * algorithm, and the final objective value is returned.
   *
   * @tparam FunctionType Type of the function to be optimized.
   * @tparam MatType Type of matrix to optimize.
   * @tparam CallbackTypes Types of callback functions.
   * @param function Function to optimize.
   * @param iterate Starting point (will be modified).
   * @param callbacks Callback functions.
   * @return Objective value of the final point.
   */
  template<typename FunctionType,
           typename MatType,
           typename... CallbackTypes>
  typename MatType::elem_type Optimize(FunctionType& function,
                                       MatType& iterate,
                                       CallbackTypes&&... callbacks);

  //! Get the population size.
  size_t PopulationSize() const { return populationSize; }
  //! Modify the population size.
  size_t& PopulationSize() { return populationSize; }

  //! Get maximum number of generations.
  size_t MaxGenerations() const { return maxGenerations; }
  //! Modify maximum number of generations.
  size_t& MaxGenerations() { return maxGenerations; }

  //! Get crossover rate.
  double CrossoverRate() const { return crossoverRate; }
  //! Modify crossover rate.
  double& CrossoverRate() { return crossoverRate; }

  //! Get differential weight.
  double DifferentialWeight() const {return differentialWeight; }
  //! Modify differential weight.
  double& DifferentialWeight() { return differentialWeight; }

  //! Get the tolerance.
  double Tolerance() const { return tolerance; }
  //! Modify the tolerance.
  double& Tolerance() { return tolerance; }

 private:
  //! The number of candidates in the population.
  size_t populationSize;

  //! Maximum number of generations before termination criteria is met.
  size_t maxGenerations;

  //! Probability that crossover will occur.
  double crossoverRate;

  //! Amplification factor for differentiation.
  double differentialWeight;

  //! The tolerance for termination.
  double tolerance;
};

} // namespace ens

// Include implementation.
#include "de_impl.hpp"

#endif
