/**
 * @file moead.hpp
 * @author Nanubala Gnana Sai
 *
 * MOEA/D-DE is a multi objective optimization algorithm. MOEA/D-DE
 * uses genetic algorithms along with a set of reference directions
 * to drive the population towards the Optimal Front.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_MOEAD_MOEAD_HPP
#define ENSMALLEN_MOEAD_MOEAD_HPP

//! Decomposition policies.
#include "decomposition_policies/tchebycheff_decomposition.hpp"
#include "decomposition_policies/weighted_decomposition.hpp"
#include "decomposition_policies/pbi_decomposition.hpp"

//! Weight initialization policies.
#include "weight_init_policies/uniform_init.hpp"
#include "weight_init_policies/bbs_init.hpp"
#include "weight_init_policies/dirichlet_init.hpp"

namespace ens {

/**
 * MOEA/D-DE (Multi Objective Evolutionary Algorithm based on Decompositon - 
 * Differential Variant) is a multiobjective optimization algorithm. This class 
 * implements the said optimizer. 
 *
 * The algorithm works by generating a candidate population from a fixed starting point. 
 * Reference directions are generated to guide the optimization process towards the Pareto Front. 
 * Further, a decomposition function is defined to decompose the problem to a scalar optimization 
 * objective. Utilizing genetic operators, offsprings are generated with better decomposition values 
 * to replace the neighboring parent solutions. 
 *
 * For more information, see the following:
 * @code
 * @article{li2008multiobjective,
 *   title={Multiobjective optimization problems with complicated Pareto sets, MOEA/D and NSGA-II},
 *   author={Li, Hui and Zhang, Qingfu},
 *   journal={IEEE transactions on evolutionary computation},
 *   pages={284--302},
 *   year={2008},
 * @endcode
 */
template<typename InitPolicyType = Uniform,
         typename DecompPolicyType = Tchebycheff>
class MOEAD {
 public:
  /**
   * Constructor for the MOEA/D optimizer.
   *
   * The default values provided here are not necessarily suitable for a
   * given function. Therefore, it is highly recommended to adjust the
   * parameters according to the problem.
   *
   * @param populationSize The number of elements in the population.
   * @param maxGenerations The maximum number of generations allowed.
   * @param crossoverProb The probability that a crossover will occur.
   * @param neighborProb The probability of sampling from neighbor.
   * @param neighborSize The number of nearest neighbours of weights
   *    to find.
   * @param distributionIndex The crowding degree of the mutation.
   * @param differentialWeight A parameter used in the mutation of candidate
   *     solutions controls amplification factor of the differentiation.
   * @param maxReplace The limit of solutions allowed to be replaced by a child.
   * @param epsilon Handle numerical stability after weight initialization.
   * @param lowerBound The lower bound on each variable of a member
   *    of the variable space.
   * @param upperBound The upper bound on each variable of a member
   *    of the variable space.
   */
  MOEAD(const size_t populationSize = 300,
        const size_t maxGenerations = 500,
        const double crossoverProb = 1.0,
        const double neighborProb = 0.9,
        const size_t neighborSize = 20,
        const double distributionIndex = 20,
        const double differentialWeight = 0.5,
        const size_t maxReplace = 2,
        const double epsilon = 1E-10,
        const arma::vec& lowerBound = arma::zeros(1, 1),
        const arma::vec& upperBound = arma::ones(1, 1),
        const InitPolicyType initPolicy = InitPolicyType(),
        const DecompPolicyType decompPolicy = DecompPolicyType());

  /**
   * Constructor for the MOEA/D optimizer. This constructor is provides an
   * overload to use lowerBound and upperBound as doubles, in case all the
   * variables in the problem have the same limits.
   *
   * The default values provided here are not necessarily suitable for a
   * given function. Therefore, it is highly recommended to adjust the
   * parameters according to the problem.
   *
   * @param populationSize The number of elements in the population.
   * @param maxGenerations The maximum number of generations allowed.
   * @param crossoverProb The probability that a crossover will occur.
   * @param neighborProb The probability of sampling from neighbor.
   * @param neighborSize The number of nearest neighbours of weights
   *    to find.
   * @param distributionIndex The crowding degree of the mutation.
   * @param differentialWeight A parameter used in the mutation of candidate
   *     solutions controls amplification factor of the differentiation.
   * @param maxReplace The limit of solutions allowed to be replaced by a child.
   * @param epsilon Handle numerical stability after weight initialization.
   * @param lowerBound The lower bound on each variable of a member
   *    of the variable space.
   * @param upperBound The upper bound on each variable of a member
   *    of the variable space.
   */
    MOEAD(const size_t populationSize = 300,
          const size_t maxGenerations = 500,
          const double crossoverProb = 1.0,
          const double neighborProb = 0.9,
          const size_t neighborSize = 20,
          const double distributionIndex = 20,
          const double differentialWeight = 0.5,
          const size_t maxReplace = 2,
          const double epsilon = 1E-10,
          const double lowerBound = 0,
          const double upperBound = 1,
          const InitPolicyType initPolicy = InitPolicyType(),
          const DecompPolicyType decompPolicy = DecompPolicyType());

  /**
   * Optimize a set of objectives. The initial population is generated
   * using the initial point. The output is the best generated front.
   *
   * @tparam MatType The type of matrix used to store coordinates.
   * @tparam ArbitraryFunctionType The type of objective function.
   * @tparam CallbackTypes Types of callback function.
   * @param objectives std::tuple of the objective functions.
   * @param iterate The initial reference point for generating population.
   * @param callbacks The callback functions.
   */
  template<typename MatType,
           typename... ArbitraryFunctionType,
           typename... CallbackTypes>
  typename MatType::elem_type Optimize(std::tuple<ArbitraryFunctionType...>& objectives,
                                       MatType& iterate,
                                       CallbackTypes&&... callbacks);

  //! Retrieve population size.
  size_t PopulationSize() const { return populationSize; }
  //! Modify the population size.
  size_t& PopulationSize() { return populationSize; }

  //! Retrieve number of generations.
  size_t MaxGenerations() const { return maxGenerations; }
  //! Modify the number of generations.
  size_t& MaxGenerations() { return maxGenerations; }

  //! Retrieve crossover rate.
  double CrossoverRate() const { return crossoverProb; }
  //! Modify the crossover rate.
  double& CrossoverRate() { return crossoverProb; }

  //! Retrieve size of the weight neighbor.
  size_t NeighborSize() const { return neighborSize; }
  //! Modify the size of the weight neighbor.
  size_t& NeighborSize() { return neighborSize; }

  //! Retrieve value of the distribution index.
  double DistributionIndex() const { return distributionIndex; }
  //! Modify the value of the distribution index.
  double& DistributionIndex() { return distributionIndex; }

  //! Retrieve value of neighbor probability.
  double NeighborProb() const { return neighborProb; }
  //! Modify the value of neigbourhood probability.
  double& NeighborProb() { return neighborProb; }

  //! Retrieve value of scaling factor.
  double DifferentialWeight() const { return differentialWeight; }
  //! Modify the value of scaling factor.
  double& DifferentialWeight() { return differentialWeight; }

  //! Retrieve value of maxReplace.
  size_t MaxReplace() const { return maxReplace; }
  //! Modify value of maxReplace.
  size_t& MaxReplace() { return maxReplace; }

   //! Retrieve value of epsilon.
  double Epsilon() const { return epsilon; }
  //! Modify value of maxReplace.
  double& Epsilon() { return epsilon; }

  //! Retrieve value of lowerBound.
  const arma::vec& LowerBound() const { return lowerBound; }
  //! Modify value of lowerBound.
  arma::vec& LowerBound() { return lowerBound; }

  //! Retrieve value of upperBound.
  const arma::vec& UpperBound() const { return upperBound; }
  //! Modify value of upperBound.
  arma::vec& UpperBound() { return upperBound; }

  //! Retrieve the Pareto optimal points in variable space. This returns an empty cube
  //! until `Optimize()` has been called.
  const arma::cube& ParetoSet() const { return paretoSet; }

  //! Retrieve the best front (the Pareto frontier). This returns an empty cube until
  //! `Optimize()` has been called.
  const arma::cube& ParetoFront() const { return paretoFront; }

  //! Get the weight initialization policy.
  const InitPolicyType& InitPolicy() const { return initPolicy; }
  //! Modify the weight initialization policy.
  InitPolicyType& InitPolicy() { return initPolicy; }

  //! Get the weight decomposition policy.
  const DecompPolicyType& DecompPolicy() const { return decompPolicy; }
  //! Modify the weight decomposition policy.
  DecompPolicyType& DecompPolicy() { return decompPolicy; }

 private:
  /**
   * @brief Randomly selects two members from the population.
   *
   * @param subProblemIdx Index of the current subproblem.
   * @param neighborSize A matrix containing indices of the neighbors.
   * @return std::tuple<size_t, size_t> The chosen pair of indices.
   */
  std::tuple<size_t, size_t> Mating(size_t subProblemIdx,
                                    const arma::umat& neighborSize,
                                    bool sampleNeighbor);

  /**
   * Mutate the child formed by the crossover of two random members of the
   * population. Uses polynomial mutation.
   *
   * @tparam MatType The type of matrix used to store coordinates.
   * @param child The candidate to be mutated.
   * @param mutationRate The probability of mutation.
   * @param lowerBound The lower bound on each variable in the matrix.
   * @param upperBound The upper bound on each variable in the matrix.
   * @return The mutated child.
   */
  template<typename MatType>
  void Mutate(MatType& child,
              double mutationRate,
              const MatType& lowerBound,
              const MatType& upperBound);

  /**
   * Evaluate objectives for the elite population.
   *
   * @tparam ArbitraryFunctionType std::tuple of multiple function types.
   * @tparam MatType Type of matrix to optimize.
   * @param population The elite population.
   * @param objectives The set of objectives.
   * @param calculatedObjectives Vector to store calculated objectives.
   */
  template<std::size_t I = 0,
           typename MatType,
           typename ...ArbitraryFunctionType>
  typename std::enable_if<I == sizeof...(ArbitraryFunctionType), void>::type
  EvaluateObjectives(
                     std::vector<MatType>&,
                     std::tuple<ArbitraryFunctionType...>&,
                     std::vector<arma::Col<typename MatType::elem_type> >&);

  template<std::size_t I = 0,
           typename MatType,
           typename ...ArbitraryFunctionType>
  typename std::enable_if<I < sizeof...(ArbitraryFunctionType), void>::type
  EvaluateObjectives(
                     std::vector<MatType>& population,
                     std::tuple<ArbitraryFunctionType...>& objectives,
                     std::vector<arma::Col<typename MatType::elem_type> >&
                     calculatedObjectives);

  //! Size of the population.
  size_t populationSize;

  //! Maximum number of generations before termination criteria is met.
  size_t maxGenerations;

  //! Probability of crossover between two members.
  double crossoverProb;

  //! The probability that two elements will be chosen from the neighbor.
  double neighborProb;

  //! Number of nearest neighbours of weights to consider.
  size_t neighborSize;

  //! The crowding degree of the mutation. Higher value produces a mutant
  //! resembling its parent.
  double distributionIndex;

  //! Amplification factor for differentiation.
  double differentialWeight;

  //! Maximum number of childs which can replace the parent. Higher value
  //! leads to a loss of diversity.
  size_t maxReplace;

  //! A small numeric value to be added to the weights after initialization.
  //! Prevents zero value inside inited weights.
  double epsilon;

  //! Lower bound on each variable in the variable space.
  arma::vec lowerBound;

  //! Upper bound on each variable in the variable space.
  arma::vec upperBound;

  //! The set of all the Pareto optimal points.
  //! Stored after Optimize() is called.
  arma::cube paretoSet;

  //! The set of all the Pareto optimal objective vectors.
  //! Stored after Optimize() is called.
  arma::cube paretoFront;

  //! Policy to initialize the reference directions (weights) matrix.
  InitPolicyType initPolicy;

  //! Policy to decompose the weights.
  DecompPolicyType decompPolicy;
};

using DefaultMOEAD = MOEAD<Uniform, Tchebycheff>;
using BBSMOEAD = MOEAD<BayesianBootstrap, Tchebycheff>;
using DirichletMOEAD = MOEAD<Dirichlet, Tchebycheff>;
} // namespace ens

// Include implementation.
#include "moead_impl.hpp"

#endif
