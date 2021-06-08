/**
 * @file moead.hpp
 * @author Utkarsh Rai
 * @author Nanubala Gnana Sai
 *
 * MOEA/D, Multi Objective Evolutionary Algorithm based on Decompositon is a
 * multi objective optimization algorithm. It employs evolutionary algorithms,
 * to find better solutions by iterating on the previous solutions and
 * decomposition approaches, to convert the multi objective problem to a single
 * objective one, to find the best Pareto Front for the given problem.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_MOEAD_MOEAD_HPP
#define ENSMALLEN_MOEAD_MOEAD_HPP

namespace ens {

/**
 * This class implements the MOEA/D algorithm with Differential Evolution
 * crossover. Step numbers used in different parts of the implementation
 * correspond to the step number used in the original algorithm by the author.
 *
 * For more information, see the following:
 * @code
 * @article{article,
 * author = {Zhang, Qingfu and Li, Hui},
 * year = {2008},
 * pages = {712 - 731},
 * title = {MOEA/D: A Multiobjective Evolutionary Algorithm Based on
 *    Decomposition},
 * journal = {Evolutionary Computation, IEEE Transactions on},
 *
 * @article{4633340,
 * author={H. {Li} and Q. {Zhang}},
 * year={2009},
 * pages={284-302},}
 * title={Multiobjective Optimization Problems With Complicated Pareto Sets, MOEA/D and NSGA-II},
 * journal={IEEE Transactions on Evolutionary Computation},
 * @endcode
 *
 * MOEA/D can optimize arbitrary multi-objective functions. For more details,
 * see the documentation on function types included with this distribution or
 * on the ensmallen website.
 */
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
   * @param neighborSize The number of nearest neighbours of weights
   *    to find.
   * @param distributionIndex The crowding degree of the mutation.
   * @param neighborProb The probability of sampling from neighbor.
   * @param differentialWeight A parameter used in the mutation of candidate
   *     solutions controls amplification factor of the differentiation.
   * @param maxReplace The limit of solutions allowed to be replaced by a child.
   * @param lowerBound The lower bound on each variable of a member
   *    of the variable space.
   * @param upperBound The upper bound on each variable of a member
   *    of the variable space.
   */
  MOEAD(const size_t populationSize = 150,
        const size_t maxGenerations = 1000,
        const double crossoverProb = 1.0,
        const size_t neighborSize = 20,
        const double distributionIndex = 20,
        const double neighborProb = 0.9,
        const double differentialWeight = 0.5,
        const size_t maxReplace = 2,
        const arma::vec& lowerBound = arma::zeros(1, 1),
        const arma::vec& upperBound = arma::ones(1, 1));

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
   * @param neighborSize The number of nearest neighbours of weights
   *    to find.
   * @param distributionIndex The crowding degree of the mutation.
   * @param neighborProb The probability of sampling from neighbor.
   * @param differentialWeight A parameter used in the mutation of candidate
   *     solutions controls amplification factor of the differentiation.
   * @param maxReplace The limit of solutions allowed to be replaced by a child.
   * @param lowerBound The lower bound on each variable of a member
   *    of the variable space.
   * @param upperBound The upper bound on each variable of a member
   *    of the variable space.
   */
    MOEAD(const size_t populationSize = 150,
          const size_t maxGenerations = 1000,
          const double crossoverProb = 1.0,
          const size_t neighborSize = 20,
          const double distributionIndex = 20,
          const double neighborProb = 0.9,
          const double differentialWeight = 0.5,
          const size_t maxReplace = 2,
          const double lowerBound = 0,
          const double upperBound = 1);

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


 private:
  /**
   * @brief Randomly selects two members from the population.
   *
   * @param subProblemIdx Index of the current subproblem.
   * @param neighborSize A matrix containing indices of the neighbors.
   * @return std::tuple<size_t, size_t> The chosen pair of indices.
   */
  std::tuple<size_t, size_t> MatingSelection(const size_t subProblemIdx,
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
   */
  template<typename MatType>
  void Mutate(MatType& child,
              const double& mutationRate,
              const arma::vec& lowerBound,
              const arma::vec& upperBound);

  /**
   * Decompose the multi objective problem to a single objective problem.
   *
   * @param weights Decomposition weights.
   * @param idealPoint The reference point z for a decomposition problem.
   * @param candidateFitness The fitness value of the candidate.
   * @return The real value obtained from the decomposed function.
   */
  double DecomposeObjectives(const arma::vec& weights,
                             const arma::vec& idealPoint,
                             const arma::vec& candidateFitness);

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
  EvaluateObjectives(std::vector<MatType>&,
                     std::tuple<ArbitraryFunctionType...>&,
                     std::vector<arma::Col<typename MatType::elem_type> >&);

  template<std::size_t I = 0,
           typename MatType,
           typename ...ArbitraryFunctionType>
  typename std::enable_if<I < sizeof...(ArbitraryFunctionType), void>::type
  EvaluateObjectives(std::vector<MatType>& population,
                     std::tuple<ArbitraryFunctionType...>& objectives,
                     std::vector<arma::Col<typename MatType::elem_type> >&
                     calculatedObjectives);

  //! Size of the population.
  size_t populationSize;

  //! Maximum number of generations before termination criteria is met.
  size_t maxGenerations;

  //! Probability of crossover between two members.
  double crossoverProb;

  //! Number of nearest neighbours of weights to consider.
  size_t neighborSize;

  //! The crowding degree of the mutation. Higher value produces a mutant
  //! resembling its parent.
  double distributionIndex;

  //! The probability that two elements will be chosen from the neighbor.
  double neighborProb;

  //! Amplification factor for differentiation.
  double differentialWeight;

  //! Maximum number of childs which can replace the parent. Higher value
  //! leads to a loss of diversity.
  size_t maxReplace;

  //! Lower bound on each variable in the variable space.
  arma::vec lowerBound;

  //! Upper bound on each variable in the variable space.
  arma::vec upperBound;

  //! The number of objectives in multi objective optimisation problem.
  size_t numObjectives;

  //! The set of all the Pareto optimal points.
  //! Stored after Optimize() is called.
  arma::cube paretoSet;

  //! The set of all the Pareto optimal objective vectors.
  //! Stored after Optimize() is called.
  arma::cube paretoFront;
};

} // namespace ens

// Include implementation.
#include "moead_impl.hpp"

#endif
