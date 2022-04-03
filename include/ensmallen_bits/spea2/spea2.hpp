
#ifndef ENSMALLEN_SPEA2_SPEA2_HPP
#define ENSMALLEN_SPEA2_SPEA2_HPP

namespace ens {

class SPEA2
{
 public:
  
  SPEA2(const size_t populationSize = 100,
        const size_t maxGenerations = 2000,
        const double crossoverProb = 0.6,
        const double mutationProb = 0.3,
        const double mutationStrength = 1e-3,
        const double epsilon = 1e-6,
        const arma::vec& lowerBound = arma::zeros(1, 1),
        const arma::vec& upperBound = arma::ones(1, 1));

  
  SPEA2(const size_t populationSize = 100,
        const size_t maxGenerations = 2000,
        const double crossoverProb = 0.6,
        const double mutationProb = 0.3,
        const double mutationStrength = 1e-3,
        const double epsilon = 1e-6,
        const double lowerBound = 0,
        const double upperBound = 1);

  
  template<typename MatType,
           typename... ArbitraryFunctionType,
           typename... CallbackTypes>
 typename MatType::elem_type Optimize(
     std::tuple<ArbitraryFunctionType...>& objectives,
     MatType& iterate,
     CallbackTypes&&... callbacks);

  //! Get the population size.
  size_t PopulationSize() const { return populationSize; }
  //! Modify the population size.
  size_t& PopulationSize() { return populationSize; }

  //! Get the maximum number of generations.
  size_t MaxGenerations() const { return maxGenerations; }
  //! Modify the maximum number of generations.
  size_t& MaxGenerations() { return maxGenerations; }

  //! Get the crossover rate.
  double CrossoverRate() const { return crossoverProb; }
  //! Modify the crossover rate.
  double& CrossoverRate() { return crossoverProb; }

  //! Get the mutation probability.
  double MutationProbability() const { return mutationProb; }
  //! Modify the mutation probability.
  double& MutationProbability() { return mutationProb; }

  //! Get the mutation strength.
  double MutationStrength() const { return mutationStrength; }
  //! Modify the mutation strength.
  double& MutationStrength() { return mutationStrength; }

  //! Get the tolerance.
  double Epsilon() const { return epsilon; }
  //! Modify the tolerance.
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

  /**
   * Retrieve the best front (the Pareto frontier).  This returns an empty
   * vector until `Optimize()` has been called.  Note that this function is
   * deprecated and will be removed in ensmallen 3.x!  Use `ParetoFront()`
   * instead.
   */
  ens_deprecated const std::vector<arma::mat>& Front()
  {
    if (rcFront.size() == 0)
    {
      // Match the old return format.
      for (size_t i = 0; i < paretoFront.n_slices; ++i)
      {
        rcFront.push_back(arma::mat(paretoFront.slice(i)));
      }
    }

    return rcFront;
  }

 private:
  
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

 
  template<typename MatType>
  void BinaryTournamentSelection(std::vector<MatType>& population,
                                 const MatType& lowerBound,
                                 const MatType& upperBound);

 
  template<typename MatType>
  void Crossover(MatType& childA,
                 MatType& childB,
                 const MatType& parentA,
                 const MatType& parentB);

  
  template<typename MatType>
  void Mutate(MatType& child,
              const MatType& lowerBound,
              const MatType& upperBound);

  /**
   * Sort the candidate population using their domination count and the set of
   * dominated nodes.
   *
   * @tparam MatType Type of matrix to optimize.
   * @param fronts The population is sorted into these Pareto fronts. The first
   *     front is the best, the second worse and so on.
   * @param ranks The assigned ranks, used for crowding distance based sorting.
   * @param calculatedObjectives The previously calculated objectives.
   */
  template<typename MatType>
  void FastNonDominatedSort(
      std::vector<std::vector<size_t> >& fronts,
      std::vector<size_t>& ranks,
      std::vector<arma::Col<typename MatType::elem_type> >& calculatedObjectives);

  /**
   * Operator to check if one candidate Pareto-dominates the other.
   *
   * A candidate is said to dominate the other if it is at least as good as the
   * other candidate for all the objectives and there exists at least one
   * objective for which it is strictly better than the other candidate.
   *
   * @tparam MatType Type of matrix to optimize.
   * @param calculatedObjectives The previously calculated objectives.
   * @param candidateP The candidate being compared from the elite population.
   * @param candidateQ The candidate being compared against.
   * @return true if candidateP Pareto dominates candidateQ, otherwise, false.
   */
  template<typename MatType>
  bool Dominates(
      std::vector<arma::Col<typename MatType::elem_type> >& calculatedObjectives,
      size_t candidateP,
      size_t candidateQ);

  template <typename MatType>
  void FitnessAssignment(
      const std::vector<size_t>& front,
      std::vector<arma::Col<typename MatType::elem_type>>& calculatedObjectives,
      std::vector<typename MatType::elem_type>& fitness);
  template<typename MatType>
  bool FitnessOperator(size_t idxP,
                        size_t idxQ,
                        const std::vector<size_t>& ranks,
                        const std::vector<typename MatType::elem_type>& fitness);

  //! The number of objectives being optimised for.
  size_t numObjectives;

  //! The numbeer of variables used per objectives.
  size_t numVariables;

  //! The number of candidates in the population.
  size_t populationSize;

  //! Maximum number of generations before termination criteria is met.
  size_t maxGenerations;

  //! Probability that crossover will occur.
  double crossoverProb;

  //! Probability that mutation will occur.
  double mutationProb;

  //! Strength of the mutation.
  double mutationStrength;

  //! The tolerance for termination.
  double epsilon;

  //! Lower bound of the initial swarm.
  arma::vec lowerBound;

  //! Upper bound of the initial swarm.
  arma::vec upperBound;

  //! The set of all the Pareto optimal points.
  //! Stored after Optimize() is called.
  arma::cube paretoSet;

  //! The set of all the Pareto optimal objective vectors.
  //! Stored after Optimize() is called.
  arma::cube paretoFront;

  //! A different representation of the Pareto front, for reverse compatibility
  //! purposes.  This can be removed when ensmallen 3.x is released!  (Along
  //! with `Front()`.)  This is only populated when `Front()` is called.
  std::vector<arma::mat> rcFront;
};

} // namespace ens

// Include implementation.
#include "spea2_impl.hpp"

#endif
