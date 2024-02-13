/**
 * @file query_front.hpp
 * @author Nanubala Gnana Sai
 *
 * Implementation of the query front callback function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CALLBACKS_QUERY_FRONT_HPP
#define ENSMALLEN_CALLBACKS_QUERY_FRONT_HPP

namespace ens {

/**
 * Query the current Pareto Front after every GenerationalStepTaken callback function.
 */
class QueryFront
{
 public:
  /**
   * Set up the query front callback class with the specified inputs.
   *
   * @param queryRate The frequency at which the Pareto Front is queried.
   * @param paretoFrontArray A reference to a vector of cube to store the
   *     queried fronts.
   */
  QueryFront(const size_t queryRate,
             std::vector<arma::cube>& paretoFrontArray) :
      queryRate(queryRate),
      paretoFrontArray(paretoFrontArray),
      genCounter(0)
  { /* Nothing to do here */ }

  /**
   * Callback function called at the end of a single generational run.
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param objectives The set of calculated objectives so far.
   * @param frontIndices The indices of the members belonging to Pareto Front.
   */
  template<typename OptimizerType,
           typename FunctionType,
           typename MatType,
           typename ObjectivesVecType,
           typename IndicesType>
  bool GenerationalStepTaken(OptimizerType& /* opt */,
                             FunctionType& /* function */,
                             const MatType& /* coordinates */,
                             const ObjectivesVecType& objectives,
                             const IndicesType& frontIndices)
  {
    arma::cube currentParetoFront{};

    if (genCounter % queryRate == 0)
    {
      currentParetoFront.resize(objectives[0].n_rows, objectives[0].n_cols,
          frontIndices[0].size());
      for (size_t solutionIdx = 0; solutionIdx < frontIndices[0].size();
          ++solutionIdx)
      {
        currentParetoFront.slice(solutionIdx) = arma::conv_to<arma::mat>::from(
            objectives[frontIndices[0][solutionIdx]]);
      }

      paretoFrontArray.emplace_back(std::move(currentParetoFront));
    }

    ++genCounter;
    return false;
  }


 private:
  //! The rate of query.
  size_t queryRate;
  //! A reference to the array of pareto fronts.
  std::vector<arma::cube>& paretoFrontArray;
  //! A counter for the current generation.
  size_t genCounter;
};

} // namespace ens

#endif
