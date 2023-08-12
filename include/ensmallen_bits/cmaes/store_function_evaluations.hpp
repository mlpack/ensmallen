/**
 * @file store_function_evaluations.hpp
 * @author Suvarsha Chennareddy
 *
 * Implementation of the store function evaluations callback function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_CMAES_CALLBACKS_STORE_FUNCTION_EVALUATIONS_HPP
#define ENSMALLEN_CMAES_CALLBACKS_STORE_FUNCTION_EVALUATIONS_HPP
namespace ens {

/**
 * Store the function evaluations in an archive, based on the Evaluate callback function.
 */

template<typename CoordinateType = arma::mat>
class StoreFunctionEvaluations
{
 public:
  /**
   * Set up the store model class, which stores the function evaluations 
   * in an archive.
   */
   StoreFunctionEvaluations() :
     start(false)
  { /* Nothing to do here. */ }

  /**
   * Callback function called after any call to Evaluate().
   *
   * @param selectionPolicy The selection policy used to evaluate the function.
   * @param functionNum Number of sub function evalutions of the seperable function.
   * @param coordinates Starting point.
   * @param objective Objective value of the current point.
   */
  template<typename SelectionPolicyType, typename MatType>
  void Evaluate(SelectionPolicyType& selectionPolicy,
                size_t&  functionNum,
                const MatType& coordinates,
                const double objective)
  {

    if (functionNum == 0 && start)
      archive.push_back(std::make_pair( lastCoordinates, lastObjective ));

    lastCoordinates = coordinates;
    lastObjective = objective;

    if (!start) start = true;
  }

  //! Get the archive function evaluations.
  const std::vector < std::pair < CoordinateType, double>>& Archive() const 
  { return archive; }
  //! Modify the archive of function evaluations.
  std::vector < std::pair < CoordinateType, double>>& Archive() 
  { return archive; }

 private:
  //! Locally-stored archive of function evaluations.
  std::vector < std::pair < CoordinateType, double>> archive;

  //! A flag variable indicating when to start adding to the archive.
  bool start;

  //! The input coordinates passed last into the seperable function.
  CoordinateType lastCoordinates;

  //! The objective value of lastCoordinates.
  double lastObjective;
};

} // namespace ens

#endif