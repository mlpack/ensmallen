/**
 * @file count_function_evaluations.hpp
 * @author Suvarsha Chennareddy
 *
 * Implementation of the count function evaluations callback function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CMAES_CALLBACKS_COUNT_FUNCTION_EVALUATIONS_HPP
#define ENSMALLEN_CMAES_CALLBACKS_COUNT_FUNCTION_EVALUATIONS_HPP

namespace ens {

/**
 * Count the number of function evaluations, based on the Evaluate callback function.
 */
class CountFunctionEvaluations
{
 public:
  /**
   * Set up the count model class, which counts the number of function 
   * evaluations .
   */
   CountFunctionEvaluations() : 
     numFunctionEvaluations(0)
  { /* Nothing to do here. */ }

  /**
   * Callback function called after any call to Evaluate().
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param objective Objective value of the current point.
   */
  template<typename OptimizerType, typename FunctionType, typename MatType>
  void Evaluate(OptimizerType& /* optimizer */,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const double /* objective */)
  {
    numFunctionEvaluations++;
  }

  //! Get the number of function evaluations.
  size_t const& NumFunctionEvaluations() const { return numFunctionEvaluations; }
  //! Modify the number of function evaluations.
  size_t& NumFunctionEvaluations() { return numFunctionEvaluations; }

 private:
  //! Locally-stored number of function evaluations.
  size_t numFunctionEvaluations;
};

} // namespace ens

#endif