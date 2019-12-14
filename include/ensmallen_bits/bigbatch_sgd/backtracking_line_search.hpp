/**
 * @file backtracking_line_search.hpp
 * @author Marcus Edel
 *
 * Definition of the backtracking line search technique as described in:
 * "Big Batch SGD: Automated Inference using Adaptive Batch Sizes" by
 * S. De et al.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_BIGBATCH_SGD_BACKTRACKING_LINE_SEARCH_HPP
#define ENSMALLEN_BIGBATCH_SGD_BACKTRACKING_LINE_SEARCH_HPP

namespace ens {

/**
 * Definition of the backtracking line search algorithm based on the
 * Armijo–Goldstein condition to determine the maximum amount to move along the
 * given search direction.
 *
 * For more information, please refer to:
 *
 * @code
 * @article{De2017,
 *   title   = {Big Batch {SGD:} Automated Inference using Adaptive Batch
 *              Sizes},
 *   author  = {Soham De and Abhay Kumar Yadav and David W. Jacobs and
                Tom Goldstein},
 *   journal = {CoRR},
 *   year    = {2017},
 *   url     = {http://arxiv.org/abs/1610.05792},
 * }
 * @endcode
 */
class BacktrackingLineSearch
{
 public:
  /**
   * Construct the BacktrackingLineSearch object with the given function and
   * parameters. The defaults here are not necessarily good for the given
   * problem, so it is suggested that the values used be tailored to the task at
   * hand.
   *
   * @param function Function to be optimized (minimized).
   */
  BacktrackingLineSearch(const double searchParameter = 0.1) :
      searchParameter(searchParameter)
  { /* Nothing to do here. */ }

  //! Get the search parameter.
  double SearchParameter() const { return searchParameter; }
  //! Modify the search parameter.
  double& SearchParameter() { return searchParameter; }

  template<typename MatType>
  class Policy
  {
   public:
    // Instantiate the policy with the given parent.
    Policy(BacktrackingLineSearch& parent) : parent(parent) { }

    /**
     * This function is called in each iteration.
     *
     * @tparam SeparableFunctionType Type of the function to be optimized.
     * @param function Function to be optimized (minimized).
     * @param stepSize Step size to be used for the given iteration.
     * @param iterate Parameters that minimize the function.
     * @param gradient The gradient matrix.
     * @param gradientNorm The gradient norm to be used for the given iteration.
     * @param offset The batch offset to be used for the given iteration.
     * @param batchSize Batch size to be used for the given iteration.
     * @param backtrackingBatchSize Backtracking batch size to be used for the
     *        given iteration.
     * @param reset Reset the step size decay parameter.
     */
    template<typename SeparableFunctionType,
             typename GradType>
    void Update(SeparableFunctionType& function,
                double& stepSize,
                MatType& iterate,
                GradType& gradient,
                double& gradientNorm,
                double& /* sampleVariance */,
                const size_t offset,
                const size_t /* batchSize */,
                const size_t backtrackingBatchSize,
                const bool reset)
    {
      if (reset)
        stepSize *= 2;

      typedef typename MatType::elem_type ElemType;

      ElemType overallObjective = function.Evaluate(iterate, offset,
          backtrackingBatchSize);

      MatType iterateUpdate = iterate - (stepSize * gradient);
      ElemType overallObjectiveUpdate = function.Evaluate(iterateUpdate, offset,
          backtrackingBatchSize);

      while (overallObjectiveUpdate >
          (overallObjective - parent.searchParameter * stepSize *
           gradientNorm))
      {
        stepSize /= 2;

        iterateUpdate = iterate - (stepSize * gradient);
        overallObjectiveUpdate = function.Evaluate(iterateUpdate,
          offset, backtrackingBatchSize);
      }
    }

   private:
    //! Reference to instantiated parent object.
    BacktrackingLineSearch& parent;
  };

 private:
  //! The search parameter for each iteration.
  double searchParameter;
};

} // namespace ens

#endif // ENSMALLEN_BIGBATCH_SGD_BACKTRACKING_LINE_SEARCH_HPP
