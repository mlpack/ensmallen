/**
 * @file adaptive_stepsize.hpp
 * @author Marcus Edel
 *
 * Definition of the adaptive stepsize technique as described in:
 * "Big Batch SGD: Automated Inference using Adaptive Batch Sizes" by
 * S. De et al.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_BIGBATCH_SGD_ADAPTIVE_STEPSIZE_HPP
#define ENSMALLEN_BIGBATCH_SGD_ADAPTIVE_STEPSIZE_HPP

namespace ens {

/**
 * Definition of the adaptive stepize technique, a non-monotonic stepsize scheme
 * that uses curvature estimates to propose new stepsize choices.
 * direction.
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
class AdaptiveStepsize
{
 public:
  /**
   * Construct the AdaptiveStepsize object with the given function and
   * parameters. The defaults here are not necessarily good for the given
   * problem, so it is suggested that the values used be tailored to the task at
   * hand.
   *
   * @param backtrackStepSize The backtracking step size for each iteration.
   * @param searchParameter The backtracking search parameter for each
   *        iteration.
   */
  AdaptiveStepsize(const double backtrackStepSize = 0.5,
                   const double searchParameter = 0.1) :
      backtrackStepSize(backtrackStepSize),
      searchParameter(searchParameter)
  { /* Nothing to do here. */ }

  //! Get the backtracking step size.
  double BacktrackStepSize() const { return backtrackStepSize; }
  //! Modify the backtracking step size.
  double& BacktrackStepSize() { return backtrackStepSize; }

  //! Get the search parameter.
  double SearchParameter() const { return searchParameter; }
  //! Modify the search parameter.
  double& SearchParameter() { return searchParameter; }


  template<typename MatType>
  class Policy
  {
   public:
    // Create the instantiated object.
    Policy(AdaptiveStepsize& parent) : parent(parent) { }

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
                double& sampleVariance,
                const size_t offset,
                const size_t batchSize,
                const size_t backtrackingBatchSize,
                const bool /* reset */)
    {
      Backtracking(function, stepSize, iterate, gradient, gradientNorm, offset,
          backtrackingBatchSize);

      // Update the iterate.
      iterate -= stepSize * gradient;

      // Update Gradient & calculate curvature of quadratic approximation.
      GradType functionGradient(iterate.n_rows, iterate.n_cols);
      GradType gradPrevIterate(iterate.n_rows, iterate.n_cols);
      GradType functionGradientPrev(iterate.n_rows, iterate.n_cols);

      double vB = 0;
      GradType delta0, delta1;

      // Initialize previous iterate, if not already initialized.
      if (iteratePrev.is_empty())
      {
        iteratePrev.zeros(iterate.n_rows, iterate.n_cols);
      }

      // Compute the stochastic gradient estimation.
      function.Gradient(iterate, offset, gradient, 1);
      function.Gradient(iteratePrev, offset, gradPrevIterate, 1);

      delta1 = gradient;

      for (size_t j = 1, k = 1; j < backtrackingBatchSize; ++j, ++k)
      {
        function.Gradient(iterate, offset + j, functionGradient, 1);
        delta0 = delta1 + (functionGradient - delta1) / k;

        // Compute sample variance.
        vB += arma::norm(functionGradient - delta1, 2.0) *
            arma::norm(functionGradient - delta0, 2.0);

        delta1 = delta0;
        gradient += functionGradient;

        // Used for curvature calculation.
        function.Gradient(iteratePrev, offset + j, functionGradientPrev, 1);
        gradPrevIterate += functionGradientPrev;
      }

      // Update sample variance & norm of the gradient.
      sampleVariance = vB;
      gradientNorm = std::pow(arma::norm(gradient / backtrackingBatchSize, 2),
          2.0);

      // Compute curvature.
      double v = arma::trace(arma::trans(iterate - iteratePrev) *
          (gradient - gradPrevIterate)) /
          std::pow(arma::norm(iterate - iteratePrev, 2), 2.0);

      // Update previous iterate.
      iteratePrev = iterate;

      // TODO: Develop an absolute strategy to deal with stepSizeDecay updates
      // in case we arrive at local minima. See #1469 for more details.
      double stepSizeDecay = 0;
      if (gradientNorm && sampleVariance && batchSize)
      {
        if (batchSize < function.NumFunctions())
        {
          stepSizeDecay = (1 - (1 / ((double) batchSize - 1) * sampleVariance) /
              (batchSize * gradientNorm)) / v;
        }
        else
        {
          stepSizeDecay = 1 / v;
        }
      }

      // Stepsize smoothing.
      stepSize *= (1 - ((double) batchSize / function.NumFunctions()));
      stepSize += stepSizeDecay * ((double) batchSize /
          function.NumFunctions());

      Backtracking(function, stepSize, iterate, gradient, gradientNorm, offset,
          backtrackingBatchSize);
    }

    /**
     * Definition of the backtracking line search algorithm based on the
     * Armijo–Goldstein condition to determine the maximum amount to move along
     * the given search direction.
     *
     * @tparam SeparableFunctionType Type of the function to be optimized.
     * @param function Function to be optimized (minimized).
     * @param stepSize Step size to be used for the given iteration.
     * @param iterate Parameters that minimize the function.
     * @param gradient The gradient matrix.
     * @param gradientNorm The gradient norm to be used for the given iteration.
     * @param offset The batch offset to be used for the given iteration.
     * @param backtrackingBatchSize The backtracking batch size.
     */
    template<typename SeparableFunctionType,
             typename GradType>
    void Backtracking(SeparableFunctionType& function,
                      double& stepSize,
                      const MatType& iterate,
                      const GradType& gradient,
                      const double gradientNorm,
                      const size_t offset,
                      const size_t backtrackingBatchSize)
    {
      typedef typename MatType::elem_type ElemType;

      ElemType overallObjective = function.Evaluate(iterate,
          offset, backtrackingBatchSize);

      MatType iterateUpdate = iterate - (stepSize * gradient);
      ElemType overallObjectiveUpdate = function.Evaluate(iterateUpdate, offset,
          backtrackingBatchSize);

      while (overallObjectiveUpdate >
          (overallObjective - parent.searchParameter * stepSize *
           gradientNorm))
      {
        stepSize *= parent.backtrackStepSize;

        iterateUpdate = iterate - (stepSize * gradient);
        overallObjectiveUpdate = function.Evaluate(iterateUpdate, offset,
            backtrackingBatchSize);
      }
    }

   private:
    //! Reference to parent.
    AdaptiveStepsize& parent;

    //! Last function parameters value.
    MatType iteratePrev;
  };

 private:
  //! The backtracking step size for each iteration.
  double backtrackStepSize;

  //! The search parameter for each iteration.
  double searchParameter;
};

} // namespace ens

#endif // ENSMALLEN_BIGBATCH_SGD_ADAPTIVE_STEPSIZE_HPP
