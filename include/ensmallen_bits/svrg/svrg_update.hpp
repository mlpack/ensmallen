/**
 * @file svrg_update.hpp
 * @author Marcus Edel
 *
 * Vanilla update for stochastic variance reduced gradient (SVRG).
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SVRG_SVRG_UPDATE_HPP
#define ENSMALLEN_SVRG_SVRG_UPDATE_HPP

namespace ens {

/**
 * Vanilla update policy for Stochastic variance reduced gradient (SVRG).
 * The following update scheme is used to update SGD in every iteration:
 */
class SVRGUpdate
{
 public:
  /**
   * The UpdatePolicyType policy classes must contain an internal 'Policy'
   * template class with two template arguments: MatType and GradType.  This is
   * instantiated at the start of the optimization.
   */
  template<typename MatType, typename GradType>
  class Policy
  {
   public:
    /**
     * This is called by the optimizer method before the start of the iteration
     * update process.
     *
     * @param parent Instantiated parent class.
     * @param rows Number of rows in the gradient matrix.
     * @param cols Number of columns in the gradient matrix.
     */
    Policy(SVRGUpdate& /* parent */,
           const size_t /* rows */,
           const size_t /* cols */)
    { /* Do nothing. */ }

    /**
     * Update step for SVRG. The function parameters are updated in the negative
     * direction of the gradient.
     *
     * @param iterate Parameters that minimize the function.
     * @param fullGradient The computed full gradient.
     * @param gradient The current gradient matrix at time t.
     * @param gradient0 The old gradient matrix at time t - 1.
     * @param batchSize Batch size to be used for the given iteration.
     * @param stepSize Step size to be used for the given iteration.
     */
    void Update(MatType& iterate,
                const GradType& fullGradient,
                const GradType& gradient,
                const GradType& gradient0,
                const size_t batchSize,
                const double stepSize)
    {
      // Perform the vanilla SVRG update.
      iterate -= stepSize * (fullGradient + (gradient - gradient0) /
          (double) batchSize);
    }
  };
};

} // namespace ens

#endif
