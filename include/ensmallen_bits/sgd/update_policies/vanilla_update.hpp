/**
 * @file vanilla_update.hpp
 * @author Arun Reddy
 *
 * Vanilla update for Stochastic Gradient Descent.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SGD_EMPTY_UPDATE_HPP
#define ENSMALLEN_SGD_EMPTY_UPDATE_HPP

namespace ens {

/**
 * Vanilla update policy for Stochastic Gradient Descent (SGD). The following
 * update scheme is used to update SGD in every iteration:
 *
 * \f[
 * A_{j + 1} = A_j + \alpha \nabla f_i(A)
 * \f]
 *
 * where \f$ \alpha \f$ is a parameter which specifies the step size.  \f$ i \f$
 * is chosen according to \f$ j \f$ (the iteration number).
 */
class VanillaUpdate
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
     * update process.  The vanilla update doesn't initialize anything.
     *
     * @param parent Instantiated parent class.
     * @param rows Number of rows in the gradient matrix.
     * @param cols Number of columns in the gradient matrix.
     */
    Policy(const VanillaUpdate& /* parent */,
           const size_t /* rows */,
           const size_t /* cols */)
    { /* Do nothing. */ }

   /**
    * Update step for SGD.  The function parameters are updated in the negative
    * direction of the gradient.
    *
    * @param iterate Parameters that minimize the function.
    * @param stepSize Step size to be used for the given iteration.
    * @param gradient The gradient matrix.
    */
    void Update(MatType& iterate,
                const double stepSize,
                const GradType& gradient)
    {
      // Perform the vanilla SGD update.
      iterate -= stepSize * gradient;
    }
  };
};

} // namespace ens

#endif
