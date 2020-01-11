/**
 * @file nesterov_momentum_update.hpp
 * @author Sourabh Varshney
 *
 * Nesterov Momentum Update for Stochastic Gradient Descent.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SGD_NESTEROV_MOMENTUM_UPDATE_HPP
#define ENSMALLEN_SGD_NESTEROV_MOMENTUM_UPDATE_HPP

namespace ens {

/**
 * Nesterov Momentum update policy for Stochastic Gradient Descent (SGD).
 *
 * Learning with SGD can be slow. Applying Standard momentum can accelerate
 * the rate of convergence. Nesterov Momentum application can accelerate the
 * rate of convergence to O(1/k^2).
 *
 * @code
 * @techreport{Nesterov1983,
 *   title       = {A Method Of Solving A Convex Programming Problem With
 *                  Convergence Rate O(1/K^2)},
 *   author      = {Yuri Nesterov},
 *   institution = {Soviet Math. Dokl.},
 *   volume      = {27},
 *   year        = {1983},
 * }
 * @endcode
 */
class NesterovMomentumUpdate
{
 public:
  /**
   * Construct the Nesterov Momentum update policy with the given parameters.
   */
  NesterovMomentumUpdate(const double momentum = 0.5) : momentum(momentum)
  {
    // Nothing to do.
  }

  //! Get the value used to initialize the momentum coefficient.
  double Momentum() const { return momentum; }
  //! Modify the value used to initialize the momentum coefficient.
  double& Momentum() { return momentum; }

  /**
   * The UpdatePolicyType policy classes must contain an internal 'Policy'
   * template class with two template arguments: MatType and GradType.  This is
   * instantiated at the start of the optimization, and holds parameters
   * specific to an individual optimization.
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
    Policy(const NesterovMomentumUpdate& parent,
           const size_t rows,
           const size_t cols) :
        parent(parent),
        velocity(arma::zeros<MatType>(rows, cols))
    {
      // Nothing to do.
    }

    /**
     * Update step for SGD.  The momentum term makes the convergence faster on
     * the way as momentum term increases for dimensions pointing in the same
     * direction and reduces updates for dimensions whose gradients change
     * directions.
     *
     * @param iterate Parameters that minimize the function.
     * @param stepSize Step size to be used for the given iteration.
     * @param gradient The gradient matrix.
     */
    void Update(MatType& iterate,
                const double stepSize,
                const GradType& gradient)
    {
      velocity = parent.momentum * velocity - stepSize * gradient;

      iterate += parent.momentum * velocity - stepSize * gradient;
    }

   private:
    // The parent class instantiation.
    const NesterovMomentumUpdate& parent;
    // The velocity matrix.
    MatType velocity;
  };

 private:
  // The Momentum coefficient.
  double momentum;
};

} // namespace ens

#endif
