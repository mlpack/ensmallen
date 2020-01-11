/**
 * @file ada_delta_update.hpp
 * @author Vasanth Kalingeri
 * @author Abhinav Moudgil
 *
 * AdaDelta update for Stochastic Gradient Descent.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_ADA_DELTA_ADA_DELTA_UPDATE_HPP
#define ENSMALLEN_ADA_DELTA_ADA_DELTA_UPDATE_HPP

namespace ens {

/**
 * Implementation of the AdaDelta update policy. AdaDelta is an optimizer that
 * uses two ideas to improve upon the two main drawbacks of the AdaGrad method:
 *
 * - Accumulate Over Window
 * - Correct Units with Hessian Approximation
 *
 * For more information, see the following.
 *
 * @code
 * @article{Zeiler2012,
 *   author  = {Matthew D. Zeiler},
 *   title   = {{ADADELTA:} An Adaptive Learning Rate Method},
 *   journal = {CoRR},
 *   year    = {2012}
 * }
 * @endcode
 *
 */
class AdaDeltaUpdate
{
 public:
  /**
   * Construct the AdaDelta update policy with given rho and epsilon parameters.
   *
   * @param rho The smoothing parameter.
   * @param epsilon The epsilon value used to initialise the squared gradient
   *    parameter.
   */
  AdaDeltaUpdate(const double rho = 0.95, const double epsilon = 1e-6) :
      rho(rho),
      epsilon(epsilon)
  {
    // Nothing to do.
  }

  //! Get the smoothing parameter.
  double Rho() const { return rho; }
  //! Modify the smoothing parameter.
  double& Rho() { return rho; }

  //! Get the value used to initialise the mean squared gradient parameter.
  double Epsilon() const { return epsilon; }
  //! Modify the value used to initialise the mean squared gradient parameter.
  double& Epsilon() { return epsilon; }

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
     * This constructor is called by the SGD optimizer method before the start
     * of the iteration update process. In AdaDelta update policy, the mean
     * squared and the delta mean squared gradient matrices are initialized to
     * the zeros matrix with the same size as gradient matrix (see ens::SGD<>).
     *
     * @param parent AdaDeltaUpdate object.
     * @param rows Number of rows in the gradient matrix.
     * @param cols Number of columns in the gradient matrix.
     */
    Policy(AdaDeltaUpdate& parent, const size_t rows, const size_t cols) :
        parent(parent)
    {
      meanSquaredGradient.zeros(rows, cols);
      meanSquaredGradientDx.zeros(rows, cols);
    }

    /**
     * Update step for SGD. The AdaDelta update dynamically adapts over time
     * using only first order information. Additionally, AdaDelta requires no
     * manual tuning of a learning rate.
     *
     * @param iterate Parameters that minimize the function.
     * @param stepSize Step size to be used for the given iteration.
     * @param gradient The gradient matrix.
     */
    void Update(MatType& iterate,
                const double stepSize,
                const GradType& gradient)
    {
      // Accumulate gradient.
      meanSquaredGradient *= parent.rho;
      meanSquaredGradient += (1 - parent.rho) * (gradient % gradient);
      GradType dx = arma::sqrt((meanSquaredGradientDx + parent.epsilon) /
          (meanSquaredGradient + parent.epsilon)) % gradient;

      // Accumulate updates.
      meanSquaredGradientDx *= parent.rho;
      meanSquaredGradientDx += (1 - parent.rho) * (dx % dx);

      // Apply update.
      iterate -= (stepSize * dx);
    }

   private:
    // The instantiated parent class.
    AdaDeltaUpdate& parent;

    // The mean squared gradient matrix.
    GradType meanSquaredGradient;

    // The delta mean squared gradient matrix.
    GradType meanSquaredGradientDx;
  };

 private:
  // The smoothing parameter.
  double rho;

  // The epsilon value used to initialise the mean squared gradient parameter.
  double epsilon;
};

} // namespace ens

#endif
