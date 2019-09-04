/**
 * @file rmsprop_update.hpp
 * @author Ryan Curtin
 * @author Marcus Edel
 * @author Vivek Pal
 *
 * RMSProp optimizer. RMSProp is an optimizer that utilizes the magnitude of
 * recent gradients to normalize the gradients.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_RMSPROP_RMSPROP_UPDATE_HPP
#define ENSMALLEN_RMSPROP_RMSPROP_UPDATE_HPP

namespace ens {

/**
 * RMSProp is an optimizer that utilizes the magnitude of recent gradients to
 * normalize the gradients. In its basic form, given a step rate \f$ \gamma \f$
 * and a decay term \f$ \alpha \f$ we perform the following updates:
 *
 * \f{eqnarray*}{
 * r_t &=& (1 - \gamma) f'(\Delta_t)^2 + \gamma r_{t - 1} \\
 * v_{t + 1} &=& \frac{\alpha}{\sqrt{r_t}}f'(\Delta_t) \\
 * \Delta_{t + 1} &=& \Delta_t - v_{t + 1}
 * \f}
 *
 * For more information, see the following.
 *
 * @code
 * @misc{tieleman2012,
 *   title = {Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine
 *            Learning},
 *   year  = {2012}
 * }
 * @endcode
 */
class RMSPropUpdate
{
 public:
  /**
   * Construct the RMSProp update policy with the given parameters.
   *
   * @param epsilon The epsilon value used to initialise the squared gradient
   *        parameter.
   * @param alpha The smoothing parameter.
   */
  RMSPropUpdate(const double epsilon = 1e-8,
                const double alpha = 0.99) :
    epsilon(epsilon),
    alpha(alpha)
  {
    // Nothing to do.
  }

  //! Get the value used to initialise the squared gradient parameter.
  double Epsilon() const { return epsilon; }
  //! Modify the value used to initialise the squared gradient parameter.
  double& Epsilon() { return epsilon; }

  //! Get the smoothing parameter.
  double Alpha() const { return alpha; }
  //! Modify the smoothing parameter.
  double& Alpha() { return alpha; }

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
     * This constructor is called by the SGD Optimize() method before the start
     * of the iteration update process.
     *
     * @param parent AdamUpdate object.
     * @param rows Number of rows in the gradient matrix.
     * @param cols Number of columns in the gradient matrix.
     */
    Policy(RMSPropUpdate& parent, const size_t rows, const size_t cols) :
        parent(parent)
    {
      // Leaky sum of squares of parameter gradient.
      meanSquaredGradient.zeros(rows, cols);
    }

    /**
     * Update step for RMSProp.
     *
     * @param iterate Parameters that minimize the function.
     * @param stepSize Step size to be used for the given iteration.
     * @param gradient The gradient matrix.
     */
    void Update(MatType& iterate,
                const double stepSize,
                const GradType& gradient)
    {
      meanSquaredGradient *= parent.alpha;
      meanSquaredGradient += (1 - parent.alpha) * (gradient % gradient);
      iterate -= stepSize * gradient / (arma::sqrt(meanSquaredGradient) +
          parent.epsilon);
    }

   private:
    // Leaky sum of squares of parameter gradient.
    GradType meanSquaredGradient;
    // Reference to instantiated parent object.
    RMSPropUpdate& parent;
  };

 private:
  // The epsilon value used to initialise the squared gradient parameter.
  double epsilon;

  // The smoothing parameter.
  double alpha;
};

} // namespace ens

#endif
