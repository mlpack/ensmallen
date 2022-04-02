/**
 * @file nadam_update.hpp
 * @author Sourabh Varshney
 *
 * Nadam update rule. Nadam is an optimizer that combines the effect of Adam
 * and NAG to the gradient descent to improve its Performance.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_ADAM_NADAM_UPDATE_HPP
#define ENSMALLEN_ADAM_NADAM_UPDATE_HPP

namespace ens {

/**
 * Nadam is an optimizer that combines the Adam and NAG optimization strategies.
 *
 * For more information, see the following.
 *
 * @code
 * @techreport{Dozat2015,
 *   title       = {Incorporating Nesterov momentum into Adam},
 *   author      = {Timothy Dozat},
 *   institution = {Stanford University},
 *   address     = {Stanford},
 *   year        = {2015},
 *   url         = {https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ}
 * }
 * @endcode
 */
class NadamUpdate
{
 public:
  /**
   * Construct the Nadam update policy with the given parameters.
   *
   * @param epsilon The epsilon value used to initialise the squared gradient
   *        parameter.
   * @param beta1 The smoothing parameter.
   * @param beta2 The second moment coefficient
   * @param scheduleDecay The decay parameter for decay coefficients
   */
  NadamUpdate(const double epsilon = 1e-8,
              const double beta1 = 0.9,
              const double beta2 = 0.99,
              const double scheduleDecay = 4e-3) :
      epsilon(epsilon),
      beta1(beta1),
      beta2(beta2),
      scheduleDecay(scheduleDecay)
  {
    // Nothing to do.
  }

  //! Get the value used to initialise the squared gradient parameter.
  double Epsilon() const { return epsilon; }
  //! Modify the value used to initialise the squared gradient parameter.
  double& Epsilon() { return epsilon; }

  //! Get the smoothing parameter.
  double Beta1() const { return beta1; }
  //! Modify the smoothing parameter.
  double& Beta1() { return beta1; }

  //! Get the second moment coefficient.
  double Beta2() const { return beta2; }
  //! Modify the second moment coefficient.
  double& Beta2() { return beta2; }

  //! Get the decay parameter for decay coefficients
  double ScheduleDecay() const { return scheduleDecay; }
  //! Modify the decay parameter for decay coefficients
  double& ScheduleDecay() { return scheduleDecay; }

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
     * This constructor is called by the optimizer before the start of the
     * iteration update process.
     *
     * @param parent Instantiated NadamUpdate parent object.
     * @param rows Number of rows in the gradient matrix.
     * @param cols Number of columns in the gradient matrix.
     */
    Policy(NadamUpdate& parent, const size_t rows, const size_t cols) :
        parent(parent),
        cumBeta1(1),
        iteration(0)
    {
      m.zeros(rows, cols);
      v.zeros(rows, cols);
    }

    /**
     * Update step for Nadam.
     *
     * @param iterate Parameters that minimize the function.
     * @param stepSize Step size to be used for the given iteration.
     * @param gradient The gradient matrix.
     */
    void Update(MatType& iterate,
                const double stepSize,
                const GradType& gradient)
    {
      // Increment the iteration counter variable.
      ++iteration;

      // And update the iterate.
      m *= parent.beta1;
      m += (1 - parent.beta1) * gradient;

      v *= parent.beta2;
      v += (1 - parent.beta2) * gradient % gradient;

      double beta1T = parent.beta1 * (1 - (0.5 *
          std::pow(0.96, iteration * parent.scheduleDecay)));

      double beta1T1 = parent.beta1 * (1 - (0.5 *
          std::pow(0.96, (iteration + 1) * parent.scheduleDecay)));

      cumBeta1 *= beta1T;

      const double biasCorrection1 = 1.0 - cumBeta1;
      const double biasCorrection2 = 1.0 - std::pow(parent.beta2, iteration);
      const double biasCorrection3 = 1.0 - (cumBeta1 * beta1T1);

      /* Note :- arma::sqrt(v) + epsilon * sqrt(biasCorrection2) is approximated
       * as arma::sqrt(v) + epsilon
       */
      iterate -= (stepSize * (((1 - beta1T) / biasCorrection1) * gradient
          + (beta1T1 / biasCorrection3) * m) * sqrt(biasCorrection2))
          / (arma::sqrt(v) + parent.epsilon);
    }

   private:
    // Instantiated parent object.
    NadamUpdate& parent;

    // The exponential moving average of gradient values.
    GradType m;

    // The exponential moving average of squared gradient values.
    GradType v;

    // The cumulative product of decay coefficients.
    double cumBeta1;

    // The number of iterations.
    size_t iteration;
  };

 private:
  // The epsilon value used to initialise the squared gradient parameter.
  double epsilon;

  // The smoothing parameter.
  double beta1;

  // The second moment coefficient.
  double beta2;

  // The decay parameter for decay coefficients.
  double scheduleDecay;
};

} // namespace ens

#endif
