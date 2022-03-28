/**
 * @file adam_update.hpp
 * @author Ryan Curtin
 * @author Vasanth Kalingeri
 * @author Marcus Edel
 * @author Vivek Pal
 *
 * Adam optimizer. Adam is an an algorithm for first-order gradient-based
 * optimization of stochastic objective functions, based on adaptive estimates
 * of lower-order moments.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_ADAM_ADAM_UPDATE_HPP
#define ENSMALLEN_ADAM_ADAM_UPDATE_HPP

namespace ens {

/**
 * Adam is an optimizer that computes individual adaptive learning rates for
 * different parameters from estimates of first and second moments of the
 * gradients as given in the section 7 of the following paper.
 *
 * For more information, see the following.
 *
 * @code
 * @article{Kingma2014,
 *   author  = {Diederik P. Kingma and Jimmy Ba},
 *   title   = {Adam: {A} Method for Stochastic Optimization},
 *   journal = {CoRR},
 *   year    = {2014},
 *   url     = {http://arxiv.org/abs/1412.6980}
 * }
 * @endcode
 */
class AdamUpdate
{
 public:
  /**
   * Construct the Adam update policy with the given parameters.
   *
   * @param epsilon The epsilon value used to initialise the squared gradient
   *        parameter.
   * @param beta1 The smoothing parameter.
   * @param beta2 The second moment coefficient.
   */
  AdamUpdate(const double epsilon = 1e-8,
             const double beta1 = 0.9,
             const double beta2 = 0.999) :
    epsilon(epsilon),
    beta1(beta1),
    beta2(beta2)
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
    Policy(AdamUpdate& parent, const size_t rows, const size_t cols) :
        parent(parent),
        iteration(0)
    {
      m.zeros(rows, cols);
      v.zeros(rows, cols);
    }

    /**
     * Update step for Adam.
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
      v += (1 - parent.beta2) * (gradient % gradient);

      const double biasCorrection1 = 1.0 - std::pow(parent.beta1, iteration);
      const double biasCorrection2 = 1.0 - std::pow(parent.beta2, iteration);

      /**
       * It should be noted that the term, m / (arma::sqrt(v) + eps), in the
       * following expression is an approximation of the following actual term;
       * m / (arma::sqrt(v) + (arma::sqrt(biasCorrection2) * eps).
       */
      iterate -= (stepSize * std::sqrt(biasCorrection2) / biasCorrection1) *
          m / (arma::sqrt(v) + parent.epsilon);
    }

   private:
    // Instantiated parent object.
    AdamUpdate& parent;

    // The exponential moving average of gradient values.
    GradType m;

    // The exponential moving average of squared gradient values.
    GradType v;

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
};

} // namespace ens

#endif
