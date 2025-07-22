/**
 * @file amsgrad_update.hpp
 * @author Haritha Nair
 *
 * Implementation of AMSGrad optimizer. AMSGrad is an exponential moving average
 * optimizer that dynamically adapts over time with guaranteed convergence.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_AMS_GRAD_AMS_GRAD_UPDATE_HPP
#define ENSMALLEN_AMS_GRAD_AMS_GRAD_UPDATE_HPP

namespace ens {

/**
 * AMSGrad is an exponential moving average variant which along with having
 * benefits of optimizers like Adam and RMSProp, also guarantees convergence.
 * Unlike Adam, it uses maximum of past squared gradients rather than their
 * exponential average for updation.
 *
 * For more information, see the following.
 *
 * @code
 * @article{
 *   title = {On the convergence of Adam and beyond},
 *   url   = {https://openreview.net/pdf?id=ryQu7f-RZ}
 *   year  = {2018}
 * }
 * @endcode
 */
class AMSGradUpdate
{
 public:
  /**
   * Construct the AMSGrad update policy with the given parameters.
   *
   * @param epsilon The epsilon value used to initialise the squared gradient
   *        parameter.
   * @param beta1 The smoothing parameter.
   * @param beta2 The second moment coefficient.
   */
  AMSGradUpdate(const double epsilon = 1e-8,
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
    typedef typename MatType::elem_type ElemType;

    /**
     * This constructor is called by the SGD Optimize() method before the start
     * of the iteration update process.
     *
     * @param parent Instantiated AMSGradUpdate parent object.
     * @param rows Number of rows in the gradient matrix.
     * @param cols Number of columns in the gradient matrix.
     */
    Policy(AMSGradUpdate& parent, const size_t rows, const size_t cols) :
        parent(parent),
        epsilon(ElemType(parent.epsilon)),
        beta1(ElemType(parent.beta1)),
        beta2(ElemType(parent.beta2)),
        iteration(0)
    {
      m.zeros(rows, cols);
      v.zeros(rows, cols);
      vImproved.zeros(rows, cols);

      // Attempt to detect underflow.
      if (epsilon == ElemType(0) && parent.epsilon != 0.0)
        epsilon = 100 * std::numeric_limits<ElemType>::epsilon();
    }

    /**
     * Update step for AMSGrad.
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
      m *= beta1;
      m += (1 - beta1) * gradient;

      v *= beta2;
      v += (1 - beta2) * (gradient % gradient);

      const ElemType biasCorrection1 = 1 - std::pow(beta1, ElemType(iteration));
      const ElemType biasCorrection2 = 1 - std::pow(beta2, ElemType(iteration));

      // Element wise maximum of past and present squared gradients.
      vImproved = max(vImproved, v);

      iterate -= (ElemType(stepSize) *
          std::sqrt(biasCorrection2) / biasCorrection1) *
          m / (sqrt(vImproved) + epsilon);
    }

   private:
    // Instantiated parent AMSGradUpdate object.
    AMSGradUpdate& parent;

    // The exponential moving average of gradient values.
    GradType m;

    // The exponential moving average of squared gradient values.
    GradType v;

    // The optimal squared gradient value.
    GradType vImproved;

    // Parameters converted to the element type of the optimization.
    ElemType epsilon;
    ElemType beta1;
    ElemType beta2;

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
