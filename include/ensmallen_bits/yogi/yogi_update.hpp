/**
 * @file yogi_update.hpp
 * @author Marcus Edel
 *
 * Implements the Yogi Optimizer. Yogi is a variant of Adam with more fine
 * grained effective learning rate control.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_YOGI_YOGI_UPDATE_HPP
#define ENSMALLEN_YOGI_YOGI_UPDATE_HPP

namespace ens {

/**
 * Yogi builds upon the Adam update strategy but provides more fine grained
 * effective learning rate control.
 *
 * For more information, see the following.
 *
 * @code
 * @inproceedings{Zaheer2018,
 *   author    = {Zaheer, Manzil and Reddi, Sashank J. and Sachan, Devendra
 *                and Kale, Satyen and Kumar, Sanjiv},
 *   title     = {Adaptive Methods for Nonconvex Optimization},
 *   year      = {2018},
 *   publisher = {Curran Associates Inc.},
 *   booktitle = {Proceedings of the 32nd International Conference on Neural
 *                Information Processing Systems},
 *   pages     = {9815–9825},
 *   series    = {NIPS'18}
 * }
 * @endcode
 */
class YogiUpdate
{
 public:
  /**
   * Construct the Yogi update policy with the given parameters.
   *
   * @param epsilon The epsilon value used to initialise the squared gradient
   *     parameter.
   * @param beta1 The smoothing parameter.
   * @param beta2 The second moment coefficient.
   * @param v1 The first quasi-hyperbolic term.
   * @param v1 The second quasi-hyperbolic term.
   */
  YogiUpdate(const double epsilon = 1e-8,
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
     * @param parent YogiUpdate object.
     * @param rows Number of rows in the gradient matrix.
     * @param cols Number of columns in the gradient matrix.
     */
    Policy(YogiUpdate& parent, const size_t rows, const size_t cols) :
        parent(parent)
    {
      m.zeros(rows, cols);
      v.zeros(rows, cols);
    }

    /**
     * Update step for Yogi.
     *
     * @param iterate Parameters that minimize the function.
     * @param stepSize Step size to be used for the given iteration.
     * @param gradient The gradient matrix.
     */
    void Update(MatType& iterate,
                const double stepSize,
                const GradType& gradient)
    {
      m *= parent.beta1;
      m += (1 - parent.beta1) * gradient;

      const MatType gSquared = arma::square(gradient);
      v -= (1 - parent.beta2) * arma::sign(v - gSquared) % gSquared;

      // Now update the iterate.
      iterate -= stepSize * m / (arma::sqrt(v) + parent.epsilon);
    }

   private:
    //! Instantiated parent object.
    YogiUpdate& parent;

    //! The exponential moving average of gradient values.
    GradType m;

    // The exponential moving average of squared gradient values.
    GradType v;
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
