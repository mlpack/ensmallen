/**
 * @file ada_bound_update.hpp
 * @author Marcus Edel
 *
 * Implments the AdaBound Optimizer. AdaBound is a variant of Adam which
 * employs dynamic bounds on learning rates.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_ADA_BOUND_UPDATE_HPP
#define ENSMALLEN_ADA_BOUND_UPDATE_HPP

namespace ens {

/**
 * AdaBound employs dynamic bounds on learning rates to achieve a gradual and
 * smooth transition from adaptive methods to SGD.
 *
 * For more information, see the following.
 *
 * @code
 * @inproceedings{Luo2019AdaBound,
 *   author    = {Luo, Liangchen and Xiong, Yuanhao and Liu, Yan and Sun, Xu},
 *   title     = {Adaptive Gradient Methods with Dynamic Bound of Learning
 *                Rate},
 *   booktitle = {Proceedings of the 7th International Conference on Learning
 *                Representations},
 *   month     = {May},
 *   year      = {2019},
 *   address   = {New Orleans, Louisiana}
 * }
 * @endcode
 */
class AdaBoundUpdate
{
 public:
  /**
   * Construct the AdaBound update policy with the given parameters.
   *
   * @param finalLr The final (SGD) learning rate.
   * @param gamma The convergence speed of the bound functions.
   * @param epsilon The epsilon value used to initialise the squared gradient
   *        parameter.
   * @param beta1 The smoothing parameter.
   * @param beta2 The second moment coefficient.
   */
  AdaBoundUpdate(const double finalLr = 0.1,
                 const double gamma = 1e-3,
                 const double epsilon = 1e-8,
                 const double beta1 = 0.9,
                 const double beta2 = 0.999) :
    finalLr(finalLr),
    gamma(gamma),
    epsilon(epsilon),
    beta1(beta1),
    beta2(beta2)
  {
    // Nothing to do.
  }

  //! Get the final (SGD) learning rate.
  double FinalLr() const { return finalLr; }
  //! Modify the final (SGD) learning rate.
  double& FinalLr() { return finalLr; }

  //! Get the convergence speed of the bound functions.
  double Gamma() const { return finalLr; }
  //! Modify the convergence speed of the bound functions.
  double& Gamma() { return finalLr; }

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
     * @param parent AdaBoundUpdate object.
     * @param rows Number of rows in the gradient matrix.
     * @param cols Number of columns in the gradient matrix.
     */
    Policy(AdaBoundUpdate& parent, const size_t rows, const size_t cols) :
        parent(parent), first(true), initialStepSize(0), iteration(0)
    {
      m.zeros(rows, cols);
      v.zeros(rows, cols);
    }

    /**
     * Update step for AdaBound.
     *
     * @param iterate Parameters that minimize the function.
     * @param stepSize Step size to be used for the given iteration.
     * @param gradient The gradient matrix.
     */
    void Update(MatType& iterate,
                const double stepSize,
                const GradType& gradient)
    {
      // Convenience typedefs.
      typedef typename MatType::elem_type ElemType;

      // Save the initial step size.
      if (first)
      {
        first = false;
        initialStepSize = stepSize;
      }

      // Increment the iteration counter variable.
      ++iteration;

      // Decay the first and second moment running average coefficient.
      m *= parent.beta1;
      m += (1 - parent.beta1) * gradient;

      v *= parent.beta2;
      v += (1 - parent.beta2) * (gradient % gradient);

      const ElemType biasCorrection1 = 1.0 - std::pow(parent.beta1, iteration);
      const ElemType biasCorrection2 = 1.0 - std::pow(parent.beta2, iteration);

      const ElemType fl = parent.finalLr * stepSize / initialStepSize;
      const ElemType lower = fl * (1.0 - 1.0 / (parent.gamma * iteration + 1));
      const ElemType upper = fl * (1.0 + 1.0 / (parent.gamma * iteration));

       // Applies bounds on actual learning rate.
      iterate -= arma::clamp((stepSize *
          std::sqrt(biasCorrection2) / biasCorrection1) / (arma::sqrt(v) +
          parent.epsilon), lower, upper) % m;
    }

   private:
    // Instantiated parent object.
    AdaBoundUpdate& parent;

    // The exponential moving average of gradient values.
    GradType m;

    // The exponential moving average of squared gradient values.
    GradType v;

    // Whether this is the first call of the Update method.
    bool first;

    // The initial (Adam) learning rate.
    double initialStepSize;

    // The number of iterations.
    size_t iteration;
  };

 private:
  // The final (SGD) learning rate.
  double finalLr;

  // The convergence speed of the bound functions.
  double gamma;

  // The epsilon value used to initialise the squared gradient parameter.
  double epsilon;

  // The smoothing parameter.
  double beta1;

  // The second moment coefficient.
  double beta2;
};

} // namespace ens

#endif
