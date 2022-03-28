/**
 * @file swats_update.hpp
 * @author Marcus Edel
 *
 * SWATS update rule for Switches from Adam to SGD method.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SWATS_SWATS_UPDATE_HPP
#define ENSMALLEN_SWATS_SWATS_UPDATE_HPP

namespace ens {

/**
 * SWATS is a simple strategy which switches from Adam to SGD when a triggering
 * condition is satisfied. The condition relates to the projection of Adam steps
 * on the gradient subspace.
 *
 * For more information, see the following.
 *
 * @code
 * @article{Keskar2017,
 *   author  = {Nitish Shirish Keskar and Richard Socher},
 *   title   = {Improving Generalization Performance by Switching from Adam to
 *              {SGD}},
 *   journal = {CoRR},
 *   volume  = {abs/1712.07628},
 *   year    = {2017}
 *   url     = {http://arxiv.org/abs/1712.07628},
 * }
 * @endcode
 */
class SWATSUpdate
{
 public:
  /**
   * Construct the SWATS update policy with given parameter.
   *
   * @param epsilon The epsilon value used to initialise the squared gradient
   *        parameter.
   * @param beta1 The smoothing parameter.
   * @param beta2 The second moment coefficient.
   */
  SWATSUpdate(const double epsilon = 1e-8,
              const double beta1 = 0.9,
              const double beta2 = 0.999) :
    epsilon(epsilon),
    beta1(beta1),
    beta2(beta2),
    phaseSGD(false),
    sgdRate(0),
    sgdLambda(0)
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

  //! Get whether the current phase is SGD.
  bool PhaseSGD() const { return phaseSGD; }
  //! Modify whether the current phase is SGD.
  bool& PhaseSGD() { return phaseSGD; }

  //! Get the SGD scaling parameter.
  double SGDRate() const { return sgdRate; }
  //! Modify the SGD scaling parameter.
  double& SGDRate() { return sgdRate; }

  //! Get the SGD step size.
  double SGDLambda() const { return sgdLambda; }
  //! Modify the SGD step size.
  double& SGDLambda() { return sgdLambda; }

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
    Policy(SWATSUpdate& parent, const size_t rows, const size_t cols) :
        parent(parent),
        iteration(0)
    {
      m.zeros(rows, cols);
      v.zeros(rows, cols);

      sgdV.zeros(rows, cols);
    }

    /**
     * Update step for SWATS.
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

      if (parent.phaseSGD)
      {
        // Note we reuse the exponential moving average parameter here instead
        // of introducing a new parameter (sgdV) as done in the paper.
        v *= parent.beta1;
        v += gradient;

        iterate -= (1 - parent.beta1) * parent.sgdRate * v;
        return;
      }

      m *= parent.beta1;
      m += (1 - parent.beta1) * gradient;

      v *= parent.beta2;
      v += (1 - parent.beta2) * (gradient % gradient);

      const double biasCorrection1 = 1.0 - std::pow(parent.beta1, iteration);
      const double biasCorrection2 = 1.0 - std::pow(parent.beta2, iteration);

      GradType delta = stepSize * m / biasCorrection1 /
          (arma::sqrt(v / biasCorrection2) + parent.epsilon);
      iterate -= delta;

      const double deltaGradient = arma::dot(delta, gradient);
      if (deltaGradient != 0)
      {
        const double rate = arma::dot(delta, delta) / deltaGradient;
        parent.sgdLambda = parent.beta2 * parent.sgdLambda +
            (1 - parent.beta2) * rate;
        parent.sgdRate = parent.sgdLambda / biasCorrection2;

        if (std::abs(parent.sgdRate - rate) < parent.epsilon && iteration > 1)
        {
          parent.phaseSGD = true;
          v.zeros();
        }
      }
    }

   private:
    //! Reference to instantiated parent object.
    SWATSUpdate& parent;

    //! The exponential moving average of gradient values.
    GradType m;

    //! The exponential moving average of squared gradient values (Adam).
    GradType v;

    //! The exponential moving average of squared gradient values (SGD).
    GradType sgdV;

    //! The number of iterations.
    size_t iteration;
  };

 private:
  //! The epsilon value used to initialise the squared gradient parameter.
  double epsilon;

  //! The smoothing parameter.
  double beta1;

  //! The second moment coefficient.
  double beta2;

  //! Wether to use the SGD or Adam update rule.
  bool phaseSGD;

  //! SGD scaling parameter.
  double sgdRate;

  //! SGD learning rate.
  double sgdLambda;
};

} // namespace ens

#endif
