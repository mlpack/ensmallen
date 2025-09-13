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
    typedef typename MatType::elem_type ElemType;

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
        iteration(0),
        epsilon(ElemType(parent.epsilon)),
        beta1(ElemType(parent.beta1)),
        beta2(ElemType(parent.beta2)),
        sgdRate(ElemType(parent.sgdRate)),
        sgdLambda(ElemType(parent.sgdLambda))
    {
      m.zeros(rows, cols);
      v.zeros(rows, cols);

      sgdV.zeros(rows, cols);

      // Attempt to catch underflow.
      if (epsilon == ElemType(0) && parent.epsilon != 0.0)
        epsilon = 10 * std::numeric_limits<ElemType>::epsilon();
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
        v *= beta1;
        v += gradient;

        iterate -= (1 - beta1) * sgdRate * v;
        return;
      }

      m *= beta1;
      m += (1 - beta1) * gradient;

      v *= beta2;
      v += (1 - beta2) * (gradient % gradient);

      const ElemType biasCorrection1 = 1 - std::pow(beta1, ElemType(iteration));
      const ElemType biasCorrection2 = 1 - std::pow(beta2, ElemType(iteration));

      GradType delta = ElemType(stepSize) * m / biasCorrection1 /
          (sqrt(v / biasCorrection2) + epsilon);
      iterate -= delta;

      const ElemType deltaGradient = dot(delta, gradient);
      if (deltaGradient != ElemType(0))
      {
        const ElemType rate = dot(delta, delta) / deltaGradient;
        sgdLambda = beta2 * sgdLambda + (1 - beta2) * rate;
        sgdRate = sgdLambda / biasCorrection2;

        parent.sgdLambda = (double) sgdLambda;
        parent.sgdRate = (double) sgdRate;

        if (std::abs(sgdRate - rate) < epsilon && iteration > 1)
        {
          parent.phaseSGD = true;
          v.zeros();
        }
      }
    }

   private:
    // Reference to instantiated parent object.
    SWATSUpdate& parent;

    // The exponential moving average of gradient values.
    GradType m;

    // The exponential moving average of squared gradient values (Adam).
    GradType v;

    // The exponential moving average of squared gradient values (SGD).
    GradType sgdV;

    // The number of iterations.
    size_t iteration;

    // Parameters casted to the element type of the optimization.
    ElemType epsilon;
    ElemType beta1;
    ElemType beta2;
    ElemType sgdRate;
    ElemType sgdLambda;
  };

 private:
  // The epsilon value used to initialise the squared gradient parameter.
  double epsilon;

  // The smoothing parameter.
  double beta1;

  // The second moment coefficient.
  double beta2;

  // Whether to use the SGD or Adam update rule.
  bool phaseSGD;

  // SGD scaling parameter.
  double sgdRate;

  // SGD learning rate.
  double sgdLambda;
};

} // namespace ens

#endif
