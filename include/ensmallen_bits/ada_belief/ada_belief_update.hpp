/**
 * @file ada_belief_update.hpp
 * @author Marcus Edel
 *
 * AdaBelief optimizer update policy. The intuition for AdaBelief is to adapt
 * the stepsize according to the "belief" in the current gradient direction.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_ADA_BELIEF_ADA_BELIEF_UPDATE_HPP
#define ENSMALLEN_ADA_BELIEF_ADA_BELIEF_UPDATE_HPP

namespace ens {

/**
 * The intuition for AdaBelief is to adapt the stepsize according to the
 * "belief" in the current gradient direction.
 *
 * For more information, see the following.
 *
 * @code
 * @misc{zhuang2020adabelief,
 *   title         = {AdaBelief Optimizer: Adapting Stepsizes by the Belief in
 *                    Observed Gradients},
 *   author        = {Juntang Zhuang and Tommy Tang and Sekhar Tatikonda and
 *                    Nicha Dvornek and Yifan Ding and Xenophon Papademetris
 *                    and James S. Duncan},
 *   year          = {2020},
 *   eprint        = {2010.07468},
 *   archivePrefix = {arXiv},
 * }
 * @endcode
 */
class AdaBeliefUpdate
{
 public:
  /**
   * Construct the AdaBelief update policy with the given parameters.
   *
   * @param epsilon A small constant for numerical stability.
   * @param beta1 The exponential decay rate for the 1st moment estimates.
   * @param beta2 The exponential decay rate for the 2nd moment estimates.
   */
  AdaBeliefUpdate(const double epsilon = 1e-8,
                  const double beta1 = 0.9,
                  const double beta2 = 0.999) :
    epsilon(epsilon),
    beta1(beta1),
    beta2(beta2)
  {
    // Nothing to do.
  }

  //! Get the value for numerical stability.
  double Epsilon() const { return epsilon; }
  //! Modify the value used for numerical stability.
  double& Epsilon() { return epsilon; }

  //! Get the exponential decay rate for the 1st moment estimates.
  double Beta1() const { return beta1; }
  //! Modify the exponential decay rate for the 1st moment estimates.
  double& Beta1() { return beta1; }

  //! Get the exponential decay rate for the 2nd moment estimates.
  double Beta2() const { return beta2; }
  //! Modify the exponential decay rate for the 2nd moment estimates.
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
     * @param parent AdaBeliefUpdate object.
     * @param rows Number of rows in the gradient matrix.
     * @param cols Number of columns in the gradient matrix.
     */
    Policy(AdaBeliefUpdate& parent, const size_t rows, const size_t cols) :
        parent(parent),
        beta1(ElemType(parent.beta1)),
        beta2(ElemType(parent.beta2)),
        epsilon(ElemType(parent.epsilon)),
        iteration(0)
    {
      m.zeros(rows, cols);
      s.zeros(rows, cols);
      // Prevent underflow.
      if (epsilon == ElemType(0) && parent.epsilon != 0.0)
        epsilon = 100 * std::numeric_limits<ElemType>::epsilon();
    }

    /**
     * Update step for AdaBelief.
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

      m *= beta1;
      m += (1 - beta1) * gradient;

      s *= beta2;
      s += (1 - beta2) * pow(gradient - m, 2) + epsilon;

      const ElemType biasCorrection1 = 1 - std::pow(beta1, ElemType(iteration));
      const ElemType biasCorrection2 = 1 - std::pow(beta2, ElemType(iteration));

      // And update the iterate.
      iterate -= ((m / biasCorrection1) * ElemType(stepSize)) /
          (sqrt(s / biasCorrection2) + epsilon);
    }

   private:
    //! Instantiated parent object.
    AdaBeliefUpdate& parent;

    //! The exponential moving average of gradient values.
    GradType m;

    // The exponential moving average of squared gradient values.
    GradType s;

    // Parent parameters converted to the element type of the matrix.
    ElemType beta1;
    ElemType beta2;
    ElemType epsilon;

    // The number of iterations.
    size_t iteration;
  };

 private:
  // The epsilon value used to initialise the squared gradient parameter.
  double epsilon;

  // The xponential decay rate for the 1st moment estimates.
  double beta1;

  // The exponential decay rate for the 2nd moment estimates.
  double beta2;
};

} // namespace ens

#endif
