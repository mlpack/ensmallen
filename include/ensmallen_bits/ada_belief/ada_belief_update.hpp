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
        iteration(0)
    {
      m.zeros(rows, cols);
      s.zeros(rows, cols);
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

      m *= parent.beta1;
      m += (1 - parent.beta1) * gradient;

      s *= parent.beta2;
      s += (1 - parent.beta2) * arma::pow(gradient - m, 2.0) + parent.epsilon;

      const double biasCorrection1 = 1.0 - std::pow(parent.beta1, iteration);
      const double biasCorrection2 = 1.0 - std::pow(parent.beta2, iteration);

      // And update the iterate.
      iterate -= ((m / biasCorrection1) * stepSize) / (arma::sqrt(s /
          biasCorrection2) + parent.epsilon);
    }

   private:
    //! Instantiated parent object.
    AdaBeliefUpdate& parent;

    //! The exponential moving average of gradient values.
    GradType m;

    // The exponential moving average of squared gradient values.
    GradType s;

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
