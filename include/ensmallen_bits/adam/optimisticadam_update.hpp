/**
 * @file optimisticadam_update.hpp
 * @author Moksh Jain
 *
 * OptmisticAdam optimizer. Implements Optimistic Adam, an algorithm which
 * uses Optimistic Mirror Descent with the Adam optimizer.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_ADAM_OPTIMISTICADAM_UPDATE_HPP
#define ENSMALLEN_ADAM_OPTIMISTICADAM_UPDATE_HPP

namespace ens {

/**
 * OptimisticAdam is an optimizer which implements the Optimistic Adam
 * algorithm which uses Optmistic Mirror Descent with the Adam Optimizer.
 * It addresses the problem of limit cycling while training GANs. It uses
 * OMD to achieve faster regret rates in solving the zero sum game of
 * training a GAN. It consistently achieves a smaller KL divergnce with
 * respect to the true underlying data distribution.
 *
 * For more information, see the following.
 *
 * @code
 * @article{
 *   author = {Constantinos Daskalakis, Andrew Ilyas, Vasilis Syrgkanis,
 *             Haoyang Zeng},
 *   title  = {Training GANs with Optimism},
 *   year   = {2017},
 *   url    = {https://arxiv.org/abs/1711.00141}
 * }
 * @endcode
 */
class OptimisticAdamUpdate
{
 public:
  /**
   * Construct the OptimisticAdam update policy with the given parameters.
   *
   * @param epsilon The epsilon value used to initialize the squared gradient
   *        parameter.
   * @param beta1 The smoothing parameter.
   * @param beta2 The second moment coefficient.
   */
  OptimisticAdamUpdate(const double epsilon = 1e-8,
                       const double beta1 = 0.9,
                       const double beta2 = 0.999) :
    epsilon(epsilon),
    beta1(beta1),
    beta2(beta2)
  {
    // Nothing to do.
  }

  //! Get the value used to initialize the squared gradient parameter.
  double Epsilon() const { return epsilon; }
  //! Modify the value used to initialize the squared gradient parameter.
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
     * @param parent Instantiated OptimisticAdamUpdate parent object.
     * @param rows Number of rows in the gradient matrix.
     * @param cols Number of columns in the gradient matrix.
     */
    Policy(OptimisticAdamUpdate& parent, const size_t rows, const size_t cols) :
        parent(parent),
        epsilon(ElemType(parent.epsilon)),
        beta1(ElemType(parent.beta1)),
        beta2(ElemType(parent.beta2)),
        iteration(0)
    {
      m.zeros(rows, cols);
      v.zeros(rows, cols);
      g.zeros(rows, cols);

      // Attempt to detect underflow.
      if (epsilon == ElemType(0) && parent.epsilon != 0.0)
        epsilon = 100 * std::numeric_limits<ElemType>::epsilon();
    }

    /**
     * Update step for OptimisticAdam.
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
      v += (1 - beta2) * square(gradient);

      GradType mCorrected = m / (1 - std::pow(beta1, ElemType(iteration)));
      GradType vCorrected = v / (1 - std::pow(beta2, ElemType(iteration)));

      GradType update = mCorrected / (sqrt(vCorrected) + epsilon);

      iterate -= (2 * ElemType(stepSize) * update - ElemType(stepSize) * g);

      g = std::move(update);
    }

   private:
    // Instantiated parent object.
    OptimisticAdamUpdate& parent;

    // The exponential moving average of gradient values.
    GradType m;

    // The exponential moving average of squared gradient values.
    GradType v;

    // The previous update.
    GradType g;

    // Parameters converted to the element type of the optimization.
    ElemType epsilon;
    ElemType beta1;
    ElemType beta2;

    // The number of iterations.
    size_t iteration;
  };

 private:
  // The epsilon value used to initialize the squared gradient parameter.
  double epsilon;

  // The smoothing parameter.
  double beta1;

  // The second moment coefficient.
  double beta2;
};

} // namespace ens

#endif
