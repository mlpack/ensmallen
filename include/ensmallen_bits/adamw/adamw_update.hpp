/**
 * @file adamw_update.hpp
 * @author Niteya Shah
 *
 * AdamW optimizer. Adam is an an algorithm for first-order gradient-based
 * optimization of stochastic objective functions, based on adaptive estimates
 * of lower-order moments.
 *
 * AdamW is extension of Adam that decouples the weight Decay segment.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_ADAM_ADAMW_UPDATE_HPP
#define ENSMALLEN_ADAM_ADAMW_UPDATE_HPP

namespace ens {

/**
 * De-Coupled Weight Decay Policy for Adam (AdamW).
 *
 * This Implments Decoupled Weight Decay Policy in which the weight decay is
 * decoupled from the optimization steps w.r.t. to the loss function.
 *
 *
 * The Update Policy for AdamW follows Adam with the following additional step
 * \f[
 * iterate -= weightDecay * iterate
 * \f]
 *
 * The Update strategy is discussed in the following paper.
 *
 * @code
 * @article{
 *   title   = {Decoupled Weight Decay Regularization},
 *   author  = {Loschilov, I. and Hutter, F.},
 *   journal = {ArXiv e-prints},
 *   url     = {https://arxiv.org/pdf/1711.05101.pdf}
 *   year    = {2019}
 * }
 * @endcode
 *
 */
class AdamWUpdate
{
 public:
  /**
   * Construct the Adam update policy with the given parameters.
   *
   * @param epsilon The epsilon value used to initialise the squared gradient
   *        parameter.
   * @param beta1 The smoothing parameter.
   * @param beta2 The second moment coefficient.
   * @param weightDecay The rate at which the update regularises the iterate.
   */
  AdamWUpdate(const double epsilon = 1e-8,
              const double beta1 = 0.9,
              const double beta2 = 0.999,
              const double weightDecay = 0.0005) :
    update(epsilon, beta1, beta2),
    weightDecay(weightDecay),
    iteration(0)
  {/* Nothing to do.*/ }

  /**
   * The Initialize method is called by SGD Optimizer method before the start of
   * the iteration update process.
   *
   * @param rows Number of rows in the gradient matrix.
   * @param cols Number of columns in the gradient matrix.
   */
  void Initialize(const size_t rows, const size_t cols)
  {
    update.Initialize(rows, cols);
  }

  /**
   * Update step for AdamW.
   *
   * @param iterate Parameters that minimize the function.
   * @param stepSize Step size to be used for the given iteration.
   * @param gradient The gradient matrix.
   */
  void Update(arma::mat& iterate,
              const double stepSize,
              const arma::mat& gradient)
  {
    update.Update(iterate, stepSize, gradient);
    iterate -= weightDecay * iterate;
  }

  //! Get the value used to initialise the squared gradient parameter.
  double Epsilon() const { return update.Epsilon(); }
  //! Modify the value used to initialise the squared gradient parameter.
  double& Epsilon() { return update.Epsilon(); }

  //! Get the smoothing parameter.
  double Beta1() const { return update.Beta1(); }
  //! Modify the smoothing parameter.
  double& Beta1() { return update.Beta1(); }

  //! Get the second moment coefficient.
  double Beta2() const { return update.Beta2(); }
  //! Modify the second moment coefficient.
  double& Beta2() { return update.Beta2(); }

  //! Get weight decay parameter.
  double WeightDecay() const { return weightDecay; }
  //! Modify weight decay parameter.
  double& WeightDecay() { return weightDecay; }

 private:
  // The number of iterations.
  double iteration;
  // The weight decay rate.
  double weightDecay;
  // The AdamWUpdate optimser.
  AdamUpdate update;
};

} // namespace ens

#endif
