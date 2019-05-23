/**
 * @file weight_decay_momentum_update.hpp
 * @author Ajinkya Tejankar
 * @author Niteya Shah
 *
 * Momentum update with weight decay for Stochastic Gradient Descent
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_WEIGHT_DECAY_MOMENTUM_UPDATE_HPP
#define ENSMALLEN_WEIGHT_DECAY_MOMENTUM_UPDATE_HPP

namespace ens {

/**
 * De-Coupled Weight Decay Policy for SGD with Momentum (SGDW).
 *
 * This implments the decoupled weight decay policy in which the weight decay is
 * decoupled from the optimization steps w.r.t. to the loss function.
 * For SGD variants, this simplifies hyperparameter search since it decouples
 * the settings of weight decay and learning rate.
 *
 *
 * The Update Policy for SGDW is given below
 * \f[
 * v = mu*v + \alpha \nabla f_i(A)
 * A_{j + 1} = A_j - \nabla \lamdba A_j - v
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
 */
class DecoupledWeightDecayMomentumUpdate
{
 public:
  /**
   * Construct the momentum update policy with weight decay. Initialize the
   * momentum and weight decay to given values.
   *
   * @param momentum The momentum parameter
   * @param weightDecay The weight decay parameter
   */
  DecoupledWeightDecayMomentumUpdate(const double momentum = 0.5,
                                     const double weightDecay = 0.0005) :
    momentum(momentum),
    weightDecay(weightDecay)
  {
    // Nothing to do
  }

  /**
   * Initialize the velocity matrix to zero. It is of the same shape as
   * gradient matrix. (see ens::MomentumUpdate::Initialize for more details)
   *
   * @param rows Number of rows in the gradient matrix.
   * @param cols Number of columns in the gradient matrix.
   */
  void Initialize(const size_t rows, const size_t cols)
  {
    velocity = arma::zeros<arma::mat>(rows, cols);
  }

  /**
   * Update the given paramters. (see ens::MomentumUpdate::Update for more
   * details).
   *
   * @param iterate Parameters that minimize the function.
   * @param stepSize Step size to be used for the given iteration.
   * @param gradient The gradient matrix.
   */
  void Update(arma::mat& iterate,
              const double stepSize,
              const arma::mat& gradient)
  {
    velocity = momentum * velocity + stepSize * gradient;
    iterate -= velocity + stepSize * weightDecay * iterate;
  }

 private:
  // The momentum hyperparameter
  double momentum;
  // The velocity matrix
  arma::mat velocity;
  // The weight decay (lambda) parameter
  double weightDecay;
};

} // namespace ens

#endif // ENSMALLEN_WEIGHT_DECAY_MOMENTUM_UPDATE_HPP
