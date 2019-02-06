/**
 * @file weight_decay_momentum_update.hpp
 * @author Ajinkya Tejankar
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
 * Momentum update for Stochastic Gradient Descent with weight decay.
 *
 * TODO: add documentation for SGD momentum as well?
 *
 * TODO: is the following equation right?
 *
 * \f[
 * v = mu*v - \alpha \nabla f_i(A)
 * A_{j + 1} = A_j - \nabla \lamdba A_j - v
 * \f]
 *
 * @code
 * @article{
 *   title   = {Decoupled Weight Decay Regularization},
 *   author  = {{Ilya}, L. and {Frank}, H.},
 *   journal = {ArXiv e-prints},
 *   url     = {https://arxiv.org/pdf/1711.05101.pdf}
 *   year    = {2019}
 * }
 * @endcode
 */
 /**
  * TODO: better name?
  */
class WeightDecayMomentumUpdate
{
 public:
  /**
   * Construct the momentum update policy with weight decay. Initialize the
   * momentum and weight decay to given values.
   *
   * @param momentum The momentum parameter
   * @param weight_decay The weight decay parameter
   */
  WeightDecayMomentumUpdate(const double momentum = 0.5,
                            const double weight_decay = 0.0005) :
      momentum(momentum),
      weight_decay(weight_decay)
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
    velocity = arma::zeros<arma::mat>(rows, cols)
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
    velocity = momentum * velocity - stepSize * gradient;
    iterate -= stepSize * weight_decay * iterate + velocity;
  }

 private:
  // The momentum hyperparameter
  double momentum;
  // The velocity matrix
  arma::mat velocity;
  // The weight decay (lambda) parameter
  double weight_decay;
};

} // namespace ens

#endif //ENSMALLEN_WEIGHT_DECAY_MOMENTUM_UPDATE_HPP
