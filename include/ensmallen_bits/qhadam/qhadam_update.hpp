/**
 * @file qhadam_update.hpp
 * @author Niteya Shah
 *
 * Implments the QHAdam Optimizer. QHAdam is a variant of Adam which introduces
 * quasi hyperbolic moment terms to improve paramterisation and performance.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_ADAM_QHADAM_UPDATE_HPP
#define ENSMALLEN_ADAM_QHADAM_UPDATE_HPP

namespace ens {

/**
 * QHAdam is a optimising strategy based on the Quasi-Hyperbolic step when
 * applied to the Adam Optimiser.QH updates can be considered to a weighted
 * average of the momentum.QHAdam,based on its paramterisation can recover
 * many algorithms such as NAdam and RMSProp.
 *
 * For more information, see the following.
 *
 * @code
 * @inproceedings{ma2019qh,
 *   title={Quasi-hyperbolic momentum and Adam for deep learning},
 *   author={Jerry Ma and Denis Yarats},
 *   booktitle={International Conference on Learning Representations},
 *   year={2019}
 * }
 * @endcode
 */
class QHAdamUpdate
{
 public:
  /**
   * Construct the QHAdam update policy with the given parameters.
   *
   * @param epsilon The epsilon value used to initialise the squared gradient
   *        parameter.
   * @param beta1 The smoothing parameter.
   * @param beta2 The second moment coefficient.
   * @param v1 The first quasi-hyperbolic term.
   * @param v1 The second quasi-hyperbolic term.
   */
  QHAdamUpdate(const double epsilon = 1e-8,
               const double beta1 = 0.9,
               const double beta2 = 0.999,
               const double v1 = 0.7,
               const double v2 = 1) :
    epsilon(epsilon),
    beta1(beta1),
    beta2(beta2),
    v1(v1),
    v2(v2),
    iteration(0)
  {
    // Nothing to do.
  }

  /**
   * The Initialize method is called by SGD Optimizer method before the start of
   * the iteration update process.
   *
   * @param rows Number of rows in the gradient matrix.
   * @param cols Number of columns in the gradient matrix.
   */
  void Initialize(const size_t rows, const size_t cols)
  {
    m = arma::zeros<arma::mat>(rows, cols);
    v = arma::zeros<arma::mat>(rows, cols);
  }

  /**
   * Update step for QHAdam.
   *
   * @param iterate Parameters that minimize the function.
   * @param stepSize Step size to be used for the given iteration.
   * @param gradient The gradient matrix.
   */
  void Update(arma::mat& iterate,
              const double stepSize,
              const arma::mat& gradient)
  {
    // Increment the iteration counter variable.
    ++iteration;

    // And update the iterate.
    m *= beta1;
    m += (1 - beta1) * gradient;

    v *= beta2;
    v += (1 - beta2) * (gradient % gradient);

    const double biasCorrection1 = 1.0 - std::pow(beta1, iteration);
    const double biasCorrection2 = 1.0 - std::pow(beta2, iteration);

    arma::mat mDash = m / biasCorrection1;
    arma::mat vDash = v / biasCorrection2;

    // QHAdam recovers Adam when v2 = v1 = 1.
    iterate -= stepSize * ((((1 - v1) * gradient) + v1 * mDash) /
               (arma::sqrt(((1 - v2) * (gradient % gradient)) +
               v2 * vDash) + epsilon));
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

  //! Get the first quasi-hyperbolic term.
  double V1() const { return v1; }
  //! Modify the first quasi-hyperbolic term.
  double& V1() { return v1; }

  //! Get the second quasi-hyperbolic term.
  double V2() const { return v2; }
  //! Modify the second quasi-hyperbolic term.
  double& V2() { return v2; }

 private:
  // The epsilon value used to initialise the squared gradient parameter.
  double epsilon;

  // The smoothing parameter.
  double beta1;

  // The second moment coefficient.
  double beta2;

  // The exponential moving average of gradient values.
  arma::mat m;

  // The exponential moving average of squared gradient values.
  arma::mat v;

  // The number of iterations.
  double iteration;

  // The first quasi-hyperbolic term.
  double v1;

  // The second quasi-hyperbolic term.
  double v2;
};

} // namespace ens

#endif
