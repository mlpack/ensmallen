/**
 * @file momentum_delta_bar_delta_update.hpp
 * @author Ranjodh Singh
 *
 * MomentumDeltaBarDelta update policy for Gradient Descent.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_MOMENTUM_DELTA_BAR_DELTA_UPDATE_HPP
#define ENSMALLEN_MOMENTUM_DELTA_BAR_DELTA_UPDATE_HPP

namespace ens {

/**
 * MomentumDeltaBarDelta update policy for Gradient Descent.
 *
 * A DeltaBarDelta variant that incorporates the following modifications:
 *  - In the original DeltaBarDelta, the momentum term (delta_bar) is used
 *    solely for sign comparison with the current gradient and does not
 *    participate in the parameter update. In this modified variant, the
 *    momentum term (velocity) is directly used to update the parameters.
 *  - Instead of adjusting the step size directly, each parameter maintains
 *    a gain value initialized to 1.0. Updates apply additive increases or
 *    multiplicative decreases to this gain. The effective step size for a
 *    parameter is the product of its initial step size and its current gain.
 *
 * Note: This variant originates from optimization of the t-SNE cost function.
 *
 * @code
 * @article{jacobs1988increased,
 *   title     = {Increased Rates of Convergence Through Learning Rate
 *                Adaptation},
 *   author    = {Jacobs, Robert A.},
 *   journal   = {Neural Networks},
 *   volume    = {1},
 *   number    = {4},
 *   pages     = {295--307},
 *   year      = {1988},
 *   publisher = {Pergamon}
 * }
 * @endcode
 */
class MomentumDeltaBarDeltaUpdate
{
 public:
  /**
   * Construct the MomentumDeltaBarDelta update policy with given parameters.
   *
   * @param kappa Additive increase constant for step size.
   * @param phi Multiplicative decrease factor for step size.
   * @param momentum The momentum decay hyperparameter.
   * @param minGain Minimum allowed gain (scaling factor) for any parameter
   *     (default: 1e-8).
   */
  MomentumDeltaBarDeltaUpdate(
      const double kappa = 0.2,
      const double phi = 0.8,
      const double momentum = 0.5,
      const double minGain = 1e-8) :
      kappa(kappa),
      phi(phi),
      momentum(momentum),
      minGain(minGain)
  {
    /* Do nothing. */
  }

  //! Access the kappa hyperparameter.
  double Kappa() const { return kappa; }
  //! Modify the kappa hyperparameter.
  double& Kappa() { return kappa; }

  //! Access the phi hyperparameter.
  double Phi() const { return phi; }
  //! Modify the phi hyperparameter.
  double& Phi() { return phi; }

  //! Access the momentum hyperparameter.
  double Momentum() const { return momentum; }
  //! Modify the momentum hyperparameter.
  double& Momentum() { return momentum; }

  //! Access the minGain hyperparameter.
  double MinGain() const { return minGain; }
  //! Modify the minGain hyperparameter.
  double& MinGain() { return minGain; }

  /**
   * The UpdatePolicyType policy classes must contain an internal 'Policy'
   * template class with two template arguments: MatType and GradType. This is
   * instantiated at the start of the optimization, and holds parameters
   * specific to an individual optimization.
   */
  template <typename MatType, typename GradType>
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
    Policy(
        const MomentumDeltaBarDeltaUpdate& parent,
        const size_t rows,
        const size_t cols) :
        parent(parent),
        kappa(ElemType(parent.kappa)),
        phi(ElemType(parent.phi)),
        momentum(ElemType(parent.momentum)),
        minGain(ElemType(parent.minGain))
    {
      gains.ones(rows, cols);
      velocity.zeros(rows, cols);
    }

    /**
     * Update step for Gradient Descent.
     *
     * @param iterate Parameters that minimize the function.
     * @param stepSize Step size to be used for the given iteration.
     * @param gradient The gradient matrix.
     */
    void Update(MatType& iterate,
                const double stepSize,
                const GradType& gradient)
    {
      gains += conv_to<MatType>::from(
          (sign(gradient) != sign(velocity)) * kappa -
          (sign(gradient) == sign(velocity)) * (1 - phi) % gains);
      gains.clamp(minGain, arma::Datum<ElemType>::inf);

      velocity = momentum * velocity - (ElemType(stepSize) * gains) % gradient;
      iterate += velocity;
    }

   private:
    //! The instantiated parent class.
    const MomentumDeltaBarDeltaUpdate& parent;

    //! The gains matrix.
    MatType gains;

    //! The velocity matrix.
    MatType velocity;

    // Parent parameters converted to the element type of the matrix.
    ElemType kappa;
    ElemType phi;
    ElemType momentum;
    ElemType minGain;
  };

 private:
  //! The kappa hyperparameter.
  double kappa;

  //! The phi hyperparameter.
  double phi;

  //! The momentum hyperparameter.
  double momentum;

  //! The minGain hyperparameter.
  double minGain;
};

} // namespace ens

#endif // ENSMALLEN_MOMENTUM_DELTA_BAR_DELTA_UPDATE_HPP
