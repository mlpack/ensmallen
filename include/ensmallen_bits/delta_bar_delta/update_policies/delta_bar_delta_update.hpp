/**
 * @file delta_bar_delta_update.hpp
 * @author Ranjodh Singh
 *
 * DeltaBarDelta update policy for Gradient Descent.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_DELTA_BAR_DELTA_UPDATE_HPP
#define ENSMALLEN_DELTA_BAR_DELTA_UPDATE_HPP

namespace ens {

/**
 * DeltaBarDelta update policy for Gradient Descent.
 *
 * A heuristic designed to accelerate convergence by
 * adapting the learning rate of each parameter individually.
 *
 * According to the Delta-Bar-Delta update:
 *
 * - If the current gradient and the exponential average of
 *   past gradients corresponding to a parameter have the same
 *   sign, then the step size for that parameter is incremented by
 *   \f$\kappa\f$. Otherwise, it is decreased by a proportion \f$\phi\f$
 *   of its current value (additive increase, multiplicative decrease).
 *
 * @note This implementation uses a minStepSize parameter to set a lower
 *     bound for the learning rate. This prevents the learning rate from
 *     dropping to zero, which can occur due to floating-point underflow.
 *     For tasks which require extreme fine-tuning, you may need to lower
 *     this parameter below its default value (1e-8) in order to allow for
 *     smaller learning rates.
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
class DeltaBarDeltaUpdate
{
 public:
  /**
   * Construct the DeltaBarDelta update policy with given parameters.
   *
   * @param initialStepSize Initial Step Size.
   * @param kappa Additive increase constant for step size.
   * @param phi Multiplicative decrease factor for step size.
   * @param theta Decay rate for the exponential moving average.
   * @param minStepSize Minimum allowed step size for any parameter
   *     (default: 1e-8).
   */
  DeltaBarDeltaUpdate(
      const double initialStepSize,
      const double kappa,
      const double phi,
      const double theta,
      const double minStepSize = 1e-8) :
      initialStepSize(initialStepSize),
      kappa(kappa),
      phi(phi),
      theta(theta),
      minStepSize(minStepSize)
  {
    /* Do nothing. */
  }

  //! Access the initialStepSize hyperparameter.
  double InitialStepSize() const { return initialStepSize; }
  //! Modify the initialStepSize hyperparameter.
  double& InitialStepSize() { return initialStepSize; }

  //! Access the kappa hyperparameter.
  double Kappa() const { return kappa; }
  //! Modify the kappa hyperparameter.
  double& Kappa() { return kappa; }

  //! Access the phi hyperparameter.
  double Phi() const { return phi; }
  //! Modify the phi hyperparameter.
  double& Phi() { return phi; }

  //! Access the theta hyperparameter.
  double Theta() const { return theta; }
  //! Modify the theta hyperparameter.
  double& Theta() { return theta; }

  //! Access the minStepSize hyperparameter.
  double MinStepSize() const { return minStepSize; }
  //! Modify the minStepSize hyperparameter.
  double& MinStepSize() { return minStepSize; }

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
        const DeltaBarDeltaUpdate& parent,
        const size_t rows,
        const size_t cols) :
        parent(parent),
        kappa(ElemType(parent.kappa)),
        phi(ElemType(parent.phi)),
        theta(ElemType(parent.theta)),
        minStepSize(ElemType(parent.minStepSize))
    {
      deltaBar.zeros(rows, cols);
      epsilon.set_size(rows, cols);
      epsilon.fill(ElemType(parent.InitialStepSize()));
    }

    /**
     * Update step for Gradient Descent.
     *
     * @param iterate Parameters that minimize the function.
     * @param stepSize Step size to be used for the given iteration.
     * @param delta The gradient matrix.
     */
    void Update(MatType& iterate,
                const double /* stepSize */,
                const GradType& delta)
    {
      const MatType signMatrix = sign(delta % deltaBar);

      epsilon += conv_to<MatType>::from((signMatrix == +1) * kappa -
          (signMatrix == -1) * phi % epsilon);
      epsilon.clamp(minStepSize, arma::Datum<ElemType>::inf);

      deltaBar = theta * deltaBar + (1 - theta) * delta;
      iterate -= epsilon % delta;
    }

   private:
    //! The instantiated parent class.
    const DeltaBarDeltaUpdate& parent;

    //! The exponential average of past gradients.
    MatType deltaBar;
    
    //! Tracks the current step size for each parameter.
    MatType epsilon;

    // Parent parameters converted to the element type of the matrix.
    ElemType kappa;
    ElemType phi;
    ElemType theta;
    ElemType minStepSize;
  };

 private:
  //! The initialStepSize hyperparameter.
  double initialStepSize;

  //! The kappa hyperparameter.
  double kappa;

  //! The phi hyperparameter.
  double phi;

  //! The theta hyperparameter.
  double theta;

  //! The minStepSize hyperparameter.
  double minStepSize;
};

} // namespace ens

#endif // ENSMALLEN_DELTA_BAR_DELTA_UPDATE_HPP
