/**
 * @file delta_bar_delta_update.hpp
 * @author Ranjodh Singh
 *
 * Delta-Bar-Delta update for Gradient Descent.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_GRAIDENT_DESCENT_DELTA_BAR_DELTA_UPDATE_HPP
#define ENSMALLEN_GRAIDENT_DESCENT_DELTA_BAR_DELTA_UPDATE_HPP

namespace ens {

/**
 * DeltaBarDelta Update Policy for Gradient Descent
 *
 * A heuristic designed to accelerate convergence by
 * adapting the learning rate of each paramenter individually.
 *
 * According to the Delta-Bar-Delta update:
 *
 * - If the current derivative of a weight and the exponential moving average
 *   of its previous derivatives have the same sign, then the learning rate
 *   for that weight is incremented by a constant \f$\kappa\f$.
 *
 * - If the current derivative of a weight and the exponential moving average
 *   of its previous derivatives have the opposite signs, then the learning
 *   rate for that weight is decremented by a proportion \f$\phi\f$ of its
 *   current value.
 *
 * @note This implementation introduces a MinimumGain parameter to ensure
 *     that repeated proportional decrements do not reduce the learning rate
 *     all the way to zero. This is not present in the paper given below.
 *
 * @code
 * @article{jacobs1988increased,
 *   title   = {Increased Rates of Convergence Through Learning Rate
 * Adaptation}, author  = {Jacobs, Robert A.}, journal = {Neural Networks},
 *   volume  = {1},
 *   number  = {4},
 *   pages   = {295--307},
 *   year    = {1988},
 *   publisher = {Pergamon}
 * }
 */
class DeltaBarDeltaUpdate
{
 public:
  /**
   * Construct the DeltaBarDelta update policy with given parameters.
   *
   * @param kappa The kappa hyperparameter
   * @param phi The phi hyperparameter
   * @param momentum The momentup hyperparameter
   * @param minGain The minGain hyperparameter
   */
  DeltaBarDeltaUpdate(const double kappa = 0.2,
                      const double phi = 0.8,
                      const double momentum = 0.5,
                      const double minGain = 0.01)
      : kappa(kappa), phi(phi), momentum(momentum), minGain(minGain)
  {
    /* Do nothing. */
  }

  //! Access kappa.
  double Kappa() const { return kappa; }
  //! Modify kappa.
  double& Kappa() { return kappa; }

  //! Access phi.
  double Phi() const { return phi; }
  //! Modify phi.
  double& Phi() { return phi; }

  //! Access the momentum.
  double Momentum() const { return momentum; }
  //! Modify the momentum.
  double& Momentum() { return momentum; }

  //! Access the minGain
  double MinimumGain() const { return minGain; }
  //! Modify the minGain.
  double& MinimumGain() { return minGain; }

  /**
   * The UpdatePolicyType policy classes must contain an internal 'Policy'
   * template class with two template arguments: MatType and GradType.  This is
   * instantiated at the start of the optimization, and holds parameters
   * specific to an individual optimization.
   */
  template <typename MatType, typename GradType>
  class Policy
  {
   public:
    /**
     * This is called by the optimizer method before the start of the iteration
     * update process.
     *
     * @param parent Instantiated parent class.
     * @param rows Number of rows in the gradient matrix.
     * @param cols Number of columns in the gradient matrix.
     */
    Policy(const DeltaBarDeltaUpdate& parent,
           const size_t rows,
           const size_t cols)
        : parent(parent), velocity(rows, cols)
    {
      gains.ones(rows, cols);
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
      arma::umat increase = (velocity % gradient) < 0.0;
      arma::umat decrease = (velocity % gradient) > 0.0;
      gains.elem(arma::find(increase)) += parent.Kappa();
      gains.elem(arma::find(decrease)) *= parent.Phi();
      gains.elem(arma::find(gains < parent.MinimumGain()))
          .fill(parent.MinimumGain());

      velocity = parent.momentum * velocity - (stepSize * gains) % gradient;
      iterate += velocity;
    }

   private:
    //! The instantiated parent class.
    const DeltaBarDeltaUpdate& parent;

    //! The gains matrix.
    MatType gains;

    //! The velocity matrix.
    MatType velocity;
  };

 private:
  //! The kappa hyperparameter
  double kappa;

  //! The phi hyperparameter
  double phi;

  //! The momentum hyperparameter.
  double momentum;

  //! The minGain hyperparameter
  double minGain;
};

} // namespace ens

#endif // ENSMALLEN_GRAIDENT_DESCENT_DELTA_BAR_DELTA_UPDATE_HPP
