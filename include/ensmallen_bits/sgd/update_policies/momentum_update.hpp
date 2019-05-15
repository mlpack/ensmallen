/**
 * @file momentum_update.hpp
 * @author Arun Reddy
 *
 * Momentum update for Stochastic Gradient Descent.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SGD_MOMENTUM_UPDATE_HPP
#define ENSMALLEN_SGD_MOMENTUM_UPDATE_HPP

namespace ens {

/**
 * Momentum update policy for Stochastic Gradient Descent (SGD).
 *
 * Learning with SGD can sometimes be slow.  Using momentum update for parameter
 * learning can accelerate the rate of convergence, specifically in the cases
 * where the surface curves much more steeply(a steep hilly terrain with high
 * curvature).  The momentum algorithm introduces a new velocity vector \f$ v
 * \f$ with the same dimension as the parameter \f$ A \f$.  Also it introduces a
 * new decay hyperparameter momentum \f$ mu \in (0,1] \f$.  The following update
 * scheme is used to update SGD in every iteration:
 *
 * \f[
 * v = mu*v - \alpha \nabla f_i(A)
 * A_{j + 1} = A_j + v
 * \f]
 *
 * where \f$ \alpha \f$ is a parameter which specifies the step size.  \f$ i \f$
 * is chosen according to \f$ j \f$ (the iteration number).  Common values of
 * \f$ mu \f$ include 0.5, 0.9 and 0.99.  Typically it begins with a small value
 * and later raised.
 *
 * For more information, please refer to the Section 8.3.2 of the following book
 *
 * @code
 * @article{rumelhart1988learning,
 *   title   = {Learning representations by back-propagating errors},
 *   author  = {Rumelhart, David E. and Hinton, Geoffrey E. and
 *              Williams, Ronald J.},
 *   journal = {Cognitive Modeling},
 *   volume  = {5},
 *   number  = {3},
 *   pages   = {1},
 *   year    = {1988}
 * }
 *
 * @code
 * @book{Goodfellow-et-al-2016,
 *  title     = {Deep Learning},
 *  author    = {Ian Goodfellow and Yoshua Bengio and Aaron Courville},
 *  publisher = {MIT Press},
 *  note      = {\url{http://www.deeplearningbook.org}},
 *  year      = {2016}
 * }
 */
class MomentumUpdate
{
 public:
  /**
   * Construct the momentum update policy with given momentum decay parameter.
   *
   * @param momentum The momentum decay hyperparameter
   */
  MomentumUpdate(const double momentum = 0.5) : momentum(momentum)
  { /* Do nothing. */ };

  //! Access the momentum.
  double Momentum() const { return momentum; }
  //! Modify the momentum.
  double& Momentum() { return momentum; }

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
     * This is called by the optimizer method before the start of the iteration
     * update process.
     *
     * @param parent Instantiated parent class.
     * @param rows Number of rows in the gradient matrix.
     * @param cols Number of columns in the gradient matrix.
     */
    Policy(const MomentumUpdate& parent, const size_t rows, const size_t cols) :
        parent(parent),
        velocity(arma::zeros<MatType>(rows, cols))
    {
      // Nothing to do.
    }

    /**
     * Update step for SGD.  The momentum term makes the convergence faster on
     * the way as momentum term increases for dimensions pointing in the same
     * and reduces updates for dimensions whose gradients change directions.
     *
     * @param iterate Parameters that minimize the function.
     * @param stepSize Step size to be used for the given iteration.
     * @param gradient The gradient matrix.
     */
    void Update(MatType& iterate,
                const double stepSize,
                const GradType& gradient)
    {
      velocity = parent.momentum * velocity - stepSize * gradient;
      iterate += velocity;
    }

   private:
    // The instantiated parent class.
    const MomentumUpdate& parent;
    // The velocity matrix.
    MatType velocity;
  };

 private:
  // The momentum hyperparameter.
  double momentum;
};

} // namespace ens

#endif
