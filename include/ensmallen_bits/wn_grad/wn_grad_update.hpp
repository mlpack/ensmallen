/**
 * @file wn_grad_update.hpp
 * @author Marcus Edel
 *
 * WNGrad update rule for the WNGrad method.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_WN_GRAD_WN_GRAD_UPDATE_HPP
#define ENSMALLEN_WN_GRAD_WN_GRAD_UPDATE_HPP

namespace ens {

/**
 * WNGrad is a general nonlinear update rule for the learning rate, so that
 * the learning rate can be initialized at a high value, and is subsequently
 * decreased according to gradient observations.
 *
 * For more information, see the following.
 *
 * @code
 * @article{Wu2018,
 *   author  = {{Wu}, X. and {Ward}, R. and {Bottou}, L.},
 *   title   = {WNGrad: Learn the Learning Rate in Gradient Descent},
 *   journal = {ArXiv e-prints},
 *   year    = {2018},
 *   url     = {https://arxiv.org/abs/1803.02865},
 * }
 * @endcode
 */
class WNGradUpdate
{
 public:
  /**
   * Construct the WNGrad update policy.
   */
  WNGradUpdate() : b(1.0)
  {
    // Nothing to do here.
  }

  //! Get the learning rate adjustment.
  double B() const { return b; }
  //! Modify the learning rate adjustment.
  double& B() { return b; }

  /**
   * The UpdatePolicyType policy classes must contain an internal 'Policy'
   * template class with two template arguments: MatType and GradType.  This is
   * instantiated at the start of the optimization.
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
    Policy(WNGradUpdate& parent,
           const size_t /* rows */,
           const size_t /* cols */) :
        parent(parent)
    {
      /* Nothing to do. */
    }

    /**
     * Update step for WNGrad.
     *
     * @param iterate Parameters that minimize the function.
     * @param stepSize Step size to be used for the given iteration.
     * @param gradient The gradient matrix.
     */
    void Update(MatType& iterate,
                const double stepSize,
                const GradType& gradient)
    {
      parent.b += std::pow(stepSize, 2.0) / parent.b *
          std::pow(arma::norm(gradient), 2);
      iterate -= stepSize * gradient / parent.b;
    }

   private:
    //! Reference to the instantiated parent object.
    WNGradUpdate& parent;
  };

 private:
  //! Learning rate adjustment.
  double b;
};

} // namespace ens

#endif
