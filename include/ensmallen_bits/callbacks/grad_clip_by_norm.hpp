/**
 * @file grad_clip_by_norm.hpp
 * @author Marcus Edel
 *
 * Clip the gradients by multiplying the unit vector of the gradients with the
 * threshold.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CALLBACKS_GRAD_CLIP_BY_NORM_HPP
#define ENSMALLEN_CALLBACKS_GRAD_CLIP_BY_NORM_HPP

namespace ens {

/**
 * Clip the gradients by multiplying the unit vector of the gradients with the
 * threshold.
 */
class GradClipByNorm
{
 public:
  /**
   * Set up the gradient clip by norm callback class with the maximum clipping
   * value.
   *
   * @param maxNorm The maximum clipping value.
   */
  GradClipByNorm(const double maxNorm) : maxNorm(maxNorm)
  { /* Nothing to do here. */ }

  /**
   * Callback function called at any call to Gradient().
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param gradient Matrix that holds the gradient.
   */
  template<typename OptimizerType, typename FunctionType, typename MatType>
  bool Gradient(OptimizerType& /* optimizer */,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                MatType& gradient)
  {
    const double gradientNorm = arma::norm(gradient);
    if (gradientNorm > maxNorm)
      gradient = maxNorm * gradient / gradientNorm;
    return false;
  }

 private:
  //! The maximum clipping value for gradient clipping.
  const double maxNorm;
};

} // namespace ens

#endif
