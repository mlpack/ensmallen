/**
 * @file grad_clip_by_value.hpp
 * @author Marcus Edel
 *
 * Clips the gradient to a specified min and max.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CALLBACKS_GRAD_CLIP_BY_VALUE_HPP
#define ENSMALLEN_CALLBACKS_GRAD_CLIP_BY_VALUE_HPP

namespace ens {

/**
 * Clip the gradient to a specified min and max.
 */
class GradClipByValue
{
 public:
  /**
   * Set up the gradient clip by value callback class with the min and max
   * value.
   *
   * @param min The minimum value to clip to.
   * @param max The maximum value to clip to.
   */
  GradClipByValue(const double min, const double max) : lower(min), upper(max)
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
    gradient = arma::clamp(gradient, lower, upper);
    return false;
  }

 private:
  //! The minimum value to clip to.
  const double lower;

  //! The maximum value to clip to.
  const double upper;
};

} // namespace ens

#endif
