/**
 * @file wn_grad.hpp
 * @author Marcus Edel
 *
 * WNGrad is a general nonlinear update rule for the learning rate.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_WN_GRAD_WN_GRAD_HPP
#define ENSMALLEN_WN_GRAD_WN_GRAD_HPP

#include <ensmallen_bits/sgd/sgd.hpp>
#include "wn_grad_update.hpp"

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
 *
 * WNGrad can optimize differentiable separable functions.  For more
 * details, see the documentation on function types included with this
 * distribution or on the ensmallen website.
 */
class WNGrad
{
 public:
  /**
   * Construct the WNGrad optimizer with the given function and parameters. The
   * defaults here are not necessarily good for the given problem, so it is
   * suggested that the values used be tailored to the task at hand.  The
   * maximum number of iterations refers to the maximum number of points that
   * are processed (i.e., one iteration equals one point; one iteration does not
   * equal one pass over the dataset).
   *
   * @param stepSize Step size for each iteration.
   * @param batchSize Number of points to process in a single step.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *        limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param shuffle If true, the function order is shuffled; otherwise, each
   *        function is visited in linear order.
   * @param resetPolicy If true, parameters are reset before every Optimize
   *        call; otherwise, their values are retained.
   */
  WNGrad(const double stepSize = 0.562,
         const size_t batchSize = 32,
         const size_t maxIterations = 100000,
         const double tolerance = 1e-5,
         const bool shuffle = true,
         const bool resetPolicy = true);

  /**
   * Optimize the given function using WNGrad. The given starting point will
   * be modified to store the finishing point of the algorithm, and the final
   * objective value is returned.
   *
   * @tparam DecomposableFunctionType Type of the function to be optimized.
   * @param function Function to optimize.
   * @param iterate Starting point (will be modified).
   * @return Objective value of the final point.
   */
  template<typename DecomposableFunctionType>
  double Optimize(DecomposableFunctionType& function, arma::mat& iterate)
  {
    return optimizer.Optimize(function, iterate);
  }

  //! Get the step size.
  double StepSize() const { return optimizer.StepSize(); }
  //! Modify the step size.
  double& StepSize() { return optimizer.StepSize(); }

  //! Get the batch size.
  size_t BatchSize() const { return optimizer.BatchSize(); }
  //! Modify the batch size.
  size_t& BatchSize() { return optimizer.BatchSize(); }

  //! Get the maximum number of iterations (0 indicates no limit).
  size_t MaxIterations() const { return optimizer.MaxIterations(); }
  //! Modify the maximum number of iterations (0 indicates no limit).
  size_t& MaxIterations() { return optimizer.MaxIterations(); }

  //! Get the tolerance for termination.
  double Tolerance() const { return optimizer.Tolerance(); }
  //! Modify the tolerance for termination.
  double& Tolerance() { return optimizer.Tolerance(); }

  //! Get whether or not the individual functions are shuffled.
  bool Shuffle() const { return optimizer.Shuffle(); }
  //! Modify whether or not the individual functions are shuffled.
  bool& Shuffle() { return optimizer.Shuffle(); }

  //! Get whether or not the update policy parameters
  //! are reset before Optimize call.
  bool ResetPolicy() const { return optimizer.ResetPolicy(); }
  //! Modify whether or not the update policy parameters
  //! are reset before Optimize call.
  bool& ResetPolicy() { return optimizer.ResetPolicy(); }

 private:
  //! The WNGrad update policy.
  SGD<WNGradUpdate> optimizer;
};

} // namespace ens

// Include implementation.
#include "wn_grad_impl.hpp"

#endif
