/**
 * @file ftml.hpp
 * @author Marcus Edel
 *
 * Definition of Follow the Moving Leader (FTML).
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_FTML_FTML_HPP
#define ENSMALLEN_FTML_FTML_HPP

#include <ensmallen_bits/sgd/sgd.hpp>

#include "ftml_update.hpp"

namespace ens {

/**
 * Follow the Moving Leader (FTML) is an optimizer where recent samples are
 * weighted more heavily in each iteration, so FTML can adapt more quickly to
 * changes.
 *
 * For more information, see the following.
 *
 * @code
 * @inproceedings{Zheng2017,
 *   author    = {Shuai Zheng and James T. Kwok},
 *   title     = {Follow the Moving Leader in Deep Learning},
 *   year      = {2017}
 *   booktitle = {Proceedings of the 34th International Conference on Machine
 *                Learning},
 *   pages     = {4110--4119},
 *   series    = {Proceedings of Machine Learning Research},
 *   publisher = {PMLR},
 * }
 * @endcode
 *
 * For FTML to work, a DecomposableFunctionType template parameter is
 * required. This class must implement the following function:
 *
 *   size_t NumFunctions();
 *   double Evaluate(const arma::mat& coordinates,
 *                   const size_t i,
 *                   const size_t batchSize);
 *   void Gradient(const arma::mat& coordinates,
 *                 const size_t i,
 *                 arma::mat& gradient,
 *                 const size_t batchSize);
 *
 * NumFunctions() should return the number of functions (\f$n\f$), and in the
 * other two functions, the parameter i refers to which individual function (or
 * gradient) is being evaluated.  So, for the case of a data-dependent function,
 * such as NCA, NumFunctions() should return the number of points in the
 * dataset, and Evaluate(coordinates, 0) will evaluate the objective function on
 * the first point in the dataset (presumably, the dataset is held internally in
 * the DecomposableFunctionType).
 */
class FTML
{
 public:
  /**
   * Construct the FTML optimizer with the given function and parameters. The
   * defaults here are not necessarily good for the given problem, so it is
   * suggested that the values used be tailored to the task at hand.  The
   * maximum number of iterations refers to the maximum number of points that
   * are processed (i.e., one iteration equals one point; one iteration does not
   * equal one pass over the dataset).
   *
   * @param stepSize Step size for each iteration.
   * @param batchSize Number of points to process in a single step.
   * @param beta1 Exponential decay rate for the first moment estimates.
   * @param beta2 Exponential decay rate for the weighted infinity norm
            estimates.
   * @param eps Value used to initialise the mean squared gradient parameter.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *        limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param shuffle If true, the function order is shuffled; otherwise, each
   *        function is visited in linear order.
   */
  FTML(const double stepSize = 0.001,
       const size_t batchSize = 32,
       const double beta1 = 0.9,
       const double beta2 = 0.999,
       const double eps = 1e-8,
       const size_t maxIterations = 100000,
       const double tolerance = 1e-5,
       const bool shuffle = true);

  /**
   * Optimize the given function using FTML. The given starting point will
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

  //! Get the smoothing parameter.
  double Beta1() const { return optimizer.UpdatePolicy().Beta1(); }
  //! Modify the smoothing parameter.
  double& Beta1() { return optimizer.UpdatePolicy().Beta1(); }

  //! Get the second moment coefficient.
  double Beta2() const { return optimizer.UpdatePolicy().Beta2(); }
  //! Modify the second moment coefficient.
  double& Beta2() { return optimizer.UpdatePolicy().Beta2(); }

  //! Get the value used to initialise the mean squared gradient parameter.
  double Epsilon() const { return optimizer.UpdatePolicy().Epsilon(); }
  //! Modify the value used to initialise the mean squared gradient parameter.
  double& Epsilon() { return optimizer.UpdatePolicy().Epsilon(); }

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

 private:
  //! The Stochastic Gradient Descent object with the FTMLUpdate update policy.
  SGD<FTMLUpdate> optimizer;
};

} // namespace ens

// Include implementation.
#include "ftml_impl.hpp"

#endif
