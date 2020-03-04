/**
 * @file sgd.hpp
 * @author Ryan Curtin
 * @author Arun Reddy
 * @author Abhinav Moudgil
 * @author Sourabh Varshney
 *
 * Stochastic Gradient Descent (SGD).
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SGD_SGD_HPP
#define ENSMALLEN_SGD_SGD_HPP

#include "update_policies/vanilla_update.hpp"
#include "update_policies/momentum_update.hpp"
#include "update_policies/nesterov_momentum_update.hpp"
#include "decay_policies/no_decay.hpp"
#include "update_policies/quasi_hyperbolic_update.hpp"

namespace ens {

/**
 * Stochastic Gradient Descent is a technique for minimizing a function which
 * can be expressed as a sum of other functions.  That is, suppose we have
 *
 * \f[
 * f(A) = \sum_{i = 0}^{n} f_i(A)
 * \f]
 *
 * and our task is to minimize \f$ A \f$.  Stochastic gradient descent iterates
 * over each function \f$ f_i(A) \f$, based on the specified update policy. By
 * default vanilla update policy (see ens::VanillaUpdate) is used. The SGD class
 * supports either scanning through each of the \f$ n \f$ functions \f$
 * f_i(A)\f$ linearly, or in a random sequence.  The algorithm continues until
 * \f$ j\f$ reaches the maximum number of iterations---or when a full sequence
 * of updates through each of the \f$ n \f$ functions \f$ f_i(A) \f$ produces an
 * improvement within a certain tolerance \f$ \epsilon \f$.  That is,
 *
 * \f[
 * | f(A_{j + n}) - f(A_j) | < \epsilon.
 * \f]
 *
 * The parameter \f$\epsilon\f$ is specified by the tolerance parameter to the
 * constructor; \f$n\f$ is specified by the maxIterations parameter.
 *
 * This class is useful for data-dependent functions whose objective function
 * can be expressed as a sum of objective functions operating on an individual
 * point.  Then, SGD considers the gradient of the objective function operating
 * on an individual point in its update of \f$ A \f$.
 *
 * SGD can optimize differentiable separable functions.  For more details, see
 * the documentation on function types included with this distribution or on the
 * ensmallen website.
 *
 * @tparam UpdatePolicyType Update policy used by SGD during the iterative
 *     update process. By default vanilla update policy (see ens::VanillaUpdate)
 *     is used.
 * @tparam DecayPolicyType Decay policy used during the iterative update
 *     process to adjust the step size. By default the step size isn't going to
 *     be adjusted (i.e. NoDecay is used).
 */
template<typename UpdatePolicyType = VanillaUpdate,
         typename DecayPolicyType = NoDecay>
class SGD
{
 public:
  /**
   * Construct the SGD optimizer with the given function and parameters. The
   * defaults here are not necessarily good for the given problem, so it is
   * suggested that the values used be tailored to the task at hand.  The
   * maximum number of iterations refers to the maximum number of points that
   * are processed (i.e., one iteration equals one point; one iteration does not
   * equal one pass over the dataset).
   *
   * @param stepSize Step size for each iteration.
   * @param batchSize Batch size to use for each step.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *     limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param shuffle If true, the function order is shuffled; otherwise, each
   *     function is visited in linear order.
   * @param updatePolicy Instantiated update policy used to adjust the given
   *                     parameters.
   * @param decayPolicy Instantiated decay policy used to adjust the step size.
   * @param resetPolicy Flag that determines whether update policy parameters
   *                    are reset before every Optimize call.
   * @param exactObjective Calculate the exact objective (Default: estimate the
   *        final objective obtained on the last pass over the data).
   */
  SGD(const double stepSize = 0.01,
      const size_t batchSize = 32,
      const size_t maxIterations = 100000,
      const double tolerance = 1e-5,
      const bool shuffle = true,
      const UpdatePolicyType& updatePolicy = UpdatePolicyType(),
      const DecayPolicyType& decayPolicy = DecayPolicyType(),
      const bool resetPolicy = true,
      const bool exactObjective = false);

  /**
   * Clean any memory associated with the SGD object.
   */
  ~SGD();

  /**
   * Optimize the given function using stochastic gradient descent.  The given
   * starting point will be modified to store the finishing point of the
   * algorithm, and the final objective value is returned.
   *
   * @tparam SeparableFunctionType Type of the function to be optimized.
   * @tparam MatType Type of matrix to optimize with.
   * @tparam GradType Type of matrix to use to represent function gradients.
   * @tparam v Types of callback functions.
   * @param function Function to optimize.
   * @param iterate Starting point (will be modified).
   * @param callbacks Callback functions.
   * @return Objective value of the final point.
   */
  template<typename SeparableFunctionType,
           typename MatType,
           typename GradType,
           typename... CallbackTypes>
  typename std::enable_if<IsArmaType<GradType>::value,
      typename MatType::elem_type>::type
  Optimize(SeparableFunctionType& function,
           MatType& iterate,
           CallbackTypes&&... callbacks);

  //! Forward the MatType as GradType.
  template<typename SeparableFunctionType,
           typename MatType,
           typename... CallbackTypes>
  typename MatType::elem_type Optimize(SeparableFunctionType& function,
                                       MatType& iterate,
                                       CallbackTypes&&... callbacks)
  {
    return Optimize<SeparableFunctionType, MatType, MatType,
        CallbackTypes...>(function, iterate,
        std::forward<CallbackTypes>(callbacks)...);
  }

  //! Get the step size.
  double StepSize() const { return stepSize; }
  //! Modify the step size.
  double& StepSize() { return stepSize; }

  //! Get the batch size.
  size_t BatchSize() const { return batchSize; }
  //! Modify the batch size.
  size_t& BatchSize() { return batchSize; }

  //! Get the maximum number of iterations (0 indicates no limit).
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations (0 indicates no limit).
  size_t& MaxIterations() { return maxIterations; }

  //! Get the tolerance for termination.
  double Tolerance() const { return tolerance; }
  //! Modify the tolerance for termination.
  double& Tolerance() { return tolerance; }

  //! Get whether or not the individual functions are shuffled.
  bool Shuffle() const { return shuffle; }
  //! Modify whether or not the individual functions are shuffled.
  bool& Shuffle() { return shuffle; }

  //! Get whether or not the actual objective is calculated.
  bool ExactObjective() const { return exactObjective; }
  //! Modify whether or not the actual objective is calculated.
  bool& ExactObjective() { return exactObjective; }

  //! Get whether or not the update policy parameters
  //! are reset before Optimize call.
  bool ResetPolicy() const { return resetPolicy; }
  //! Modify whether or not the update policy parameters
  //! are reset before Optimize call.
  bool& ResetPolicy() { return resetPolicy; }

  //! Get the update policy.
  const UpdatePolicyType& UpdatePolicy() const { return updatePolicy; }
  //! Modify the update policy.
  UpdatePolicyType& UpdatePolicy() { return updatePolicy; }

  //! Get the instantiated update policy type.  Be sure to check its type with
  //! Has() before using!
  const Any& InstUpdatePolicy() const { return instUpdatePolicy; }
  //! Modify the instantiated update policy type.  Be sure to check its type
  //! with Has() before using!
  Any& InstUpdatePolicy() { return instUpdatePolicy; }

  //! Get the step size decay policy.
  const DecayPolicyType& DecayPolicy() const { return decayPolicy; }
  //! Modify the step size decay policy.
  DecayPolicyType& DecayPolicy() { return decayPolicy; }

  //! Get the instantiated decay policy type.  Be sure to check its type with
  //! Has() before using!
  const Any& InstDecayPolicy() const { return instDecayPolicy; }
  //! Modify the instantiated decay policy type.  Be sure to check its type with
  //! Has() before using!
  Any& InstDecayPolicy() { return instDecayPolicy; }

 private:
  //! The step size for each example.
  double stepSize;

  //! The batch size for processing.
  size_t batchSize;

  //! The maximum number of allowed iterations.
  size_t maxIterations;

  //! The tolerance for termination.
  double tolerance;

  //! Controls whether or not the individual functions are shuffled when
  //! iterating.
  bool shuffle;

  //! Controls whether or not the actual Objective value is calculated.
  bool exactObjective;

  //! The update policy used to update the parameters in each iteration.
  UpdatePolicyType updatePolicy;

  //! The decay policy used to update the step size.
  DecayPolicyType decayPolicy;

  //! Flag indicating whether update policy
  //! should be reset before running optimization.
  bool resetPolicy;

  //! Flag indicating whether the update policy
  //! parameters have been initialized.
  bool isInitialized;

  //! The initialized update policy.
  Any instUpdatePolicy;
  //! The initialized decay policy.
  Any instDecayPolicy;
};

using StandardSGD = SGD<VanillaUpdate>;

using MomentumSGD = SGD<MomentumUpdate>;

using NesterovMomentumSGD = SGD<NesterovMomentumUpdate>;

using QHSGD = SGD<QHUpdate>;
} // namespace ens

// Include implementation.
#include "sgd_impl.hpp"

#endif
