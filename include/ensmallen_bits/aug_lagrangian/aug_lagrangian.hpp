/**
 * @file aug_lagrangian.hpp
 * @author Ryan Curtin
 *
 * Definition of AugLagrangian class, which implements the Augmented Lagrangian
 * optimization method (also called the 'method of multipliers'.  This class
 * uses the L-BFGS optimizer.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_AUG_LAGRANGIAN_AUG_LAGRANGIAN_HPP
#define ENSMALLEN_AUG_LAGRANGIAN_AUG_LAGRANGIAN_HPP

#include <ensmallen_bits/lbfgs/lbfgs.hpp>

#include "aug_lagrangian_function.hpp"

namespace ens {

/**
 * The AugLagrangian class implements the Augmented Lagrangian method of
 * optimization.  In this scheme, a penalty term is added to the Lagrangian.
 * This method is also called the "method of multipliers".
 *
 * AugLagrangian can optimize constrained functions.  For more details, see the
 * documentation on function types included with this distribution or on the
 * ensmallen website.
 */
class AugLagrangian
{
 public:
  /**
   * Initialize the Augmented Lagrangian with the default L-BFGS optimizer.
   * @param penaltyThresholdFactor When the penalty threshold is updated set
   *    the penalty threshold to the penalty multplied by this factor. The
   *    default value of 0.25 is is taken from Burer and Monteiro (2002).
   * @param sigmaUpdateFactor When sigma is updated  multiply sigma by this
   *    value. The default value of 10 is taken from Burer and Monteiro (2002).
   * @param maxIterations Maximum number of iterations of the Augmented
   *     Lagrangian algorithm.  0 indicates no maximum.
   */
  AugLagrangian(const size_t maxIterations = 1000,
                const double penaltyThresholdFactor = 0.25,
                const double sigmaUpdateFactor = 10.0,
                const L_BFGS& lbfgs = L_BFGS());

  /**
   * Optimize the function.  The value '1' is used for the initial value of each
   * Lagrange multiplier.  To set the Lagrange multipliers yourself, use the
   * other overload of Optimize().
   *
   * @tparam LagrangianFunctionType Function which can be optimized by this
   *     class.
   * @tparam MatType Type of matrix to optimize with.
   * @tparam GradType Type of matrix to use to represent function gradients.
   * @tparam CallbackTypes Types of callback functions.
   * @param function The function to optimize.
   * @param coordinates Output matrix to store the optimized coordinates in.
   * @param callbacks Callback functions.
   */
  template<typename LagrangianFunctionType,
           typename MatType,
           typename GradType,
           typename... CallbackTypes>
  typename std::enable_if<IsArmaType<GradType>::value, bool>::type
  Optimize(LagrangianFunctionType& function,
           MatType& coordinates,
           CallbackTypes&&... callbacks);

  //! Forward the MatType as GradType.
  template<typename LagrangianFunctionType,
           typename MatType,
           typename... CallbackTypes>
  bool Optimize(LagrangianFunctionType& function,
                MatType& coordinates,
                CallbackTypes&&... callbacks)
  {
    return Optimize<LagrangianFunctionType, MatType, MatType,
        CallbackTypes...>(function, coordinates,
        std::forward<CallbackTypes>(callbacks)...);
  }

  /**
   * Optimize the function, giving initial estimates for the Lagrange
   * multipliers.  The vector of Lagrange multipliers will be modified to
   * contain the Lagrange multipliers of the final solution (if one is found).
   *
   * @tparam LagrangianFunctionType Function which can be optimized by this
   *      class.
   * @tparam MatType Type of matrix to optimize with.
   * @tparam GradType Type of matrix to use to represent function gradients.
   * @tparam CallbackTypes Types of callback functions.
   * @param function The function to optimize.
   * @param coordinates Output matrix to store the optimized coordinates in.
   * @param initLambda Vector of initial Lagrange multipliers.  Should have
   *     length equal to the number of constraints.
   * @param initSigma Initial penalty parameter.
   * @param callbacks Callback functions.
   */
  template<typename LagrangianFunctionType,
           typename MatType,
           typename GradType,
           typename... CallbackTypes>
  typename std::enable_if<IsArmaType<GradType>::value, bool>::type
  Optimize(LagrangianFunctionType& function,
           MatType& coordinates,
           const arma::vec& initLambda,
           const double initSigma,
           CallbackTypes&&... callbacks);

  //! Forward the MatType as GradType.
  template<typename LagrangianFunctionType,
           typename MatType,
           typename... CallbackTypes>
  bool Optimize(LagrangianFunctionType& function,
                MatType& coordinates,
                const arma::vec& initLambda,
                const double initSigma,
                CallbackTypes&&... callbacks)
  {
    return Optimize<LagrangianFunctionType, MatType, MatType,
        CallbackTypes...>(function, coordinates, initLambda, initSigma,
        std::forward<CallbackTypes>(callbacks)...);
  }

  //! Get the L-BFGS object used for the actual optimization.
  const L_BFGS& LBFGS() const { return lbfgs; }
  //! Modify the L-BFGS object used for the actual optimization.
  L_BFGS& LBFGS() { return lbfgs; }

  //! Get the Lagrange multipliers.
  const arma::vec& Lambda() const { return lambda; }
  //! Modify the Lagrange multipliers (i.e. set them before optimization).
  arma::vec& Lambda() { return lambda; }

  //! Get the penalty parameter.
  double Sigma() const { return sigma; }
  //! Modify the penalty parameter.
  double& Sigma() { return sigma; }

  //! Get the maximum iterations
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum iterations
  size_t& MaxIterations() { return maxIterations; }

  //! Get the penalty threshold updating parameter
  double PenaltyThresholdFactor() const { return penaltyThresholdFactor; }
  //! Modify the penalty threshold updating parameter
  double& PenaltyThresholdFactor() { return penaltyThresholdFactor; }

  //! Get the sigma update factor
  double SigmaUpdateFactor() const { return sigmaUpdateFactor; }
  //! Modify the sigma update factor
  double& SigmaUpdateFactor() { return sigmaUpdateFactor; }

 private:
  //! Maximum number of iterations.
  size_t maxIterations;

  //! Parameter for updating the penalty threshold
  double penaltyThresholdFactor;

  //! Parameter for updating sigma
  double sigmaUpdateFactor;

  //! The L-BFGS optimizer that we will use.
  L_BFGS lbfgs;

  //! Controls early termination of the optimization process.
  bool terminate;

  //! Lagrange multipliers.
  arma::vec lambda;

  //! Penalty parameter.
  double sigma;

  /**
   * Internal optimization function: given an initialized AugLagrangianFunction,
   * perform the optimization itself.
   */
  template<typename LagrangianFunctionType,
           typename MatType,
           typename GradType,
           typename... CallbackTypes>
  typename std::enable_if<IsArmaType<GradType>::value, bool>::type
  Optimize(AugLagrangianFunction<LagrangianFunctionType>& augfunc,
           MatType& coordinates,
           CallbackTypes&&... callbacks);

  //! Forward the MatType as GradType.
  template<typename LagrangianFunctionType,
           typename MatType,
           typename... CallbackTypes>
  bool Optimize(AugLagrangianFunction<LagrangianFunctionType>& function,
                MatType& coordinates,
                CallbackTypes&&... callbacks)
  {
    return Optimize<LagrangianFunctionType, MatType, MatType,
        CallbackTypes...>(function, coordinates,
        std::forward<CallbackTypes>(callbacks)...);
  }
};

} // namespace ens

#include "aug_lagrangian_impl.hpp"

#endif // ENSMALLEN_AUG_LAGRANGIAN_AUG_LAGRANGIAN_HPP

