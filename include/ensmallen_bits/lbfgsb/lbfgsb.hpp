/**
 * @file lbfgsb.hpp
 * @author Khizir Siddiqui
 *
 * The generic L-BFGS-B optimizer.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_LBFGSB_LBFGSB_HPP
#define ENSMALLEN_LBFGSB_LBFGSB_HPP

#include <ensmallen_bits/function.hpp>

namespace ens {

/**
 * The L-BFGS-B optimizer, which is an extension of L-BFGS designed to handle
 * simple bound constraints. It uses gradient projection to identify the active
 * of constraints, and then performs minimization using the L-BFGS approximation
 * to the Hessian.
 *
 * L_BFGS_B can optimize differentiable functions with box constraints.
 * For more details, see the documentation on function types included with this
 * distribution or on the ensmallen website.
 *
 * The algorithm is based on the paper:
 * "A Limited Memory Algorithm for Bound Constrained Optimization"
 * by R.H. Byrd and P. Lu and J. Nocedal
 */
class L_BFGS_B
{
 public:
  /**
   * Initialize the L-BFGS-B object.  There are many parameters that can be set
   * for the optimization, but default values are given for each of them.
   *
   * @param numBasis Number of memory points to be stored (default 5).
   * @param lowerBound Lower bound for the coordinates (can be a scalar or matrix).
   * @param upperBound Upper bound for the coordinates (can be a scalar or matrix).
   * @param maxIterations Maximum number of iterations for the optimization
   *     (0 means no limit and may run indefinitely).
   * @param armijoConstant Controls the accuracy of the line search routine for
   *     determining the Armijo condition.
   * @param wolfe Parameter for detecting the Wolfe condition.
   * @param minGradientNorm Minimum gradient norm required to continue the
   *     optimization.
   * @param factr Minimum relative function value decrease to continue
   *     the optimization.
   * @param maxLineSearchTrials The maximum number of trials for the line search
   *     (before giving up).
   * @param minStep The minimum step of the line search.
   * @param maxStep The maximum step of the line search.
   */
  L_BFGS_B(const size_t numBasis = 10, /* same default as scipy */
           const arma::mat& lowerBound = arma::mat(),
           const arma::mat& upperBound = arma::mat(),
           const size_t maxIterations = 10000, /* many but not infinite */
           const double armijoConstant = 1e-4,
           const double wolfe = 0.9,
           const double minGradientNorm = 1e-6,
           const double factr = 1e-15,
           const size_t maxLineSearchTrials = 50,
           const double minStep = 1e-20,
           const double maxStep = 1e20);

  /**
   * Initialize the L-BFGS-B object with scalar bounds.
   */
  L_BFGS_B(const size_t numBasis,
           const double lowerBound,
           const double upperBound,
           const size_t maxIterations = 10000,
           const double armijoConstant = 1e-4,
           const double wolfe = 0.9,
           const double minGradientNorm = 1e-6,
           const double factr = 1e-15,
           const size_t maxLineSearchTrials = 50,
           const double minStep = 1e-20,
           const double maxStep = 1e20);

  /**
   * Use L-BFGS-B to optimize the given function, starting at the given iterate
   * point and finding the minimum.  The maximum number of iterations is set in
   * the constructor (or with MaxIterations()).  Alternately, another overload
   * is provided which takes a maximum number of iterations as a parameter.  The
   * given starting point will be modified to store the finishing point of the
   * algorithm, and the final objective value is returned.
   *
   * @tparam FunctionType Type of the function to be optimized.
   * @tparam MatType Type of matrix to optimize with.
   * @tparam GradType Type of matrix to use to represent function gradients.
   * @tparam CallbackTypes Types of callback functions.
   * @param function Function to optimize; must have Evaluate() and Gradient().
   * @param iterate Starting point (will be modified).
   * @param callbacks Callback functions.
   * @return Objective value of the final point.
   */
  template<typename FunctionType,
           typename MatType,
           typename GradType,
           typename... CallbackTypes>
  typename std::enable_if<IsMatrixType<GradType>::value,
      typename MatType::elem_type>::type
  Optimize(FunctionType& function,
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

  //! Get the memory size.
  size_t NumBasis() const { return numBasis; }
  //! Modify the memory size.
  size_t& NumBasis() { return numBasis; }

  //! Get the lower bound.
  const arma::mat& LowerBound() const { return lowerBound; }
  //! Modify the lower bound.
  arma::mat& LowerBound() { return lowerBound; }

  //! Get the upper bound.
  const arma::mat& UpperBound() const { return upperBound; }
  //! Modify the upper bound.
  arma::mat& UpperBound() { return upperBound; }

  //! Get the maximum number of iterations.
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations.
  size_t& MaxIterations() { return maxIterations; }

  //! Get the Armijo condition constant.
  double ArmijoConstant() const { return armijoConstant; }
  //! Modify the Armijo condition constant.
  double& ArmijoConstant() { return armijoConstant; }

  //! Get the Wolfe parameter.
  double Wolfe() const { return wolfe; }
  //! Modify the Wolfe parameter.
  double& Wolfe() { return wolfe; }

  //! Get the minimum gradient norm.
  double MinGradientNorm() const { return minGradientNorm; }
  //! Modify the minimum gradient norm.
  double& MinGradientNorm() { return minGradientNorm; }

  //! Get the factr value.
  double Factr() const { return factr; }
  //! Modify the factr value.
  double& Factr() { return factr; }

  //! Get the maximum number of line search trials.
  size_t MaxLineSearchTrials() const { return maxLineSearchTrials; }
  //! Modify the maximum number of line search trials.
  size_t& MaxLineSearchTrials() { return maxLineSearchTrials; }

  //! Return the minimum line search step size.
  double MinStep() const { return minStep; }
  //! Modify the minimum line search step size.
  double& MinStep() { return minStep; }

  //! Return the maximum line search step size.
  double MaxStep() const { return maxStep; }
  //! Modify the maximum line search step size.
  double& MaxStep() { return maxStep; }

 private:
  //! Size of memory for this L-BFGS-B optimizer.
  size_t numBasis;
  //! Lower bound for coordinates.
  arma::mat lowerBound;
  //! Upper bound for coordinates.
  arma::mat upperBound;
  //! Maximum number of iterations.
  size_t maxIterations;
  //! Parameter for determining the Armijo condition.
  double armijoConstant;
  //! Parameter for detecting the Wolfe condition.
  double wolfe;
  //! Minimum gradient norm required to continue the optimization.
  double minGradientNorm;
  //! Minimum relative function value decrease to continue the optimization.
  double factr;
  //! Maximum number of trials for the line search.
  size_t maxLineSearchTrials;
  //! Minimum step of the line search.
  double minStep;
  //! Maximum step of the line search.
  double maxStep;
  //! Controls early termination of the optimization process.
  bool terminate;
  //! Flag indicating whether scalar bounds were provided.
  bool usingScalarBounds;

  /**
   * Project the given point onto the bounds.
   */
  template<typename MatType>
  void ProjectPoint(MatType& iterate);

  /**
   * Find the generalized Cauchy point.
   *
   * @param iterate The current point.
   * @param gradient The gradient at the current point.
   * @param theta The scaling factor from the L-BFGS matrix.
   * @param W The W matrix from the L-BFGS approximation.
   * @param M The M matrix from the L-BFGS approximation.
   * @param cauchyPoint Vector to store the resulting Cauchy point.
   * @param c Vector to store the c vector used in subspace minimization.
   * @param activeSet Boolean vector specifying if a coordinate is at bound.
   */
  template<typename MatType>
  void GeneralizedCauchyPoint(const MatType& iterate,
                              const MatType& gradient,
                              const typename MatType::elem_type theta,
                              const arma::mat& W,
                              const arma::mat& M,
                              MatType& cauchyPoint,
                              arma::vec& c,
                              arma::uvec& activeSet);

  /**
   * Perform subspace minimization over the free variables.
   *
   * @param iterate The current point.
   * @param cauchyPoint The generalized Cauchy point.
   * @param gradient The gradient at the current point.
   * @param theta The scaling factor from the L-BFGS matrix.
   * @param W The W matrix from the L-BFGS approximation.
   * @param M The M matrix from the L-BFGS approximation.
   * @param c The c vector from the Cauchy point computation.
   * @param activeSet Boolean vector specifying if a coordinate is at bound.
   * @param searchDirection Vector to store the resulting search direction.
   */
  template<typename MatType>
  void SubspaceMinimization(const MatType& iterate,
                            const MatType& cauchyPoint,
                            const MatType& gradient,
                            const typename MatType::elem_type theta,
                            const arma::mat& W,
                            const arma::mat& M,
                            const arma::vec& c,
                            const arma::uvec& activeSet,
                            MatType& searchDirection);

  /**
   * Reconstruct the L-BFGS memory representation into W and M matrices.
   *
   * @param iterationNum The iteration number.
   * @param s Differences between the iterate and old iterate matrix.
   * @param y Differences between the gradient and the old gradient matrix.
   * @param theta Scaling factor to use.
   * @param W The resulting W matrix.
   * @param M The resulting M matrix.
   */
  template<typename CubeType>
  void ComputeWM(const size_t iterationNum,
                 const CubeType& s,
                 const CubeType& y,
                 const typename CubeType::elem_type theta,
                 arma::mat& W,
                 arma::mat& M);

  /**
   * Perform a projected back-tracking line search along the search direction
   * to calculate a step size satisfying the Wolfe conditions.
   *
   * @param function Function to optimize.
   * @param functionValue Value of the function at the initial point.
   * @param iterate The initial point to begin the line search from.
   * @param gradient The gradient at the initial point.
   * @param searchDirection A vector specifying the search direction.
   * @param finalStepSize The resulting step size (0 if no step).
   * @param callbacks Callback functions.
   *
   * @return false if no step size is suitable, true otherwise.
   */
  template<typename FunctionType,
           typename ElemType,
           typename MatType,
           typename GradType,
           typename... CallbackTypes>
  bool LineSearch(FunctionType& function,
                  ElemType& functionValue,
                  MatType& iterate,
                  GradType& gradient,
                  MatType& newIterateTmp,
                  const GradType& searchDirection,
                  ElemType& finalStepSize,
                  CallbackTypes&... callbacks);
};

using LBFGSB = L_BFGS_B;

} // namespace ens

#include "lbfgsb_impl.hpp"

#endif // ENSMALLEN_LBFGSB_LBFGSB_HPP
