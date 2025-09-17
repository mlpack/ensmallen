/**
 * @file aug_lagrangian_function.hpp
 * @author Ryan Curtin
 *
 * Contains a utility class for AugLagrangian.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_AUG_LAGRANGIAN_AUG_LAGRANGIAN_FUNCTION_HPP
#define ENSMALLEN_AUG_LAGRANGIAN_AUG_LAGRANGIAN_FUNCTION_HPP

namespace ens {

/**
 * This is a utility class used by AugLagrangian, meant to wrap a
 * LagrangianFunction into a function usable by a simple optimizer like L-BFGS.
 * Given a LagrangianFunction which follows the format outlined in the
 * documentation for AugLagrangian, this class provides Evaluate(), Gradient(),
 * and GetInitialPoint() functions which allow this class to be used with a
 * simple optimizer like L-BFGS.
 *
 * This class can be specialized for your particular implementation -- commonly,
 * a faster method for computing the overall objective and gradient of the
 * augmented Lagrangian function can be implemented than the naive, default
 * implementation given.  Use class template specialization and re-implement all
 * of the methods (unfortunately, C++ specialization rules mean you have to
 * re-implement everything).
 *
 * @tparam LagrangianFunction Lagrangian function to be used.
 */
template<typename LagrangianFunction, typename VecType>
class AugLagrangianFunction
{
 public:
  /**
   * Initialize the AugLagrangianFunction with the given LagrangianFunction,
   * Lagrange multipliers, and initial penalty parameter.
   *
   * @param function Lagrangian function.
   * @param lambda Initial Lagrange multipliers.
   * @param sigma Initial penalty parameter.
   */
  AugLagrangianFunction(LagrangianFunction& function,
                        VecType& lambda,
                        double& sigma);
  /**
   * Evaluate the objective function of the Augmented Lagrangian function, which
   * is the standard Lagrangian function evaluation plus a penalty term, which
   * penalizes unsatisfied constraints.
   *
   * @param coordinates Coordinates to evaluate function at.
   * @return Objective function.
   */
  template<typename MatType>
  typename MatType::elem_type Evaluate(const MatType& coordinates) const;

  /**
   * Evaluate the gradient of the Augmented Lagrangian function.
   *
   * @param coordinates Coordinates to evaluate gradient at.
   * @param gradient Matrix to store gradient into.
   */
  template<typename MatType, typename GradType>
  void Gradient(const MatType& coordinates, GradType& gradient) const;

  /**
   * Get the initial point of the optimization (supplied by the
   * LagrangianFunction).
   *
   * @return Initial point.
   */
  template<typename MatType>
  const MatType& GetInitialPoint() const;

  // Get the Lagrange multipliers.
  VecType& Lambda() { return lambda; }
  // Get the penalty parameter.
  double& Sigma() { return sigma; }

  //! Get the Lagrangian function.
  const LagrangianFunction& Function() const { return function; }
  //! Modify the Lagrangian function.
  LagrangianFunction& Function() { return function; }

 private:
  //! Instantiation of the function to be optimized.
  LagrangianFunction& function;

  //! The Lagrange multipliers.
  VecType& lambda;
  //! The penalty parameter.
  double& sigma;
};

} // namespace ens

// Include basic implementation.
#include "aug_lagrangian_function_impl.hpp"

#endif // ENSMALLEN_AUG_LAGRANGIAN_AUG_LAGRANGIAN_FUNCTION_HPP

