/**
 * @file gradient_descent_test_function.hpp
 * @author Sumedh Ghaisas
 *
 * Very simple test function for SGD.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_GRADIENT_DESCENT_TEST_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_GRADIENT_DESCENT_TEST_FUNCTION_HPP

namespace ens {
namespace test {

//! Very, very simple test function which is the composite of three other
//! functions.  The gradient is not very steep far away from the optimum, so a
//! larger step size may be required to optimize it in a reasonable number of
//! iterations.
class GDTestFunction
{
 public:
  //! Nothing to do for the constructor.
  GDTestFunction() { }

  //! Evaluate a function.
  template<typename MatType>
  typename MatType::elem_type Evaluate(const MatType& coordinates) const;

  //! Evaluate the gradient of a function.
  template<typename MatType, typename GradType>
  void Gradient(const MatType& coordinates, GradType& gradient) const;

  // Note: GetInitialPoint(), GetFinalPoint(), and GetFinalObjective() are not
  // required for using ensmallen to optimize this function!  They are
  // specifically used as a convenience just for ensmallen's testing
  // infrastructure.

  //! Get the starting point.
  template<typename MatType>
  MatType GetInitialPoint() const { return MatType("1; 3; 2"); }

  //! Get the final point.
  template<typename MatType = arma::mat>
  MatType GetFinalPoint() const { return MatType("0; 0; 0"); }

  //! Get the final objective.
  double GetFinalObjective() const { return 0.0; }
};

} // namespace test
} // namespace ens

#include "gradient_descent_test_function_impl.hpp"

#endif
