/**
 * @file sparse_test_function.hpp
 * @author Shikhar Bhardwaj
 *
 * Sparse test function for Parallel SGD.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_SPARSE_TEST_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_SPARSE_TEST_FUNCTION_HPP

namespace ens {
namespace test {

// A simple test function. Each dimension has a parabola with a
// distinct minimum. Each update is guaranteed to be sparse(only a single
// dimension is updated in the decision variable by each thread). At the end of
// a reasonable number of iterations, each value in the decision variable should
// be at the vertex of the parabola in that dimension.
class SparseTestFunction
{
 public:
  //! Set members in the default constructor.
  SparseTestFunction();

  //! Return 4 (the number of functions).
  size_t NumFunctions() const { return 4; }

  //! Return 4 (the number of features).
  size_t NumFeatures() const { return 4; }

  //! Evaluate a function.
  template<typename MatType>
  typename MatType::elem_type Evaluate(const MatType& coordinates,
                                       const size_t i,
                                       const size_t batchSize = 1) const;

  //! Evaluate all the functions.
  template<typename MatType>
  typename MatType::elem_type Evaluate(const MatType& coordinates) const;

  //! Evaluate the gradient of a function.
  template<typename MatType,
           typename GradType = arma::SpMat<typename MatType::elem_type>>
  void Gradient(const MatType& coordinates,
                const size_t i,
                GradType& gradient,
                const size_t batchSize = 1) const;

  //! Evaluate the gradient of a feature function.
  template<typename MatType,
           typename GradType = arma::SpMat<typename MatType::elem_type>>
  void PartialGradient(const MatType& coordinates,
                       const size_t j,
                       GradType& gradient) const;

  // Note: GetInitialPoint(), GetFinalPoint(), and GetFinalObjective() are not
  // required for using ensmallen to optimize this function!  They are
  // specifically used as a convenience just for ensmallen's testing
  // infrastructure.

  //! Get the starting point.
  template<typename MatType>
  MatType GetInitialPoint() const { return MatType("0 0 0 0;"); }

  //! Get the final point.
  template<typename MatType = arma::mat>
  MatType GetFinalPoint() const { return MatType("2.0 1.0 1.5 4.0"); }

  //! Get the final objective.
  double GetFinalObjective() const { return 123.75; }

 private:
  // Each quadratic polynomial is monic. The intercept and coefficient of the
  // first order term is stored.

  //! The vector storing the intercepts
  arma::vec intercepts;

  //! The vector having coefficients of the first order term
  arma::vec bi;
};

} // namespace test
} // namespace ens

// Include implementation.
#include "sparse_test_function_impl.hpp"

#endif // ENSMALLEN_PROBLEMS_SPARSE_TEST_FUNCTION_HPP
