/**
 * @file NelderMead.hpp
 * @author Marcus Edel
 *
 * Definition of an Nelder-Mead with adaptive parameters proposed by F. Gao and
 * L. Han in "Implementing the Nelder-Mead simplex algorithm with adaptive
 * parameters".
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_NELDER_MEAD_HPP
#define ENSMALLEN_NELDER_MEAD_HPP

#include "affine_simplexer.hpp"

namespace ens {

/**
 * Nelder-Mead is a direct search method. It keeps track of the function value
 * at a number of points in the search space and iteratively generates a
 * sequence of simplices to approximate an optimal point.
 *
 * For more information, please refer to:
 *
 * @code
 * @article{Nelder1965ASM,
 *   author  = {J. Nelder and R. Mead},
 *   title   = {A Simplex Method for Function Minimization},
 *   year    = {1965},
 *   journal = {Comput. J.},
 *   volume  = {7},
 *   pages   = {308-313}
 * }
 * @endcode
 *
 * @code
 * @article{Fuchang2012,
 *   author    = {Gao, Fuchang and Han, Lixing},
 *   title     = {Implementing the Nelder-Mead Simplex Algorithm with Adaptive
 *                Parameters},
 *   year      = {2012},
 *   publisher = {Kluwer Academic Publishers},
 *   volume    = {51},
 *   number    = {1},
 *   journal   = {Comput. Optim. Appl.},
 *   month     = jan,
 *   pages     = {259–277}
 * }
 * @endcode
 *
 * NelderMead can optimize arbitrary functions.  For more details, see the
 * documentation on function types included with this distribution or on the
 * ensmallen website.
 *
 * @tparam SimplexerType Simplex policy used by Nelder Mead to construct the
 * initial simplex.
 */
template<typename SimplexerType = AffineSimplexer>
class NelderMeadType
{
 public:
  /**
   * Construct the NelderMead optimizer with the default adaptive parameters
   * scheme as used in "Implementing the Nelder-Mead simplex algorithm with
   * adaptive parameters" by F. Gao and L. Han (2010). These are based on the
   * dimensionality of the problem, and are given by
   *
   * alpha = 1
   * beta = 1 + 2 / n
   * gamma = 0.75 - 0.5 / n
   * delta = 1 - 1 / n
   *
   * where n is the dimensionality of the problem.
   *
   * The defaults here are not necessarily good for the given problem, so it is
   * suggested that the values used be tailored to the task at hand.  The
   * maximum number of iterations refers to the maximum number of points that
   * are processed (i.e., one iteration equals one point; one iteration does not
   * equal one pass over the dataset).
   *
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *     limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param simplexer The simplex policy used to construct the initial simplex.
   */
  NelderMeadType(const size_t maxIterations = 100000,
                 const double tolerance = 1e-15,
                 const SimplexerType& simplexer = SimplexerType());
  /**
   * Construct the NelderMead optimizer with the given parameters. The defaults
   * here are not necessarily good for the given problem, so it is suggested
   * that the values used be tailored to the task at hand.  The maximum number
   * of iterations refers to the maximum number of points that are processed
   * (i.e., one iteration equals one point; one iteration does not equal one
   * pass over the dataset).
   *
   * @param alpha The reflection parameter.
   * @param beta The expansion parameter.
   * @param gamma The contraction parameter.
   * @param delta The shrink step parameter.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *     limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param simplexer The simplex policy used to construct the initial simplex.
   */
  NelderMeadType(const double alpha,
                 const double beta,
                 const double gamma,
                 const double delta,
                 const size_t maxIterations = 100000,
                 const double tolerance = 1e-15,
                 const SimplexerType& simplexer = SimplexerType());

  /**
   * Optimize the given function using NelderMead. The given starting point will
   * be modified to store the finishing point of the algorithm, and the final
   * objective value is returned.
   *
   * @tparam ArbitraryFunctionType Type of the function to be optimized.
   * @tparam MatType Type of matrix to optimize.
   * @tparam CallbackTypes Types of callback functions.
   * @param function Function to optimize.
   * @param iterate Starting point (will be modified).
   * @param callbacks Callback functions.
   * @return Objective value of the final point.
   */
  template<typename ArbitraryFunctionType,
           typename MatType,
           typename... CallbackTypes>
  typename MatType::elem_type Optimize(ArbitraryFunctionType& function,
                                       MatType& iterate,
                                       CallbackTypes&&... callbacks);

  //! Get the reflection parameter.
  double Alpha() const { return alpha; }
  //! Modify the reflection parameter.
  double& Alpha() { return alpha; }

  //! Get the expansion parameter.
  double Beta() const { return beta; }
  //! Modify the expansion parameter.
  double& Beta() { return beta; }

  //! Get the contraction parameter.
  double Gamma() const { return gamma; }
  //! Modify the contraction parameter.
  double& Gamma() { return gamma; }

  //! Get the shrink step parameter.
  double Delta() const { return delta; }
  //! Modify the shrink step parameter.
  double& Delta() { return delta; }

  //! Get the maximum number of iterations (0 indicates no limit).
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations (0 indicates no limit).
  size_t& MaxIterations() { return maxIterations; }

  //! Get the tolerance for termination.
  double Tolerance() const { return tolerance; }
  //! Modify the tolerance for termination.
  double& Tolerance() { return tolerance; }

  //! Get the simplexer policy.
  const SimplexerType& Simplexer() const { return simplexer; }
  //! Modify the simplexer policy.
  SimplexerType& Simplexer() { return simplexer; }

 private:
  /**
   * Helper function to shift the given order based on the given function
   * objective.
   *
   * @param order The current order of the function objectives.
   * @param fx The current function values.
   * @param value Function value to be inserted into the current order.
   * @return Position of the given objective in the given order.
   */
  size_t ShiftOrder(arma::uvec& order,
                    const arma::rowvec& fx,
                    const double value)
  {
    // Figure out at which index we have to put the new value.
    size_t i = 0;
    for (; i < fx.n_elem; ++i)
      if (fx(order(i)) > value) break;

    // Shift every value after 'i' by one to the right.
    const size_t tmp = order(fx.n_elem - 1);
    for (size_t d = fx.n_elem - 1; d > i; --d)
      order(d) = order(d - 1);
    order(i) = tmp;

    return i;
  }

  //! The reflection parameter.
  double alpha;

  //! The expansion parameter.
  double beta;

  //! The contraction parameter.
  double gamma;

  //! The shrink step parameter.
  double delta;

  //! The maximum number of allowed iterations.
  size_t maxIterations;

  //! The tolerance for termination.
  double tolerance;

  //! Locally stored simplexer.
  SimplexerType simplexer;

  //! Whether to use the default adaptive parameters from the "Implementing the
  //! Nelder-Mead simplex algorithmwith adaptive parameters" paper.
  bool adaptiveDefault;
};

// Convenience typedef.
using NelderMead = NelderMeadType<AffineSimplexer>;

} // namespace ens

// Include implementation.
#include "nelder_mead_impl.hpp"

#endif
