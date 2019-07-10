/**
 * @file holder_table_function.hpp
 * @author Suryoday Basak
 *
 * Definition of the Holder table function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_HOLDER_TABLE_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_HOLDER_TABLE_FUNCTION_HPP

namespace ens {
namespace test {

/**
 * The Holder table function, defined by
 *
 * \f[
 * f(x1, x2) = - |sin(x1) * cos(x2) * exp(|1 - (sqrt(x1^2 + x2^2) / pi)|)|
 * \f]
 *
 * This should optimize to f(x1, x2) = -19.2085, at
 *                                     (x1, x2) = [-8.05502, -9.66459], or
 *                                     (x1, x2) = [8.05502, -9.66459],  or
 *                                     (x1, x2) = [-8.05502, 9.66459],  or
 *                                     (x1, x2) = [8.05502, 9.66459]
 *
 * For more information, please refer to:
 *
 * @code
 * @article{Mishra2006,
 *   doi = {10.2139/ssrn.926132},
 *   year = {2006},
 *   publisher = {Elsevier {BV}},
 *   author = {S. K. Mishra},
 *   title = {Some New Test Functions for Global Optimization and
 *            Performance of Repulsive Particle Swarm Method},
 *   journal = {{SSRN} Electronic Journal}
 * }
 * @endcode
 */
class HolderTableFunction
{
 public:
  //! Initialize the HolderTableFunction.
  HolderTableFunction();

  /**
   * Shuffle the order of function visitation. This may be called by the
   * optimizer.
   */
  void Shuffle();

  //! Return 1 (the number of functions).
  size_t NumFunctions() const { return 1; }

  //! Get the starting point.
  arma::mat GetInitialPoint() const { return arma::mat("7; 7"); }

  /**
   * Evaluate a function for a particular batch-size.
   *
   * @param coordinates The function coordinates.
   * @param begin The first function.
   * @param batchSize Number of points to process.
   */
  double Evaluate(const arma::mat& coordinates,
      const size_t begin,
      const size_t batchSize) const;

  /**
   * Evaluate a function with the given coordinates.
   *
   * @param coordinates The function coordinates.
   */
  double Evaluate(const arma::mat& coordinates) const;
};

} // namespace test
} // namespace ens

// Include implementation.
#include "holder_table_function_impl.hpp"

#endif // ENSMALLEN_PROBLEMS_HOLDER_TABLE_FUNCTION_HPP
