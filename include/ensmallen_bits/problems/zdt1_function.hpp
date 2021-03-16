/**
 * @file zdt1_function.hpp
 * @author Nanubala Gnana Sai
 *
 * Implementation of the first ZDT(Zitzler, Deb and Thiele) test.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_PROBLEMS_ZDT_ONE_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_ZDT_ONE_FUNCTION_HPP

namespace ens {
namespace test {
/**
 * The ZDT1 function, defined by:
 * \f[
 * g(x) = 1 + 9(\sum_{i=2}^{n} x_i )/(n-1)
 * f_1(x) = x_i
 * h(f1, g) = g(x)[1-\sqrt{f_1/g}\ ]
 * \f]
 *
 * This is a 30-variable problem(n = 30) with a
 * convex optimal set.
 *
 * Bounds of the variable space is:
 * 0 <= x_i <= 1 for i = 1,...,n.
 *
 * This should be optimized to g(x) = 1.0, at:
 * x_1* in [0, 1] ; x_i* = 0 for i = 2,...,n
 *
 * For more information, please refer to:
 *
 * @code
 * @article{Zitzler2000,
 *   title   = {Comparison of multiobjective evolutionary algorithms:
 *              Empirical results},
 *   author  = {Zitzler, Eckart and Deb, Kalyanmoy and Thiele, Lothar},
 *   journal = {Evolutionary computation},
 *   year    = {2000},
 *   doi     = {10.1162/106365600568202}
 * }
 * @endcode
 */


  }
}








#endif