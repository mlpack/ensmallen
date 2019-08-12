/**
 * @file dtlz5.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Definition of the fifth DTLZ function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_DTLZ_FIVE_HPP
#define ENSMALLEN_PROBLEMS_DTLZ_FIVE_HPP

namespace ens {
namespace test {

/**
 * A class that represents the fifth DTLZ function.
 *
 * For more info, please see:
 * 
 * @code
 * @article{Deb2005,
 *    title  = {Scalable Test Problems for Evolutionary Multiobjective
 *              Optimization},
 *    author = {Kalyanmoy Deb, Lothar Thiele, Marco Laumanns and Zitzler
 *              Eckart}
 *    year   = {2005}
 *    url    = {https://link.springer.com/chapter/10.1007/1-84628-137-7_6}  
 * }
 * @endcode
 */
class DTLZ5
{
 public:
 	//! Initialize the DTLZ5 function.
 	DTLZ5(const size_t numVariables,
 				const size_t numObjectives);

  //! Return the number of functions.
  size_t NumObjectives() const { return numObjectives; }

  //! Return the number of variables.
  size_t NumVariables() const { return numVariables; }

  //! Get the starting point.
  arma::mat GetInitialPoint() const
  { 
    return arma::mat(numVariables, 1, arma::fill::zeros);
  }

  /*
   * Evaluate a function with the given coordinates.
   *
   * @param coordinates The function coordinates.
   */
  arma::vec Evaluate(const arma::mat& coordinates) const;

 private:
 	size_t numVariables;

 	size_t numObjectives;
};

} // namespace test
} // namespace ens

// Include implementation.
#include "dtlz5_impl.hpp"

#endif
