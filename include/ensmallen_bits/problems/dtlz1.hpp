/**
 * @file dtlz1.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Definition of the first DTLZ function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_DTLZ_ONE_HPP
#define ENSMALLEN_PROBLEMS_DTLZ_ONE_HPP

namespace ens {
namespace test {

class DTLZ1
{
 public:
 	//! Initialize the DTLZ1 function.
 	DTLZ1(const size_t numVariables,
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

  /**
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
#include "dtlz1_impl.hpp"

#endif
