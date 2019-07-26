/**
 * @file dtlz2.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Definition of the second DTLZ function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_DTLZ_TWO_HPP
#define ENSMALLEN_PROBLEMS_DTLZ_TWO_HPP

namespace ens {
namespace test {

class DTLZ2
{
 public:
 	//! Initialize the DTLZ2 function.
 	DTLZ2(const size_t numVariables,
 				const size_t numObjectives);

  //! Return the number of functions.
  size_t NumObjectives() const { return numObjectives; }

  //! Return the number of variables.
  size_t NumVariables() const { return numVariables; }

  //! Get the starting point.
  arma::mat GetInitialPoint() const
  { 
    return arma::fill::zeros<arma::mat>(numVariables, 1);
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
#include "dtlz2_impl.hpp"

#endif
