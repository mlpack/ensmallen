/**
 * @file dtlz3_impl.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Implementation of the second DTLZ function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_DTLZ_TWO_IMPL_HPP
#define ENSMALLEN_PROBLEMS_DTLZ_TWO_IMPL_HPP

// In case it hasn't been included yet.
#include "dtlz2.hpp"

namespace ens {
namespace test {

inline DTLZ2::DTLZ2(const size_t numVariables,
										const size_t numObjectives) :
		numVariables(numVariables),
		numObjectives(numObjectives)
{ /* Nothing to do here */ }

inline arma::vec DTLZ2::Evaluate(const arma::mat& coordinates) const
{
	arma::vec f(numObjectives);

	size_t k = numVariables - numObjectives + 1;

	double g = 0;
	for (size_t i = numVariables - k; i < numVariables; i++)
		g += (coordinates(i, 0) - 0.5) * (coordinates(i, 0) - 0.5);

	for (size_t i = 0; i < numObjectives; i++)
	{
		f[i] = 1 + g;
		for (size_t j = 0; j < numObjectives - (i + 1); j++)
			f[i] *= std::cos(coordinates(j, 0) * 0.5 * M_PI);
		if (i != 0)
			f[i] *= std::sin(coordinates(numObjectives - (i + 1), 0) * M_PI / 2);
	}
	
	return f;	
}

} // namespace test
} // namespace ens

#endif
