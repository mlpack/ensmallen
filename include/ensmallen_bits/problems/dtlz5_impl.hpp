/**
 * @file dtlz5_impl.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Implementation of the fifth DTLZ function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_DTLZ_FIVE_IMPL_HPP
#define ENSMALLEN_PROBLEMS_DTLZ_FIVE_IMPL_HPP

// In case it hasn't been included yet.
#include "dtlz5.hpp"

namespace ens {
namespace test {

inline DTLZ5::DTLZ5(const size_t numVariables,
										const size_t numObjectives) :
		numVariables(numVariables),
		numObjectives(numObjectives)
{ /* Nothing to do here */ }

inline arma::vec DTLZ5::Evaluate(const arma::mat& coordinates) const
{
	arma::vec f(numObjectives);

	size_t k = numVariables - numObjectives + 1;

	double g = 0;
	for (size_t i = numVariables - k; i < numVariables; i++)
		g += (coordinates[i][0] - 0.5) * (coordinates[i][0] - 0.5);

	double t = M_PI / (4 * (1 + g));
	arma::vec theta(numObjectives - 1);
	for (size_t i = 0; i < numObjectives - 1; i++)
		theta[i] = t * (1 + 2 * g * coordinates[i][0]);

	for (size_t i = 0; i < numObjectives; i++)
	{
		f[i] = 1 + g;
		for (size_t j = 0; j < numObjectives - (i + 1); j++)
			f[i] *= std::cos(theta[j]);
		if (i != 0)
			f[i] *= std::sin(theta[numObjectives - i - 1]);
	}

	return f;	
}

} // namespace test
} // namespace ens

#endif
