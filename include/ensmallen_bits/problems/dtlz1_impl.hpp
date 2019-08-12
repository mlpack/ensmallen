/**
 * @file dtlz1_impl.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Implementation of the first DTLZ function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_DTLZ_ONE_IMPL_HPP
#define ENSMALLEN_PROBLEMS_DTLZ_ONE_IMPL_HPP

// In case it hasn't been included yet.
#include "dtlz1.hpp"

namespace ens {
namespace test {

inline DTLZ1::DTLZ1(const size_t numVariables,
										const size_t numObjectives) :
		numVariables(numVariables),
		numObjectives(numObjectives)
{ /* Nothing to do here */ }

inline arma::vec DTLZ1::Evaluate(const arma::mat& coordinates) const
{
	arma::vec f(numObjectives);

	size_t k = numVariables - numObjectives + 1;

	double g = 0;
	for (size_t i = numVariables - k; i < numVariables; i++)
	{
		double x = coordinates(i, 0);
		g += (x - 0.5) * (x - 0.5) - std::cos(20 * M_PI * (x - 0.5));
	}

	g = 100 * (k + g);
	for (size_t i = 0; i < numObjectives; i++)
	{
		f[i] = (1 + g) / 2;
		for (size_t j = 0; j < numObjectives - (i + 1); j++)
			f[i] *= coordinates(j, 0);
		if (i != 0)
			f[i] *= 1 - coordinates(numObjectives - (i + 1), 0);
	}

	return f;
}

} // namespace test
} // namespace ens

#endif
