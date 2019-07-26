/**
 * @file dtlz7_impl.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Implementation of the seventh DTLZ function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_DTLZ_SEVEN_IMPL_HPP
#define ENSMALLEN_PROBLEMS_DTLZ_SEVEN_IMPL_HPP

// In case it hasn't been included yet.
#include "dtlz7.hpp"

namespace ens {
namespace test {

inline DTLZ7::DTLZ7(const size_t numVariables,
										const size_t numObjectives) :
		numVariables(numVariables),
		numObjectives(numObjectives)
{ /* Nothing to do here */ }

inline arma::vec DTLZ7::Evaluate(const arma::mat& coordinates) const
{
	arma::vec f(numObjectives);

	size_t k = numVariables - numObjectives + 1;

	double g = 0;
	for (size_t i = numVariables - k; i < numVariables; i++)
		g += coordinates[i][0];

	g = 1 + (9 * g) / k;

	for (size_t i = 0; i < numObjectives - 1; i++)
		f[i] = coordinates[i][0];

	double h = 0;
	for (size_t i = 0; i < numObjectives - 1)
	{
		f[i] = coordinates[i][0];
		h += (f[i] / (1.0 + g)) * (1 + std::sin(3.0 * M_PI * f[i]));
	}

	h = numObjectives - h;
	f[numObjectives - 1] = (1 + g) * h;

	return f;
}

} // namespace test
} // namespace ens

#endif
