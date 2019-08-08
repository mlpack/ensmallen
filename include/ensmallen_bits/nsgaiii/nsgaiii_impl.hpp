/**
 * @file nsgaiii_impl.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Non-dominated Sorting Genetic Algorithm - III (NSGA-III)
 * A multi-objective optimizer. \temp
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_NSGAIII_NSGAIII_IMPL_HPP
#define ENSMALLEN_NSGAIII_NSGAIII_IMPL_HPP

#include "nsgaiii.hpp"

namespace ens {

inline NSGAIII::NSGAIII(const size_t populationSize,
							          const size_t maxGenerations,
							          const double mutationProb,
							          const double mutationSize,
							          const double selectPercent) :
		populationSize(populationSize),
		maxGenerations(maxGenerations),
		mutationProb(mutationProb),
		mutationSize(mutationSize),
		selectPercent(selectPercent)
{ /* Nothing to do here */ }

template<typename MultiObjectiveFunctionType>
double NSGAIII::Optimize(MultiObjectiveFunctionType& function, arma::mat& iterate)
{
	// Initialize initial population.
	arma::cube population = arma::randn(iterate.n_rows, iterate.n_cols, populationSize);
	population.each_slice() += iterate;

	// Evaluate initial population.

	std::vector<size_t> S;
	arma::cube R, Q;

	for (size_t gen = 0; gen < maxGenerations; gen++)
	{
		// Mate and create offsprings.
		Q = Mate(population);

		// Create a population made of the old population and the new offsprings.
		R = arma::join_slices(population, Q);
	
		// Evaluate R.
		arma::mat fitnessValues(function.NumObjectives(), R.n_slices);
		for (size_t i = 0; i < R.n_slices; i++)
			fitnessValues.col(i) = function.Evaluate(R.slice(i));

		// Find next population.
		std::vector<std::vector<size_t>> fronts = NonDominatedSorting(fitnessValues);
		size_t l = 0, numMembers = 0;
		arma::cube nextPop = population;
		arma::mat newFitness(populationSize, function.NumObjectives());
		while (numMembers < populationSize)
		{
			if (fronts[l].size() + numMembers >= populationSize)
			{
				size_t i = 0;
				while (numMembers != populationSize)
					nextPop.slice(numMembers++) = population.slice(fronts[l][i++]);
				continue;
			}
			for (size_t i = 0; i < fronts[l].size(); i++)
				nextPop.slice(numMembers++) = population.slice(fronts[l][i]);
			l++;
		}

		// Normalization step.
		arma::mat asf(function.NumObjectives(), function.NumObjectives(),
				arma::fill::eye);

		for (size_t i = 0; i < asf.n_rows; i++)
		{
			for (size_t j = 0; j < asf.n_cols; j++)
			{
				if (asf(i, j) == 0)
					asf(i, j) = 1e6;
			}
		}
		
		arma::mat f(function.NumObjectives(), fronts[0].size());
		for (size_t i = 0; i < f.n_cols; i++)
			f.col(i) = fitnessValues.col(fronts[0][i]);

		arma::vec zmin(function.NumObjectives());
		for (size_t i = 0; i < function.NumObjectives(); i++)
			zmin[i] = f.row(i).min();

		for (size_t i = 0; i < fronts[0].size(); i++)
			f.col(i) -= zmin;

		for (auto it = fitnessValues.begin(); it != fitnessValues.end(); it++)
				if (*it <= 1e-3)
					*it = 0;

		arma::mat max(asf.n_rows, fronts[0].size());
		for (size_t i = 0; i < asf.n_rows; i++)
		{
			arma::mat f_asf = f.t() * asf.row(i);
			for (size_t j = 0; j < f_asf.n_rows; j++)
				max(i, j) = f_asf.row(j).max();
		}

		arma::vec min(max.n_rows);
		for (size_t i = 0; i < min.n_elem; i++)
			min[i] = max.row(i).index_max();

		arma::mat extremePoints(function.NumObjectives(), function.NumObjectives());
		for (size_t i = 0; i < function.NumObjectives(); i++)
			extremePoints.col(i) = f.col(min[i]);

		arma::mat b(extremePoints.n_rows, extremePoints.n_rows, arma::fill::ones);
		arma::mat plane = arma::solve(extremePoints.t(), b);
		arma::mat intercepts = 1 / plane;
		fitnessValues.each_col() /= intercepts;

		// Association step.

	}
}

std::vector<std::vector<size_t>> NSGAIII::NonDominatedSorting
		(const arma::mat& fitnessValues)
{
	std::vector<std::vector<size_t>> fronts;
	std::vector<std::vector<size_t>> dominatedSolutions(fitnessValues.n_cols);
	arma::vec n(fitnessValues.n_cols, arma::fill::zeros);
	arma::vec ranks(fitnessValues.n_cols, arma::fill::zeros);
	for (size_t i = 0; i < fitnessValues.n_cols; i++)
	{
		for (size_t j = 0; j < fitnessValues.n_cols; j++)
		{
			if (i == j)
				continue;

			if (arma::all(fitnessValues.col(i) >= fitnessValues.col(j)) &&
					arma::any(fitnessValues.col(i) > fitnessValues.col(j)))
				dominatedSolutions[i].push_back(j);
			else if (arma::all(fitnessValues.col(j) >= fitnessValues.col(i)) &&
					arma::any(fitnessValues.col(j) > fitnessValues.col(i)))
				n[i]++;
		}
		if (n[i] == 0)
		{
			ranks[i] = 1;
			if (fronts.size() == 0)
				fronts.push_back(std::vector<size_t>());
			fronts[0].push_back(i);
		}
	}

	size_t i = 1;
	while (fronts[i].size() != 0)
	{
		std::vector<size_t> Q;
		for (auto const& y : fronts[i])
		{
			for (auto const& x : dominatedSolutions[y])
			{
				n[x]--;
				if (n[x] == 0)
				{
					ranks[x] = i + 1;
					Q.push_back(x);
				}
			}
		}
		i++;
		fronts[i] = Q;
	}

	return fronts;
}

arma::cube NSGAIII::Mate(arma::cube& population)
{
	arma::cube offspring = arma::cube(population.n_rows, population.n_cols,
			numOffspring, arma::fill::zeros);
	size_t currentOffspring = 0, numOffspring = populationSize;
	while (currentOffspring < numOffspring)
	{
		arma::uvec selection = arma::randi<arma::uvec>(2, arma::distr_param(0,
				populationSize));
		if (arma::randu() > crossoverProb)
		{
			offspring.slice(currentOffspring++) = population.slice(selection[0]);
			offspring.slice(currentOffspring++) = population.slice(selection[1]);
		}
		else
		{
			for (size_t i = 0; i < population.n_rows; i++)
			{
				if (arma::randu() < 0.5)
				{
					double u = arma::randu();
					double beta = 0;
					if (u < 0.5)
						beta = std::pow(u / 0.5, 1 / (n + 1));
					else
						beta = std::pow(2 * (1 - u), -(n + 1));
					double p1 = population.slice(selection[0])[i][0]; 
					double p2 = population.slice(selection[1])[i][0];

					double c1 = 0.5 * ((p1 + p2) - beta * std::abs(p1 - p2));
					double c2 = 0.5 * ((p1 + p2) + beta * std::abs(p1 - p2));

					offspring[i][0][currentOffspring] = c1;
					offspring[i][0][currentOffspring + 1] = c2;
				}
				else
				{
					double p1 = population[i][0][selection[0]]; 
					double p2 = population[i][0][selection[1]];

					offspring[i][0][currentOffspring] = c1;
					offspring[i][0][currentOffspring + 1] = c2;
				}
			}
			currentOffspring += 2;
		}
	}

	return offspring;

}
	
}

#endif