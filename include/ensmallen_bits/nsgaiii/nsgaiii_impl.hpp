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

// In case it hasn't been included yet.
#include "nsgaiii.hpp"

namespace ens {

inline NSGAIII::NSGAIII(const size_t populationSize,
					    const size_t maxGenerations,
					    const double crossoverProb,
					    const double distrIndex,
					    const size_t numPartitions) :
		populationSize(populationSize),
		maxGenerations(maxGenerations),
		crossoverProb(crossoverProb),
		distrIndex(distrIndex),
		numPartitions(numPartitions),
		userDefinedSet(false)
{ /* Nothing to do here */ }

template<typename MultiObjectiveFunctionType>
arma::cube NSGAIII::Optimize(MultiObjectiveFunctionType& function, arma::mat& iterate)
{
  // The user didn't define a reference set. We need to define one ourselves.
  if (!userDefinedSet)
  {
  	std::vector<arma::mat> referenceVec;
  	arma::mat refPoint(function.NumObjectives(), 1);
  	FindReferencePoints(referenceVec, refPoint, numPartitions, numPartitions, 0);
  	referenceSet = arma::cube(iterate.n_rows, iterate.n_cols, referenceVec.size());
  	for (size_t i = 0; i < referenceVec.size(); i++)
 	  referenceSet.slice(i) = referenceVec[i];
  }

  // Initialize initial population.
  arma::cube population = arma::randn(iterate.n_rows, iterate.n_cols, populationSize);
  population.each_slice() += iterate;

  // Initialize helper variables.
  arma::cube R, Q;
  std::vector<std::vector<size_t>> fronts;
  arma::mat fitnessValues(function.NumObjectives(), 2 * populationSize);

  for (size_t gen = 0; gen < maxGenerations; gen++)
  {
	// Mate and create offsprings.
	Q = Mate(population);

	// Create a population made of the old population and the new offsprings.
	R = arma::join_slices(population, Q);

	// Evaluate R.
	for (size_t i = 0; i < R.n_slices; i++)
	  fitnessValues.col(i) = function.Evaluate(R.slice(i));

	// Find next population.
	fronts.clear();
	NonDominatedSorting(fitnessValues, fronts);
	size_t l = 0, numMembers = 0;
	arma::cube nextPop = population;
	arma::mat newFitness(function.NumObjectives(), populationSize);
	while (numMembers < populationSize)
	{
	  if (fronts[l].size() + numMembers >= populationSize)
	    break;
	  for (size_t i = 0; i < fronts[l].size(); i++)
	  {
		newFitness.col(numMembers) = fitnessValues.col(fronts[l][i]);
		nextPop.slice(numMembers++) = population.slice(fronts[l][i]);
	  }
	  l++;
	}
	size_t K = population - numMembers;
	if (K == 0)
	  continue;

	// Normalization step.
	arma::mat asf(function.NumObjectives(), function.NumObjectives(),
		arma::fill::eye);

	asf.diag().fill(1e6);

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

	arma::vec mins(max.n_rows);
	for (size_t i = 0; i < mins.n_elem; i++)
	  mins[i] = max.row(i).index_max();

	arma::mat extremePoints(function.NumObjectives(), function.NumObjectives());
	for (size_t i = 0; i < function.NumObjectives(); i++)
	  extremePoints.col(i) = f.col(mins[i]);

	arma::mat b(extremePoints.n_rows, extremePoints.n_rows, arma::fill::ones);
	arma::mat plane = arma::solve(extremePoints.t(), b);
	arma::mat intercepts = 1 / plane;
	fitnessValues.each_col() /= intercepts;

	// Association step.
	arma::mat perpDists(referenceSet.n_slices, numMembers);
	arma::vec associates(numMembers);
	for (size_t i = 0; i < numMembers; i++)
	{
	  double min = DBL_MAX;
	  size_t minIdx = 0;
	  for (size_t j = 0; j < referenceSet.n_slices; j++)
	  {
	  	arma::mat s = nextPop.slice(i);
	  	arma::mat w = referenceSet.slice(j);
	  	double wnorm = arma::norm(w, 2);
	  	perpDists(j, i) = arma::norm(s - (w.t() * s * w) / std::pow(wnorm, 2), 2);
	  	if (perpDists(j, i) < min)
	  	{
	  	  min = perpDists(j, i);
	  	  minIdx = j;
	  	}
	  }
	  associates[i] = minIdx;
	}

	// Compute niche count.
	arma::uvec nicheCount(referenceSet.n_slices);
	for (size_t i = 0; i < associates.n_elem; i++)
	  nicheCount[i] = (arma::uword)arma::accu(arma::find(associates == i));

	// Niching step.
	size_t k = 1;
	double min = nicheCount.min();
	while (k <= K)
	{
	  std::vector<size_t> jmin;
	  for (size_t i = 0; i < nicheCount.n_elem; i++)
	  {
	  	if (nicheCount[i] == min)
	  	  jmin.push_back(i);
	  }
	  size_t randIdx = arma::randi<arma::uvec>(1, arma::distr_param(0, jmin.size()))[0];

	  double distMin = DBL_MAX;
	  size_t minIdx = 0;
	  for (size_t i = 0; i < fronts[l].size(); i++)
	  {
		double wnorm = arma::norm(referenceSet.slice(randIdx), 2);
		double perpDist = arma::norm(nextPop.slice(i) -
			(referenceSet.slice(randIdx).t() * nextPop.slice(i) *
			referenceSet.slice(randIdx)) / (wnorm * wnorm), 2);
		if (perpDist < distMin)
		{
		  distMin = perpDist;
		  minIdx = i;
		}
	  }
	  nextPop.slice(numMembers++) = population.slice(fronts[l][minIdx]);
      fronts[l].erase(fronts[l].begin() + minIdx);
	  nicheCount[randIdx]++;
	  k++;
	}
	population = nextPop;
  }

  // Evaluate final population.
  arma::mat fitnessValues(function.NumObjectives(), population.n_slices);
  for (size_t i = 0; i < R.n_slices; i++)
	fitnessValues.col(i) = function.Evaluate(population.slice(i));

  // Find the fronts.
  fronts.clear();
  NonDominatedSorting(fitnessValues, fronts);

  arma::cube bestFront(population.n_rows, population.n_slices, fronts[0].size());
  for (size_t i = 0; i < fronts[0].size(); i++)
	bestFront.slice(i) = population.slice(fronts[0][i]);

  return bestFront;
}

inline void NSGAIII::NonDominatedSorting(const arma::mat& fitnessValues,
								  std::vector<std::vector<size_t>>& fronts)
{
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
}

inline arma::cube NSGAIII::Mate(arma::cube& population)
{
  arma::cube offspring = arma::cube(population.n_rows, population.n_cols,
	  populationSize, arma::fill::zeros);
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
			beta = std::pow(u / 0.5, 1 / (distrIndex + 1));
		  else
			beta = std::pow(2 * (1 - u), -(distrIndex + 1));
		  double p1 = population(i, 0, selection[0]);
		  double p2 = population(i, 0, selection[1]);

		  double c1 = 0.5 * ((p1 + p2) - beta * std::abs(p1 - p2));
		  double c2 = 0.5 * ((p1 + p2) + beta * std::abs(p1 - p2));

		  offspring(i, 0, currentOffspring) = c1;
		  offspring(i, 0, currentOffspring + 1) = c2;
		}
		else
		{
		  double p1 = population(i, 0, selection[0]);
		  double p2 = population(i, 0, selection[1]);

		  offspring(i, 0, currentOffspring) = p1;
		  offspring(i, 0, currentOffspring + 1) = p2;
		}
	  }
	  currentOffspring += 2;
	}
  }

  return offspring;
}

inline arma::cube& NSGAIII::ReferenceSet()
{
  userDefinedSet = true;
  return referenceSet;
}

inline void NSGAIII::FindReferencePoints(std::vector<arma::mat>& referenceVec,
										 arma::mat& refPoint,
										 size_t numPartitions,
										 size_t beta,
										 size_t depth)
{
  if (depth == refPoint.n_elem - 1)
  {
  	refPoint(depth, 0) = beta / (numPartitions);
  	referenceSet.push_back(refPoint);
  }
  else
  {
  	for (size_t i = 0; i <= beta; i++)
  	{
  	  refPoint(depth, 0) = i / numPartitions;
  	  FindReferencePoints(referenceVec, refPoint, numPartitions, beta - i, depth
  	      + 1);
  	}
  }
}

} // namespace ens

#endif
