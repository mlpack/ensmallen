/**
 * @file moead_impl.hpp
 * @author Utkarsh Rai
 *
 * Implementation of the MOEA/D algorithm. Used for multi-objective
 * optimization problems on arbitrary functions.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more Information.
 */

#ifndef ENSMALLEN_MOEAD_MOEAD_IMPL_HPP
#define ENSMALLEN_MOEAD_MOEAD_IMPL_HPP

#include "moead.hpp"

namespace ens {

inline MOEAD::MOEAD(const size_t populationSize,
                    const size_t numGeneration,
                    const double crossoverProb,
                    const double mutationProb,
                    const double mutationStrength,
                    const size_t neighborSize,
                    const double scalingFactor,
                    const double distributionIndex,
                    const double neighborProb,
                    const arma::vec& lowerBound,
                    const arma::vec& upperBound) :
    populationSize(populationSize),
    numGeneration(numGeneration),
    crossoverProb(crossoverProb),
    mutationProb(mutationProb),
    mutationStrength(mutationStrength),
    neighborSize(neighborSize),
    scalingFactor(scalingFactor),
    distributionIndex(distributionIndex),
    neighborProb(neighborProb),
    lowerBound(lowerBound),
    upperBound(upperBound),
    numObjectives(0)
  { /* Nothing to do here. */ }

inline MOEAD::MOEAD(const size_t populationSize,
                    const size_t numGeneration,
                    const double crossoverProb,
                    const double mutationProb,
                    const double mutationStrength,
                    const size_t neighborSize,
                    const double scalingFactor,
                    const double distributionIndex,
                    const double neighborProb,
                    const double lowerBound,
                    const double upperBound) :
    populationSize(populationSize),
    numGeneration(numGeneration),
    crossoverProb(crossoverProb),
    mutationProb(mutationProb),
    mutationStrength(mutationStrength),
    neighborSize(neighborSize),
    scalingFactor(scalingFactor),
    distributionIndex(distributionIndex),
    neighborProb(neighborProb),
    lowerBound(lowerBound * arma::ones(1, 1)),
    upperBound(upperBound * arma::ones(1, 1)),
    numObjectives(0)
  { /* Nothing to do here. */ }

//! Optimize the function.
template<typename MatType,
         typename... ArbitraryFunctionType,
         typename... CallbackTypes>
typename MatType::elem_type MOEAD::Optimize(std::tuple<ArbitraryFunctionType...>& objectives,
                                            MatType& iterate,
                                            CallbackTypes&&... callbacks)
{
  // Check if lower bound is a vector of a single dimension.
  if (lowerBound.n_rows == 1)
    lowerBound = lowerBound(0, 0) * arma::ones(iterate.n_rows, iterate.n_cols);

  // Check if lower bound is a vector of a single dimension.
  if (upperBound.n_rows == 1)
    upperBound = upperBound(0, 0) * arma::ones(iterate.n_rows, iterate.n_cols);
  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;

  if(lowerBound.size() != upperBound.size())
  {
    throw std::logic_error("MOEAD::Optimize(): size of lowerBound and upperBound "
        "must be equal.");
  }

  if(lowerBound.size() != iterate.n_elem)
  {
    throw std::logic_error("MOEAD::Optimize(): there should be a lower bound and "
        "an upper bound for each variable in the initial point.");
  }

  if(populationSize < neighborSize + 1)
  {
      std::ostringstream oss;
      oss << "MOEAD::Optimize(): " << "neighborSize is " << neighborSize
          << "but populationSize is " << populationSize << "(should be"
          << " atleast " << neighborSize + 1 << std::endl;
      throw std::logic_error(oss.str());
  }

  // Number of objective functions. Represented by M in the paper.
  numObjectives = sizeof...(ArbitraryFunctionType);
  // Dimensionality of variable space. Also referred to as number of genes.
  size_t numVariables = iterate.n_rows;

  // Controls early termination of the optimization process.
  bool terminate = false;
  //TODO: Add more checks?
  // 1.1 The external population, non-dominated solutions.
  std::vector<MatType> externalPopulation;
  std::vector<arma::vec> externalPopulationFValue;

  arma::Col<size_t> shuffle;
  // The Lambda matrix. Each vector represents a decomposition subproblem.
  arma::Mat<ElemType> weights(numObjectives, populationSize, arma::fill::randu); //FIXME: Should use weight generation method

  // 1.2 Storing the indices of nearest neighbours of each weight vector.
  arma::Mat<arma::uword> neighborIndices(neighborSize, populationSize);
  for (size_t i = 0; i < populationSize; ++i)
  {
      // To temporarily store the distance between weights(i) and each other weights.
      arma::rowvec distances(populationSize);
      distances =
              arma::sqrt(arma::sum(arma::pow(weights.col(i) - weights.each_col(), 2)));
      arma::uvec sortedIndices = arma::stable_sort_index(distances);
      // Ignore distance from self
      neighborIndices.col(i) = sortedIndices(arma::span(1, neighborSize));
  }

  // 1.3 Random generation of the initial population.
  std::vector<MatType> population(populationSize);
  for (size_t i = 0; i < populationSize; ++i)
  {
	  population[i] =
		  arma::randu<MatType>(iterate.n_rows, iterate.n_cols) - 0.5 + iterate;
    
	  for (size_t geneIdx = 0; geneIdx < numVariables; ++geneIdx)
	  {
		  if (population[i][geneIdx] < lowerBound[geneIdx])
			  population[i][geneIdx] = lowerBound[geneIdx];
		  else if (population[i][geneIdx] > upperBound[geneIdx])
			  population[i][geneIdx] = upperBound[geneIdx];
	  }
  }

  // 1.3 F-value initialisation for the population.
  arma::mat FValue(numObjectives, populationSize);
  EvaluateObjectives(population, objectives, FValue);

  // 1.4 Initialize the ideal point z.
  arma::mat idealPoint(numObjectives, 1);
  idealPoint.fill(std::numeric_limits<ElemType>::max());
  for (size_t i = 0; i < numObjectives; ++i)
  {
    for (size_t j = 0; j < populationSize; ++j)
    {
      idealPoint(i, 0) = std::min(idealPoint(i, 0), FValue(i, j));
    }
  }

  terminate |= Callback::BeginOptimization(*this, objectives, iterate, callbacks...);

  // 2 The main loop.
  for (size_t g = 0; g < numGeneration; ++g)
  {
    shuffle = std::get<0>(objectives).Shuffle(populationSize);
    for (size_t i : shuffle)
    {
      terminate |= Callback::StepTaken(*this, objectives, iterate, callbacks...);

      // 2.1 Randomly select two indices in neighborIndices(i) and use them
      // to make a child.
      size_t r1, r2, r3;
      r1 = i;
      P_TYPE pFlag = P_TYPE::NONE; //TODO: Perhaps a boolean with comment would be better
	    (arma::randu < neighborProb) ? pFlag = P_TYPE::FROM_NEIGHBOR
				                    : pFlag = P_TYPE::FROM_POPULATION;
	    std::tie(r2, r3) = MatingSelection(i, neighborIndices, pFlag);

      // 2.2 Reproduction: Apply the Differential Operator on the selected indices
      // followed by Mutation.
      MatType candidate(iterate.nrows, iterate.ncols); //TODO: Potentially wrong, because iterate can be a population swarm
      double delta = arma::randu();
      if (delta < crossoverProb)
      {
        for (size_t geneIdx = 0; geneIdx < numVariables, ++geneIdx)
        {
          candidate[geneIdx] = population[r1][geneIdx] +
                        scaleFactor * (population[r2][geneIdx] -
                                        population[r3][geneIdx]);

          // Handle boundary condition
          if (candidate[geneIdx] < lowerBound[geneIdx])
          {
            candidate[geneIdx] = lowerBound[geneIdx] +
                          arma::randu() * (population[r1][geneIdx] -
                                            lowerBound[geneIdx]);
          }
          else if (candidate[geneIdx] > upperBound[geneIdx])
          {
            candidate[geneIdx] = upperrBound[geneIdx] +
                          arma::randu() * (upperBound[geneIdx] -
                                            population[r1][geneIdx]);
          }
          else
            candidate[geneIdx] = population[r1][geneIdx];
        }
      }

	    Mutate(candidate, 1.f / numVariables, lowerBound, upperBound);

      // Store solution for candidate.
      arma::mat candidateFval(numObjectives, 1);
      EvaluateObjectives(std::vector<MatType>(candidate), objectives, candidateFval);

      // 2.3 Update of ideal point.
      for (size_t idx = 0;idx < numObjectives;++idx)
      {
        idealPoint(idx, 0) = std::min(idealPoint(idx, 0),
            candidateFval(idx, 0));
      }

      // 2.4 Update of the neighbouring solutions. //FIXME: Sample from either the population OR neighbor based on pFlag
      for (size_t idx = 0;idx < neighborSize;++idx)
      {
        if (DecomposedSingleObjective(weights.col(neighborIndices(idx, i)),
              idealPoint.col(0), candidateFval.col(0))
            <= DecomposedSingleObjective(
               weights.col(neighborIndices(idx,i)),
               idealPoint.col(0), FValue.col(neighborIndices(idx, i))))
        {
          population.at(neighborIndices(idx, i)) = candidate[0];
          FValue.col(neighborIndices(idx, i)) = candidateFval.col(0);
        }
      }

      // 2.5 Updating External Population.
      if (!externalPopulation.empty())
      {
        arma::mat first(numObjectives, 1);
        auto df = [&](MatType firstMat) -> bool
        {
          std::vector<MatType> wrapperFirst(1), wrapperSecond(1);
          wrapperFirst[0] = firstMat;
          EvaluateObjectives(wrapperFirst, objectives, first);
          return Dominates(candidateFval.col(0), first.col(0));
        };
        //! Remove the part that is dominated by candidate.
        externalPopulation.erase(
            std::remove_if(
              externalPopulation.begin(),
              externalPopulation.end(),
              df), externalPopulation.end());

        //! Check if any of the remaining members of external population dominate
        //! candidate.
        bool flag = 0;
        std::vector<MatType> wrapperFirst(1);
        for (size_t idx = 0; idx < externalPopulation.size(); ++idx)
        {
          wrapperFirst[0]=externalPopulation[idx];
          first.clear();
          first.resize(numObjectives, 1);
          EvaluateObjectives(wrapperFirst, objectives, first);
          if (Dominates(first.col(0), candidateFval.col(0)))
          {
            flag = 1;
            break;
          }
        }
        if (flag == 0)
        {
          externalPopulation.push_back(candidate[0]);
          externalPopulationFValue.push_back(candidateFval.col(0));
        }
      }
      else
      {
        externalPopulation.push_back(candidate[0]);
        externalPopulationFValue.push_back(candidateFval.col(0));
      }
    }
  }
  bestFront = std::move(externalPopulation);

  Callback::EndOptimization(*this, objectives, iterate, callbacks...);

  ElemType performance = std::numeric_limits<ElemType>::max();
  for (arma::Col<ElemType> objective: externalPopulationFValue)
  {
    if (arma::accu(objective) < performance)
      performance = arma::accu(objective);
  }

  return performance;
}

inline std::tuple<int, int>
MOEAD::MatingSelection(const size_t popIdx,
    const arma::Mat<arma::uword>& neighborIndices,
    P_TYPE pFlag)
{
	size_t k, l;
	if (pFlag == P_TYPE::FROM_NEIGHBOR)
	{
		k = neighborIndices(
			arma::randi(arma::distr_param(0, neighborSize - 1)), popIdx);
		l = neighborIndices(
			arma::randi(arma::distr_param(0, neighborSize - 1)), popIdx);
		if (k == l)
		{
			if (k == neighborSize - 1)
				--k;
			else
				++k;
		}
	}
	else
	{
		k = arma::randi(arma::distr_param(0, populationSize - 1));
		l = arma::randi(arma::distr_param(0, populationSize - 1));
		if (k == l)
		{
			if (k == populationSize - 1)
				--k;
			else
				++k;
		}
	}

  return std::make_tuple(k, l);
}

//! Perform mutation of the candidate.
template<typename MatType>
inline void MOEAD::Mutate(MatType& child,
    const double& rate,
    const arma::vec& lowerBound,
    const arma::vec& upperBound)
{
  size_t numVariables = lowerBound.n_elem;
  double rnd, delta1, delta2, mutationPower, deltaq;
  double current, currentLower, currentUpper, value, upperDelta;

  for (size_t j = 0; j < numVariables; ++j)
  {
    double determiner = arma::randu();
    if (determiner <= rate)
    {
      current = child[j];
      currentLower = lowerBound(j);
      currentUpper = upperBound(j);
      delta1 = (current - currentLower) / (currentUpper - currentLower);
      delta2 = (currentUpper - current) / (currentUpper - currentLower);    
      rnd = arma::randu();
      mutationPower = 1.0 /( distributionIndex + 1.0 );
      if (rnd < 0.5)
      {
        upperDelta = 1.0 - delta1;
        value = 2.0 * rnd + (1.0 - 2.0 * rnd) *
            (std::pow(upperDelta, (distributionIndex + 1.0)));
        deltaq = std::pow(value, mutationPower) - 1.0;
      }
      else
      {
        upperDelta = 1.0 - delta2;
        value = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) *
            (pow(upperDelta, (distributionIndex + 1.0)));
        deltaq = 1.0 - (pow(value, mutationPower));
      }

      current += deltaq * (currentUpper - currentLower);
      if (current < currentLower) current = currentLower;
      if (current > currentUpper) current = currentUpper;
      child[j] = current;
    }
  }
}

//! Calculate the output for single objective function using the Tchebycheff
//! approach.
inline double MOEAD::DecomposedSingleObjective(const arma::vec& weights,
    const arma::vec& idealPoint,
    const arma::vec& candidateFval)
{ //FIXME: weights[i] == 0? 1e-4 : weights[i]
  //TODO: Add more methods perhaps? (BI, TCHEBYCHEFF, WEIGHTED)
  return arma::min(weights % arma::abs(candidateFval - idealPoint));
}

inline bool MOEAD::Dominates(const arma::vec& first,
    const arma::vec& second)
{
  int flag = 0;
  for (size_t i = 0; i < numObjectives; ++i)
  {
    if (first(i) > second(i))
      return false;

    if(first(i) < second(i))
      flag = 1;
  }
  if (flag)
    return true;

  return false;
}

//! No objectives to evaluate.
template<std::size_t I,
  typename MatType,
  typename ...ArbitraryFunctionType>
  typename std::enable_if<I == sizeof...(ArbitraryFunctionType), void>::type
MOEAD::EvaluateObjectives(
    std::vector<MatType>&,
    std::tuple<ArbitraryFunctionType...>&,
    arma::mat &)
{
  // Nothing to do here.
}

//! Evaluate the objectives for the entire population.
template<std::size_t I,
  typename MatType,
  typename ...ArbitraryFunctionType>
  typename std::enable_if<I < sizeof...(ArbitraryFunctionType), void>::type
MOEAD::EvaluateObjectives(
    std::vector<MatType>& population,
    std::tuple<ArbitraryFunctionType...>& objectives,
    arma::mat& calculatedObjectives)
{
  for (size_t i = 0; i < population.size(); ++i)
  {
    calculatedObjectives(I, i) = std::get<I>(objectives).Evaluate(population[i]);
    EvaluateObjectives<I + 1, MatType, ArbitraryFunctionType...>(population, objectives,
        calculatedObjectives);
  }
}
}

#endif

