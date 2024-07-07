/**
 * @file bipop_cmaes_impl.hpp
 * @author Benjami Parellada
 *
 * Implementation of the BIPOP Covariance Matrix Adaptation Evolution Strategy 
 * as proposed by N. Hansen in "Benchmarking a BI-population CMA-ES on the
 * BBOB-2009 function testbed".
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_CMAES_BIPOP_CMAES_IMPL_HPP
#define ENSMALLEN_CMAES_BIPOP_CMAES_IMPL_HPP

#include "bipop_cmaes.hpp"
#include <ensmallen_bits/function.hpp>
#include <random>

namespace ens {

template<typename CMAESType>
BIPOPCMAES<CMAESType>::BIPOPCMAES(const CMAESType& CMAES,
                                  const size_t maxRestarts) :
    cmaes(CMAES),
    maxRestarts(maxRestarts)
{ /* Nothing to do. */ }

template<typename CMAESType>
BIPOPCMAES<CMAESType>::BIPOPCMAES(const size_t lambda,
                                  const typename CMAESType::transformationPolicyType& transformationPolicy,
                                  const size_t batchSize,
                                  const size_t maxIterations,
                                  const double tolerance,
                                  const typename CMAESType::selectionPolicyType& selectionPolicy,
                                  double stepSize,
                                  const size_t maxRestarts) :
    cmaes(lambda, transformationPolicy, batchSize, maxIterations, tolerance, selectionPolicy, stepSize),
    maxRestarts(maxRestarts)
{ /* Nothing to do. */  }

template<typename CMAESType>
template<typename SeparableFunctionType, typename MatType, typename... CallbackTypes>
typename MatType::elem_type BIPOPCMAES<CMAESType>::Optimize(
    SeparableFunctionType& function,
    MatType& iterateIn,
    CallbackTypes&&... callbacks)
{
    StoreBestCoordinates<MatType> sbc;
    size_t totalFunctionEvaluations = 0;
    size_t largePopulationBudget = 0;
    size_t smallPopulationBudget = 0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // First single run with default population size
    MatType iterate = iterateIn;
    cmaes.Optimize(function, iterate, sbc, callbacks...);

    size_t defaultLambda = cmaes.PopulationSize();
    size_t currentLargeLambda = defaultLambda;

    for (size_t i = 0; i < maxRestarts; ++i)
    {
        if (largePopulationBudget <= smallPopulationBudget || i == 0 || i == maxRestarts - 1)
        {
            // Large population regime
            currentLargeLambda *= 2;
            cmaes.PopulationSize() = currentLargeLambda;
            cmaes.StepSize() = 2.0;

            iterate = iterateIn;
            cmaes.Optimize(function, iterate, sbc, callbacks...);
            
            size_t evaluations = cmaes.FunctionEvaluations();
            totalFunctionEvaluations += evaluations;
            largePopulationBudget += evaluations;
        }
        else
        {
            // Small population regime
            double u = dis(gen);
            size_t smallLambda = static_cast<size_t>(defaultLambda * std::pow(0.5 * currentLargeLambda / defaultLambda, u * u));
            cmaes.PopulationSize() = smallLambda;
            cmaes.StepSize() = 2 * std::pow(10, -2*dis(gen));

            iterate = iterateIn;
            cmaes.Optimize(function, iterate, sbc, callbacks...);
            
            size_t evaluations = cmaes.FunctionEvaluations();
            totalFunctionEvaluations += evaluations;
            smallPopulationBudget += evaluations;
        }

        // Check if the total number of evaluations has exceeded the limit
        if (totalFunctionEvaluations >= 1e9) {
            break;
        }
    }

    // Store the best coordinates
    iterateIn = sbc.BestCoordinates();
    // Return the best objective
    return sbc.BestObjective();
}

} // namespace ens

#endif
