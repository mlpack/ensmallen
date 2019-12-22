/**
 * @file nsga2.hpp
 * @author Sayan Goswami
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_NSGA2_NSGA2_HPP
#define ENSMALLEN_NSGA2_NSGA2_HPP

#include <vector>

namespace ens {

class NSGA2 {
 public:
  NSGA2(const size_t populationSize = 100,
        const size_t maxGenerations = 2000,
        const double crossoverProb = 0.6,
        const double epsilon = 1e-6);

  template<typename ArbitraryFunctionType,
           typename MatType,
           typename... CallbackTypes>
  typename MatType::elem_type Optimize(std::vector<ArbitraryFunctionType>& objectives,
                                       MatType& iterate,
                                       CallbackTypes&&... callbacks);

  size_t PopulationSize() const { return populationSize; }
  size_t& PopulationSize() { return populationSize; }

  size_t MaxGenerationSize() const { return maxGenerations; }
  size_t& MaxGenerationSize() { return maxGenerations; }

  double CrossoverProb() const { return crossoverProb; }
  double& CrossoverProb() { return crossoverProb; }

  double Epsilon() const { return epsilon; }
  double& Epsilon() { return epsilon; }


 private:
  template<typename MatType>
  void BinaryTournamentSelection(std::vector<MatType>& population);

  template<typename MatType>
  void Crossover(std::vector<MatType>& population,
                 const size_t parentA,
                 const size_t parentB);

  template<typename MatType>
  void Mutate(std::vector<MatType>& population);

  size_t populationSize;
  size_t maxGenerations;
  double crossoverProb;
  double epsilon;
};

} // namespace ens

// Include implementation.
#include "nsga2_impl.hpp"

#endif
