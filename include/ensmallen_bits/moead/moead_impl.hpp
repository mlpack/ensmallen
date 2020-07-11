#include "moead.hpp"

namespace ens {

inline MOEAD::MOEAD(const size_t populationSize,
                    const size_t maxGenerations,
                    const double crossoverProb,
                    const double mutationProb,
                    const double mutationStrength,
                    const double epsilon,
                    const size_t numWeights,
                    const size_t T) :
    populationSize(populationSize),
    maxGenerations(maxGenerations),
    crossoverProb(crossoverProb),
    mutationProb(mutationProb),
    mutationStrength(mutationStrength),
    epsilon(epsilon),
    T(T)
  { /* Nothing to do here. */ }

template<typename MatType,
         typename... ArbitraryFunctionType>
arma::vec<MatType> MOEAD::Optimize(std::tuple<ArbitraryFunctionType...>& objectives,
                                   MatType& iterate)
{
  // Number of objective functions. Represented by m in the paper.
  numObjectives = sizeof...(ArbitraryFunctionType);
  // The external population, non-dominated solutions.
  arma::vec<MatType> externalPopulation;
  // Weight vectors, where each one of them represents a decomposition.
  arma::mat weights(numWeights, numObjectives, fill::randu);
  // Storing the T nearest neighbours of each weight vector.
  arma::mat<size_t> B(numWeights, T);
  for(int i = 0; i < numWeights; i++)
  {
    arma::vec distances(numWeights);
    for(int j = 0; j < numWeights; j++)
    {
      double distance = 0;
      for(int w = 0; w < numObjectives ; w++)
      {
        distance += std::pow(weights[i][w] - weights[j][w], 2);
      }
      distance = std::sqrt(distance);
      distances[w]=distance;
    }
    for(int iter = 1; iter<=T; iter++)
      B(i, iter) = arma::stable_sort_index(distances)(iter);
  }
  // Random generation of the initial population.
  arma::vec<MatType> population(populationSize);
  for(size_t i = 0; i < populationSize; i++)
  {
    population(i) = arma::randu<MatType>(iterate.n_rows, iterate.n_cols) - 0.5 + iterate;
  }
  arma::mat FValue(populationSize, numObjectives);

}
