/**
 * @file ranking_svm_impl.hpp
 * @author Suvarsha Chennareddy
 *
 * Implementation of the Ranking SVM.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CMAES_RANKING_SVM_IMPL_HPP
#define ENSMALLEN_CMAES_RANKING_SVM_IMPL_HPP

#include "ranking_svm.hpp"

#include "ranking_svm_function.hpp"
#include <ensmallen_bits/aug_lagrangian/aug_lagrangian.hpp>

 /*
  Function used to sort and extract the latest points (with an offset from end)
  from the given training and testing data vector. These data points will be
  used to train the model.
 */
template<typename CoordinateType>
std::vector<std::pair<CoordinateType, double>> getSortedLastPoints(
  std::vector<std::pair<CoordinateType, double>>& inputVector,
  size_t subvectorSize, size_t offset) {

  // Check if the subvector size is valid
  if (subvectorSize <= 0 || subvectorSize > inputVector.size()) {
    std::cout << subvectorSize << std::endl;
    std::cerr << "Invalid subvector size." << std::endl;
    return {};
  }

  std::sort(inputVector.end() - subvectorSize - offset, 
    inputVector.end() - offset,
    [](const std::pair<CoordinateType, double>& a, 
      const std::pair<CoordinateType, double>& b) {
      return a.second < b.second;
    });

  std::vector<std::pair<CoordinateType, double>> subvector(
    inputVector.end() - subvectorSize - offset, inputVector.end() - offset);

  return subvector;
}


namespace ens {

template<typename CoordinateType>
RankingSVM<CoordinateType>::RankingSVM(
          const std::vector < std::pair < CoordinateType, double>>& data,
          const size_t numTestingPoints,
          const CoordinateType& covarianceMatrixInverse,
          typename CoordinateType::elem_type sigma):
  data(data),
  numTestingPoints(numTestingPoints),
  kernel(covarianceMatrixInverse, sigma)
{ /* Nothing to do */ }


template<typename CoordinateType>
typename CoordinateType::elem_type RankingSVM<CoordinateType>::Evaluate(
           const CoordinateType& coordinates,
           const size_t i, const size_t batchSize)
{

  typename CoordinateType::elem_type objective = 0;

  for (size_t j = i; j < i + batchSize; ++j)
  {
    objective += parameters(j) * (
      kernel.Evaluate(coordinates, 
        data[data.size() - parameters.n_elem - numTestingPoints + j].first) - 
      kernel.Evaluate(coordinates, 
        data[data.size() - parameters.n_elem - numTestingPoints + j - 1].first));
  }

  return objective;
}


template<typename CoordinateType>
void RankingSVM<CoordinateType>::Train(const size_t numTrainingPoints, 
                                       const double costPow)
{

  // Extract the training data.
  std::vector < std::pair < CoordinateType, double>>  trainingData =
    getSortedLastPoints(data, numTrainingPoints, numTestingPoints);

  // Initialize the Ranking SVM function with the hyper-parameters and 
  // reference to kernel.
  RankingSVMFunction<CoordinateType> rsvmf(trainingData, costPow, kernel);
  
  parameters = rsvmf.GetInitialPoint();
  
  // Train the model using the Augmented Lagrangian optimizer.
  AugLagrangian s;
  if (!s.Optimize(rsvmf, parameters))
    Warn << "The Ranking SVM couldn't be trained." << std::endl;
}

} // namespace ens


#endif
