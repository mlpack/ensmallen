/**
 * @file ranking_svm.hpp
 * @author Suvarsha Chennareddy
 *
 * Definition of the Ranking SVM.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CMAES_RANKING_SVM_HPP
#define ENSMALLEN_CMAES_RANKING_SVM_HPP

#include "rbf_kernel.hpp"

namespace ens {

/**
 * 
 * The ranking SVM class implements a rudimentary ranking SVM model (currently)
 * built for saACMES.
 * It uses the Augmented Lagrangian optimizer to find the optimal 
 * parameters (dual variables).
 * 
 * For more information, please refer to:
 * 
 * @code
 * @inproceedings{10.1145/775047.775067,
 *   author = {Joachims, Thorsten},
 *   title = {Optimizing Search Engines Using Clickthrough Data},
 *   year = {2002},
 *   isbn = {158113567X},
 *   publisher = {Association for Computing Machinery},
 *   address = {New York, NY, USA},
 *   url = {https://doi.org/10.1145/775047.775067},
 *   doi = {10.1145/775047.775067},
 *   pages = {133–142},
 *   numpages = {10},
 *   location = {Edmonton, Alberta, Canada},
 *   series = {KDD '02}}
 * @encode
 *
 * @code
 * @inproceedings{kuo2014large,
 *   title={Large-scale kernel ranksvm},
 *   author={Kuo, Tzu-Ming and Lee, Ching-Pei and Lin, Chih-Jen},
 *   booktitle={Proceedings of the 2014 SIAM international conference on data mining},
 *   pages={812--820},
 *   year={2014},
 *   organization={SIAM}}
 * @endcode
 * 
 * @tparam CoordinateType The type of data (coordinate vectors).
 */
template<typename CoordinateType = arma::mat>
class RankingSVM
{
 public:

  /**
   * Construct the Ranking SVM model with the given data. This involves
   * initializing an RBF Kernel Function with the inverse of the covariance 
   * matrix and sigma parameter.
   *
   * @param data Vector consisting of pairs of coordinates and their function
   *      values.
   * @param numTestingPoints Number of points that will be used to test the
   *      model.
   * @param covarianceMatrixInverse The inverse of the covaraince matrix used
   *      in the RBF Kernel.
   * @param sigma The sigma parameter used in the RBF Kernel.
   *
   */
  RankingSVM(const std::vector < std::pair < CoordinateType, double>>& data,
             const size_t numTestingPoints,
             const CoordinateType& covarianceMatrixInverse,
             typename CoordinateType::elem_type sigma);

 /**
  * Evaluate the Ranking SVM model at the given coordinates for a particular 
  * batch-size
  *
  * @param coordinates The function coordinates.
  * @param i The first function.
  * @param batchSize Number of points to process.
  */
  typename CoordinateType::elem_type Evaluate(const CoordinateType& coordinates,
                                              const size_t i, 
                                              const size_t batchSize);

 /**
  * Train the Ranking SVM with the given hyper-parameters.
  *
  * @param numTrainingPoints The number of training points that will
         be used to train the Ranking SVM.
  * @param costPow This hyper-parameter is used to compute the Ranking SVM
  *      constraint violation weights.
  */
  void Train(const size_t numTrainingPoints, const double costPow);

  //! Get the training and testing data.
  const std::vector < std::pair < CoordinateType, double>> Data() const
  { return data; }
  //! Modify the training and testing data.
  std::vector < std::pair < CoordinateType, double>>& Data()
  { return data; }

  //! Get the kernel function object.
  RBFKernel<CoordinateType> Kernel() const{ return kernel; }
  //! Modify the kernel function object.
  RBFKernel<CoordinateType>& Kernel() { return kernel; }

  //! Return the number of parameters (number of functions).
  size_t NumFunctions()
  { return parameters.n_elem; }

  //! Return the number of points used for testing.
  size_t NumTestingPoints()
  { return numTestingPoints; }

 private:
   //! The training and testing data
   std::vector < std::pair < CoordinateType, double>> data;

   //! The number of testing points.
   size_t numTestingPoints;

   //! The parameters of the Ranking SVM model.
   CoordinateType parameters;

   //! The RBF kernel function.
   RBFKernel<CoordinateType> kernel;

};

} // namespace ens

// Include implementation.
#include "ranking_svm_impl.hpp"

#endif
