/**
 * @file surrogate_error.hpp
 * @author Suvarsha Chennareddy
 *
 * Compute the error of the surrogate (Ranking SVM) model.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CMAES_SURROGATE_ERROR_HPP
#define ENSMALLEN_CMAES_SURROGATE_ERROR_HPP

#include "ranking_svm.hpp"

namespace ens {

/**
 * Compute the error of the given Ranking SVM model
 * using the testing points.
 *
 * @tparam CoordinateType The type of data (coordinate vectors).
 */
  template<typename CoordinateType = arma::mat>
  class SurrogateError
  {
  public:

  /**
   * Construct the Surrogate Error Function with a reference to the 
   * surrogate (Ranking SVM) model.
   *
   * @param rsvm Reference to the Ranking SVM model being used as the
   *      surrogate.
   */
  SurrogateError(RankingSVM<CoordinateType>& rsvm) :
    rsvm(rsvm)
  {/* Nothing to do */}


  /**
   * Evaluate the surrogate error with the given hyper-parameters.
   *
   * @param hyperparameters The surrogate hyper-parameters.
   * @param i The index of the first testing point.
   * @param batchSize Number of testing points to process.
   */
    typename CoordinateType::elem_type Evaluate(
      const CoordinateType& hyperparameters,
      const size_t i,
      const size_t batchSize)
    {
      // Train the model with the given hyper-parameters.
      if (i == 0) {
        rsvm.Train((size_t) hyperparameters(0), hyperparameters(1));
      }

      // Check if the ranking of each testing point is correct.
      typename CoordinateType::elem_type objective = 0;
      for (size_t j = i; j < i + batchSize; j++) {

        // Compare two consecutive testing points and check their relative 
        // rankings.
        std::pair <CoordinateType, double> t1 = 
          rsvm.Data()[rsvm.Data().size() - rsvm.NumTestingPoints() + j];
        std::pair <CoordinateType, double> t2 =
          rsvm.Data()[rsvm.Data().size() - rsvm.NumTestingPoints() + j + 1];

        typename CoordinateType::elem_type e1 =
          rsvm.Evaluate(t1.first, 0, rsvm.NumFunctions());

        typename CoordinateType::elem_type e2 =
          rsvm.Evaluate(t2.first, 0, rsvm.NumFunctions());

        // If the ranking is incorrect, increment the objective (error).
        if (((e1 > e2) && (t1.second < t2.second)) ||
          ((e1 < e2) && (t1.second > t2.second)))
          ++objective;
      
      }
      
      return objective / NumFunctions();
    }
    
    //@ Return the Rranking SVM model used.
    RankingSVM<CoordinateType>& RSVM()
    { return rsvm; }

    //! Return the number of functions (one less than the 
    //! number of testing points)
    const size_t NumFunctions() const
    { return rsvm.NumTestingPoints() - 1; }

  private:

    //! The Ranking SVM.
    RankingSVM<CoordinateType>& rsvm;

  };

} // namespace ens

#endif