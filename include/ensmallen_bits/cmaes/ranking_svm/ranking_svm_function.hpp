/**
 * @file ranking_svm_function.hpp
 * @author Suvarsha Chennareddy
 *
 * Definition of the dual problem of the hinge loss function with
 * L2-regularization.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_ACTIVE_CMAES_RANKING_SVM_FUNCTION_HPP
#define ENSMALLEN_ACTIVE_CMAES_RANKING_SVM_FUNCTION_HPP

#include "rbf_kernel.hpp"

namespace ens {

 /**
  * The dual problem of the Ranking SVM's hinge loss function with
  * L2-regularization. The parameters are the dual vaariables.
  *
  *
  * @tparam CoordinateType The type of data (coordinate vectors).
  */
  template<typename CoordinateType = arma::mat>
  class RankingSVMFunction
  {
  public:

  /**
   * Construct the Ranking SVM objective function.
   *
   * @param costPow Hyper-parameter used to compute the Ranking SVM
   *      constraint violation weights.
   * @param kernel RBF Kernel.
   */
  RankingSVMFunction(
      const std::vector < std::pair < CoordinateType, double>>& data,
      const double costPow,
      RBFKernel<CoordinateType>& kernel);

  /**
   * Evaluate the complete objective along with the gradient 
   * with the given parameters.
   *
   * @param parameters The parameters of the Ranking SVM.
   * @param gradient Gradient evaluated with the parameters.
   * @return The value of the objective function.
   */
  typename CoordinateType::elem_type EvaluateWithGradient(
      const CoordinateType& parameters,
      CoordinateType& gradient);

  /**
   * Evaluate the objective with the given parameters.
   *
   * @param parameters The parameters of the Ranking SVM.
  *  @param k The first function.
  *  @param batchSize Number of functions to process.
   */
   typename CoordinateType::elem_type Evaluate(
      const CoordinateType& parameters,
      const size_t k, const size_t batchSize);

  /**
   * Evaluate the complete objective with the given parameters.
   *
   * @param parameters The parameters of the Ranking SVM.
   */
  typename CoordinateType::elem_type Evaluate(
      const CoordinateType& parameters);

  /**
   * Evaluate the gradient with the given parameters.
   *
   * @param parameters The parameters of the Ranking SVM.
   * @param gradient Gradient evaluated with the parameters.
   */
  void Gradient(const CoordinateType& coordinates, CoordinateType& gradient);

  /**
   * Evaluate constraint i at the given parameters.
   *
   * @param i Index of the constraint.
   * @param parameters The parameters of the Ranking SVM.
   */
  typename CoordinateType::elem_type EvaluateConstraint(
      const size_t index,
      const CoordinateType& parameters);
  /**
   * Evaluate the gradient of constraint i at the given parameters.
   *
   * @param i Index of the constraint.
   * @param parameters The parameters of the Ranking SVM.
   * @param gradient Gradient of the i-th constraint at the given parameters.
   */
    void GradientConstraint(const size_t i,
      const CoordinateType& parameters,
      CoordinateType& gradient);


    //! Get the starting point.
    CoordinateType GetInitialPoint() const
    { return initialPoint; }

    //! Get the number of constraints.
    size_t NumConstraints() const
    { return data.size() - 1; }

    //! Return the number of functions (one less 
    //! than the number of training points)
    size_t NumFunctions()
    { return data.size() - 1; }

  private:

    //! The training points with their function values.
    std::vector < std::pair < CoordinateType, double>> data;

    //! The staring point (for the parameters).
    CoordinateType initialPoint;

    //! The RBF kernel. 
    RBFKernel<CoordinateType>& kernel;

    //! The constraint violation weights (upper bounds of parameters).
    arma::mat violationCosts;

    //! Kernel matrix used as cache.
    CoordinateType K;
  };

} // namespace ens

// Include implementation.
#include "ranking_svm_function_impl.hpp"

#endif
