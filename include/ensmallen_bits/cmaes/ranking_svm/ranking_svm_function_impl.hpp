/**
 * @file ranking_svm_function_impl.hpp
 * @author Suvarsha Chennareddy
 *
 * Implementation of the dual problem of the hinge loss function with 
 * L2-regularization.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CMAES_RANKING_SVM_FUNCTION_IMPL_HPP
#define ENSMALLEN_CMAES_RANKING_SVM_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "ranking_svm_function.hpp"

namespace ens {

template<typename CoordinateType>
RankingSVMFunction<CoordinateType>::RankingSVMFunction(
    const std::vector < std::pair < CoordinateType, double>>& data,
    const double costPow,
    RBFKernel<CoordinateType>& kernel) :
    data(data),
    kernel(kernel)
  {
    K.set_size(data.size(), data.size());
    K.fill(0);

    // Initialize the Ranking SVM constraint violation weights and
    // the initial point (for the parameters).
    violationCosts.set_size(data.size() - 1, 1);
    initialPoint.set_size(data.size() - 1, 1);
    for (size_t i = 0; i < data.size() - 1; i++) {
      violationCosts(i) = std::pow((data.size() - (i + 1)), costPow);
      initialPoint(i) = violationCosts(i) / 2;
    }
  }

template<typename CoordinateType>
typename CoordinateType::elem_type 
RankingSVMFunction<CoordinateType>::EvaluateWithGradient(
    const CoordinateType& parameters,
    CoordinateType& gradient)
  {
    typename CoordinateType::elem_type objective = 0;

    for (size_t i = 0; i < data.size() - 1; i++) {

      gradient(i) = 0;

      for (size_t j = 0; j < data.size() - 1; j++) {

        // Check if the kernel matrix entries are null. If so, compute them.
        if (K(i + 1, j + 1) == 0) {
          if (K(j + 1, i + 1) == 0)
            K(i + 1, j + 1) = kernel.Evaluate(data[i + 1].first, data[j + 1].first);
          else
            K(i + 1, j + 1) = K(j + 1, i + 1);
        }

        if (K(i, j) == 0) {
          if (K(j, i) == 0)
            K(i, j) = kernel.Evaluate(data[i].first, data[j].first);
          else
            K(i, j) = K(j, i);
        }

        if (K(i + 1, j) == 0) {
          if (K(j, i + 1) == 0)
            K(i + 1, j) = kernel.Evaluate(data[i + 1].first, data[j].first);
          else
            K(i + 1, j) = K(j, i + 1);
        }
        if (K(i, j + 1) == 0) {
          if (K(j + 1, i) == 0)
            K(i, j + 1) = kernel.Evaluate(data[i].first, data[j + 1].first);
          else
            K(i, j + 1) = K(j + 1, i);
        }

        gradient(i) += parameters(j) * ( K(i + 1, j + 1) + K(i, j) - 
          K(i + 1, j) - K(i, j + 1));
      }

      objective += parameters(i) * (0.5 * gradient(i) - 1);
      gradient(i) -= 1;

    }
    return objective;
  }

template<typename CoordinateType>
typename CoordinateType::elem_type RankingSVMFunction<CoordinateType>::Evaluate(
    const CoordinateType& parameters,
    const size_t k, const size_t batchSize)
{
  typename CoordinateType::elem_type objective = 0;

  for (size_t i = k; i < k + batchSize; i++) {

    typename CoordinateType::elem_type temp = 0;

    for (size_t j = 0; j < data.size() - 1; j++) {

      // Check if the kernel matrix entries are null. If so, compute them.
      if (K(i + 1, j + 1) == 0) {
        if (K(j + 1, i + 1) == 0)
          K(i + 1, j + 1) = kernel.Evaluate(data[i + 1].first, data[j + 1].first);
        else
          K(i + 1, j + 1) = K(j + 1, i + 1);
      }

      if (K(i, j) == 0) {
        if (K(j, i) == 0)
          K(i, j) = kernel.Evaluate(data[i].first, data[j].first);
        else
          K(i, j) = K(j, i);
      }

      if (K(i + 1, j) == 0) {
        if (K(j, i + 1) == 0)
          K(i + 1, j) = kernel.Evaluate(data[i + 1].first, data[j].first);
        else
          K(i + 1, j) = K(j, i + 1);
      }
      if (K(i, j + 1) == 0) {
        if (K(j + 1, i) == 0)
          K(i, j + 1) = kernel.Evaluate(data[i].first, data[j + 1].first);
        else
          K(i, j + 1) = K(j + 1, i);
      }

      temp += parameters(j) * (K(i + 1, j + 1) + K(i, j) -
        K(i + 1, j) - K(i, j + 1));
    }

    objective += parameters(i) * (0.5 * temp - 1);
  }

  return objective;
}

template<typename CoordinateType>
typename CoordinateType::elem_type RankingSVMFunction<CoordinateType>::Evaluate(
  const CoordinateType& parameters)
{
  return Evaluate(parameters, 0, data.size() - 1);
}


template<typename CoordinateType>
void RankingSVMFunction<CoordinateType>::Gradient(
    const CoordinateType & parameters, CoordinateType & gradient)
{
  for (size_t i = 0; i < data.size() - 1; i++) {

    gradient(i) = -1;

    for (size_t j = 0; j < data.size() - 1; j++) {

      // Check if the kernel matrix entries are null. If so, compute them.
      if (K(i + 1, j + 1) == 0) {
        if (K(j + 1, i + 1) == 0)
          K(i + 1, j + 1) = kernel.Evaluate(data[i + 1].first, data[j + 1].first);
        else
          K(i + 1, j + 1) = K(j + 1, i + 1);
      }

      if (K(i, j) == 0) {
        if (K(j, i) == 0)
          K(i, j) = kernel.Evaluate(data[i].first, data[j].first);
        else
          K(i, j) = K(j, i);
      }

      if (K(i + 1, j) == 0) {
        if (K(j, i + 1) == 0)
          K(i + 1, j) = kernel.Evaluate(data[i + 1].first, data[j].first);
        else
          K(i + 1, j) = K(j, i + 1);
      }
      if (K(i, j + 1) == 0) {
        if (K(j + 1, i) == 0)
          K(i, j + 1) = kernel.Evaluate(data[i].first, data[j + 1].first);
        else
          K(i, j + 1) = K(j + 1, i);
      }

      gradient(i) += parameters(j) * (K(i + 1, j + 1) + K(i, j) -
        K(i + 1, j) - K(i, j + 1));
    }
  }
}

template<typename CoordinateType>
typename CoordinateType::elem_type 
RankingSVMFunction<CoordinateType>::EvaluateConstraint(
    const size_t i,
    const CoordinateType& parameters)
{

  // Constraint i: 0 <= parameter_i <= violationCost_i
  if (parameters(i) > violationCosts(i))
    return parameters(i) - violationCosts(i);
  else if (parameters(i) < 0)
    return -parameters(i);
  else
    return 0;
}

template<typename CoordinateType>
void RankingSVMFunction<CoordinateType>::GradientConstraint(
    const size_t i,
    const CoordinateType& parameters,
    CoordinateType& gradient)
{
  gradient.zeros(data.size() - 1);
  if (parameters(i) > violationCosts(i))
    gradient(i) = 1;
  else if(parameters(i) < 0)
    gradient(i) = -1;
  else
    gradient(i) = 0;
}

} // namespace ens

#endif
