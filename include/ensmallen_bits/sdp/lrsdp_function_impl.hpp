/**
 * @file lrsdp_function.cpp
 * @author Ryan Curtin
 * @author Abhishek Laddha
 *
 * Implementation of the LRSDPFunction class, and also template specializations
 * for faster execution with the AugLagrangian optimizer.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SDP_LRSDP_FUNCTION_IMPL_HPP
#define ENSMALLEN_SDP_LRSDP_FUNCTION_IMPL_HPP

#include "lrsdp_function.hpp"

namespace ens {

template <typename SDPType>
LRSDPFunction<SDPType>::LRSDPFunction(
    const SDPType& sdp,
    const arma::Mat<typename SDPType::ElemType>& initialPoint):
    sdp(sdp),
    initialPoint(initialPoint)
{
  if (initialPoint.n_rows < initialPoint.n_cols)
  {
    Warn << "LRSDPFunction::LRSDPFunction(): solution matrix will have "
        << "more columns than rows.  It may be more efficient to find the "
        << "transposed solution." << std::endl;
  }
}

template<typename SDPType>
LRSDPFunction<SDPType>::LRSDPFunction(
    const size_t numSparseConstraints,
    const size_t numDenseConstraints,
    const arma::Mat<typename SDPType::ElemType>& initialPoint):
    sdp(initialPoint.n_rows, numSparseConstraints, numDenseConstraints),
    initialPoint(initialPoint)
{
  if (initialPoint.n_rows < initialPoint.n_cols)
  {
    Warn << "LRSDPFunction::LRSDPFunction(): solution matrix will have "
        << "more columns than rows.  It may be more efficient to find the "
        << "transposed solution." << std::endl;
  }
}

template<typename SDPType>
LRSDPFunction<SDPType>::~LRSDPFunction()
{
  rrt.Clean();
}

template<typename SDPType>
template<typename MatType>
typename MatType::elem_type LRSDPFunction<SDPType>::Evaluate(
    const MatType& /* coordinates */) const
{
  // Note: We don't require to update the R*R^T matrix here as the current
  // function is only used by AugLagrangian, which do not update the coordinates
  // matrix.
  return arma::accu(SDP().C() % rrt.As<MatType>());
}

template<typename SDPType>
template<typename MatType, typename GradType>
void LRSDPFunction<SDPType>::Gradient(const MatType& /* coordinates */,
                                      GradType& /* gradient */) const
{
  throw std::logic_error("LRSDPFunction::Gradient() not implemented for "
         "arbitrary optimizers!");
}

template<typename SDPType>
template<typename MatType>
typename MatType::elem_type LRSDPFunction<SDPType>::EvaluateConstraint(
    const size_t index,
    const MatType& coordinates) const
{
  // Note: We don't require to update the R*R^T matrix here as the current
  // function is only used by AugLagrangian, which do not update the coordinates
  // matrix.

  // Using cached R*R^T gives better optimization for sparse matrices.
  if (index < SDP().NumSparseConstraints())
  {
    return accu(SDP().SparseA()[index] % rrt.As<MatType>()) -
        SDP().SparseB()[index];
  }
  const size_t index1 = index - SDP().NumSparseConstraints();

  // For computation optimization we will be taking R^T * A first.
  return trace((trans(coordinates) * SDP().DenseA()[index1]) * coordinates)
                 - SDP().DenseB()[index1];
}

template<typename SDPType>
template<typename MatType, typename GradType>
void LRSDPFunction<SDPType>::GradientConstraint(
    const size_t /* index */,
    const MatType& /* coordinates */,
    GradType& /* gradient */) const
{
  throw std::logic_error("LRSDPFunction::GradientConstraint() not implemented "
         "for arbitrary optimizers!");
}

//! Utility function for updating R*R^T matrix.
//! Note: Caching R*R^T provide significant computation optimization
//! by reducing redundant R*R^T calculations in case of functions are not used
//! updating coordinates matrix, hence leaving R*R^T unchanged.
template<typename SDPType, typename MatType>
void UpdateRRT(LRSDPFunction<SDPType>& function,
               MatType&& newrrt)
{
  function.template RRT<MatType>() = std::move(newrrt);
}

//! Utility function for calculating part of the objective when AugLagrangian is
//! used with an LRSDPFunction.
template <typename MatrixType, typename VecType, typename MatType>
static inline void
UpdateObjective(typename MatType::elem_type& objective,
                const MatType& rrt,
                const std::vector<MatrixType>& ais,
                const VecType& bis,
                const arma::vec& lambda,
                const size_t lambdaOffset,
                const double sigma)
{
  for (size_t i = 0; i < ais.size(); ++i)
  {
    // Take the trace subtracted by the b_i.
    // Here taking R^T * A first is not recommended as we are already
    // using pre-computed R * R^T. Taking R^T * A first will result in increase
    // in number of computations.
    const double constraint = arma::accu(ais[i] % rrt) - bis[i];
    objective -= (lambda[lambdaOffset + i] * constraint);
    objective += (sigma / 2.) * constraint * constraint;
  }
}

//! Utility function for calculating part of the gradient when AugLagrangian is
//! used with an LRSDPFunction.
template <typename MatrixType, typename VecType, typename MatType>
static inline void
UpdateGradient(MatType& s,
               const MatType& rrt,
               const std::vector<MatrixType>& ais,
               const VecType& bis,
               const arma::vec& lambda,
               const size_t lambdaOffset,
               const double sigma)
{
  for (size_t i = 0; i < ais.size(); ++i)
  {
    // Here taking R^T * A first is not recommended as we are already
    // using pre-computed R * R^T. Taking R^T * A first will result in increase
    // in number of computations.
    const double constraint = arma::accu(ais[i] % rrt) - bis[i];
    const double y = lambda[lambdaOffset + i] - sigma * constraint;
    s -= y * ais[i];
  }
}

template<typename SDPType, typename MatType>
static inline double
EvaluateImpl(LRSDPFunction<SDPType>& function,
             const MatType& coordinates,
             const arma::vec& lambda,
             const double sigma)
{
  // We can calculate the entire objective in a smart way.
  // L(R, y, s) = Tr(C * (R R^T)) -
  //     sum_{i = 1}^{m} (y_i (Tr(A_i * (R R^T)) - b_i)) +
  //     (sigma / 2) * sum_{i = 1}^{m} (Tr(A_i * (R R^T)) - b_i)^2

  // Let's start with the objective: Tr(C * (R R^T)).
  // Simple, possibly slow solution-- see below for optimization opportunity
  //
  // Note that Tr(C^T * (R R^T)) = Tr( (CR)^T * R ), so
  // multiplying C * R first, and then taking the trace dot should be more
  // memory efficient.
  //
  // Similarly for the constraints, taking R^T * A first should be
  // more efficient.
  //
  // For computation optimization we will be taking R^T * C first.
  // Objective function = Tr((R^T * C) * R)

  // Calculate R*R^T for updating cache.
  MatType rrt = coordinates * trans(coordinates);

  // Update R*R^T matrix.
  // Note that we can only use this optimization in case of L-BFGS optimizer
  // or any other similar optimizer which calls Evaluate() before Gradient()
  // with same coordinates matrix and uses only Evaluate() to update
  // coordinates matrix.

  // Note: In case optimizer also uses Gradient() for updating coordinates
  // matrix than the same line of code can be used to update R*R^T through
  // Gradient().
  UpdateRRT(function, rrt);

  // Optimized objective function.
  typename MatType::elem_type objective =
      trace((trans(coordinates) * function.SDP().C()) * coordinates);

  // Now each constraint.
  UpdateObjective(objective, function.template RRT<MatType>(),
      function.SDP().SparseA(), function.SDP().SparseB(), lambda, 0, sigma);
  UpdateObjective(objective, function.template RRT<MatType>(),
      function.SDP().DenseA(), function.SDP().DenseB(), lambda,
      function.SDP().NumSparseConstraints(), sigma);

  return objective;
}

template<typename SDPType, typename MatType, typename GradType>
static inline void
GradientImpl(const LRSDPFunction<SDPType>& function,
             const MatType& coordinates,
             const arma::vec& lambda,
             const double sigma,
             GradType& gradient)
{
  // We can calculate the gradient in a smart way.
  // L'(R, y, s) = 2 * S' * R
  //   with
  // S' = C - sum_{i = 1}^{m} y'_i A_i
  // y'_i = y_i - sigma * (Trace(A_i * (R R^T)) - b_i)

  // Directly retrieve R*R^T from cache.
  const MatType& rrt = function.template RRT<MatType>();
  MatType s(function.SDP().C());

  UpdateGradient(
      s, rrt, function.SDP().SparseA(), function.SDP().SparseB(),
      lambda, 0, sigma);
  UpdateGradient(
      s, rrt, function.SDP().DenseA(), function.SDP().DenseB(),
      lambda, function.SDP().NumSparseConstraints(), sigma);

  gradient = 2 * s * coordinates;
}

// Template specializations for function and gradient evaluation.
// Note that C++ does not allow partial specialization of class members,
// so we have to go about this in a somewhat round-about way.
template<>
template<typename MatType>
inline typename MatType::elem_type
AugLagrangianFunction<LRSDPFunction<SDP<arma::sp_mat>>>::Evaluate(
    const MatType& coordinates) const
{
  return EvaluateImpl(function, coordinates, lambda, sigma);
}

template<>
template<typename MatType>
inline typename MatType::elem_type
AugLagrangianFunction<LRSDPFunction<SDP<arma::mat>>>::Evaluate(
    const MatType& coordinates) const
{
  return EvaluateImpl(function, coordinates, lambda, sigma);
}

template<>
template<typename MatType, typename GradType>
inline void AugLagrangianFunction<LRSDPFunction<SDP<arma::sp_mat>>>::Gradient(
    const MatType& coordinates,
    GradType& gradient) const
{
  GradientImpl(function, coordinates, lambda, sigma, gradient);
}

template<>
template<typename MatType, typename GradType>
inline void AugLagrangianFunction<LRSDPFunction<SDP<arma::mat>>>::Gradient(
    const MatType& coordinates,
    GradType& gradient) const
{
  GradientImpl(function, coordinates, lambda, sigma, gradient);
}

template<>
template<typename MatType>
inline typename MatType::elem_type
AugLagrangianFunction<LRSDPFunction<SDP<arma::sp_fmat>>>::Evaluate(
    const MatType& coordinates) const
{
  return EvaluateImpl(function, coordinates, lambda, sigma);
}

template<>
template<typename MatType>
inline typename MatType::elem_type
AugLagrangianFunction<LRSDPFunction<SDP<arma::fmat>>>::Evaluate(
    const MatType& coordinates) const
{
  return EvaluateImpl(function, coordinates, lambda, sigma);
}

template<>
template<typename MatType, typename GradType>
inline void AugLagrangianFunction<LRSDPFunction<SDP<arma::sp_fmat>>>::Gradient(
    const MatType& coordinates,
    GradType& gradient) const
{
  GradientImpl(function, coordinates, lambda, sigma, gradient);
}

template<>
template<typename MatType, typename GradType>
inline void AugLagrangianFunction<LRSDPFunction<SDP<arma::fmat>>>::Gradient(
    const MatType& coordinates,
    GradType& gradient) const
{
  GradientImpl(function, coordinates, lambda, sigma, gradient);
}

} // namespace ens

#endif
