/**
 * @file lrsdp_function.hpp
 * @author Ryan Curtin
 * @author Abhishek Laddha
 *
 * A class that represents the objective function which LRSDP optimizes.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SDP_LRSDP_FUNCTION_HPP
#define ENSMALLEN_SDP_LRSDP_FUNCTION_HPP

#include <ensmallen_bits/aug_lagrangian/aug_lagrangian.hpp>
#include "sdp.hpp"

namespace ens {

/**
 * The objective function that LRSDP is trying to optimize.
 *
 * Note: LRSDPfunction is designed and implemented to specifically work
 * with a combination of AugLagrangian and L-BFGS optimizer. Implemenation
 * of LRSDPFunction includes caching R * R^T matrix in order to avoid redundant
 * computations(Specifically with above two optimizers) of R * R^T matrix
 * through AugLagrangian optimizer call, as only L-BFGS takes part in updating
 * R(coordinates) matrix. So, be careful while using LRSDP with some other
 * optimizer. You may need to modify caching process of R * R^T matrix.
 * See EvaluateImpl() in lrsdp_function_impl.hpp for more details.
 */
template<typename SDPType>
class LRSDPFunction
{
 public:
  /**
   * Construct the LRSDPFunction from the given SDP.
   *
   * @param sdp
   * @param initialPoint
   */
  LRSDPFunction(const SDPType& sdp,
                const arma::Mat<typename SDPType::ElemType>& initialPoint);

  /**
   * Construct the LRSDPFunction with the given initial point and number of
   * constraints. Note n_cols of the initialPoint specifies the rank.
   *
   * Set the A_x, B_x, and C_x  matrices for each constraint using the A_x(),
   * B_x(), and C_x() functions, for x in {sparse, dense}.
   *
   * @param numSparseConstraints
   * @param numDenseConstraints
   * @param initialPoint
   */
  LRSDPFunction(const size_t numSparseConstraints,
                const size_t numDenseConstraints,
                const arma::Mat<typename SDPType::ElemType>& initialPoint);

  /**
   * Clean any memory associated with the LRSDPFunction.
   */
  ~LRSDPFunction();

  /**
   * Evaluate the objective function of the LRSDP (no constraints) at the given
   * coordinates.
   */
  template<typename MatType>
  typename MatType::elem_type Evaluate(const MatType& coordinates) const;

  /**
   * Evaluate the gradient of the LRSDP (no constraints) at the given
   * coordinates.
   */
  template<typename MatType, typename GradType>
  void Gradient(const MatType& coordinates, GradType& gradient) const;

  /**
   * Evaluate a particular constraint of the LRSDP at the given coordinates.
   */
  template<typename MatType>
  typename MatType::elem_type EvaluateConstraint(
      const size_t index,
      const MatType& coordinates) const;

  /**
   * Evaluate the gradient of a particular constraint of the LRSDP at the given
   * coordinates.
   */
  template<typename MatType, typename GradType>
  void GradientConstraint(const size_t index,
                          const MatType& coordinates,
                          GradType& gradient) const;

  //! Get the total number of constraints in the LRSDP.
  size_t NumConstraints() const { return sdp.NumConstraints(); }

  //! Get the initial point of the LRSDP.
  template<typename MatType = arma::mat>
  MatType GetInitialPoint() const
  {
    MatType result = arma::conv_to<MatType>::from(initialPoint);
    return result;
  }

  //! Return the SDP object representing the problem.
  const SDPType& SDP() const { return sdp; }

  //! Modify the SDP object representing the problem.
  SDPType& SDP() { return sdp; }

  //! Get R*R^T matrix.
  template<typename MatType>
  const MatType& RRT() const
  {
    return rrt.As<typename std::remove_reference<MatType>::type>();
  }

  //! Modify R*R^T matrix.
  template<typename MatType>
  MatType& RRT()
  {
    return rrt.As<typename std::remove_reference<MatType>::type>();
  }

  //! Get the Any object for rrt.
  Any& RRTAny() { return rrt; }

 private:
  //! SDP object representing the problem
  SDPType sdp;

  //! Initial point.
  arma::Mat<typename SDPType::ElemType> initialPoint;

  //! Cache R*R^T matrix.
  Any rrt;
};

// Declare specializations in lrsdp_function.cpp.
template<>
template<typename MatType>
inline typename MatType::elem_type
AugLagrangianFunction<LRSDPFunction<SDP<arma::sp_mat>>>::Evaluate(
    const MatType& coordinates) const;

template<>
template<typename MatType>
inline typename MatType::elem_type
AugLagrangianFunction<LRSDPFunction<SDP<arma::mat>>>::Evaluate(
    const MatType& coordinates) const;

template<>
template<typename MatType, typename GradType>
inline void AugLagrangianFunction<LRSDPFunction<SDP<arma::sp_mat>>>::Gradient(
    const MatType& coordinates,
    GradType& gradient) const;

template<>
template<typename MatType, typename GradType>
inline void AugLagrangianFunction<LRSDPFunction<SDP<arma::mat>>>::Gradient(
    const MatType& coordinates,
    GradType& gradient) const;

template<>
template<typename MatType>
inline typename MatType::elem_type
AugLagrangianFunction<LRSDPFunction<SDP<arma::sp_fmat>>>::Evaluate(
    const MatType& coordinates) const;

template<>
template<typename MatType>
inline typename MatType::elem_type
AugLagrangianFunction<LRSDPFunction<SDP<arma::fmat>>>::Evaluate(
    const MatType& coordinates) const;

template<>
template<typename MatType, typename GradType>
inline void AugLagrangianFunction<LRSDPFunction<SDP<arma::sp_fmat>>>::Gradient(
    const MatType& coordinates,
    GradType& gradient) const;

template<>
template<typename MatType, typename GradType>
inline void AugLagrangianFunction<LRSDPFunction<SDP<arma::fmat>>>::Gradient(
    const MatType& coordinates,
    GradType& gradient) const;

} // namespace ens

// Include implementation
#include "lrsdp_function_impl.hpp"

#endif // ENSMALLEN_SDP_LRSDP_FUNCTION_HPP
