/**
 * @file sdp.hpp
 * @author Stephen Tu
 *
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SDP_SDP_HPP
#define ENSMALLEN_SDP_SDP_HPP

namespace ens {

/**
 * Specify an SDP in primal form
 *
 *     min    dot(C, X)
 *     s.t.   dot(Ai, X) = bi, i=1,...,m, X >= 0
 *
 * This representation allows the constraint matrices Ai to be specified as
 * either dense matrices (arma::mat) or sparse matrices (arma::sp_mat).  After
 * initializing the SDP object, you will need to set the constraints yourself,
 * via the SparseA(), SparseB(), DenseA(), DenseB(), and C() functions.  Note
 * that for each matrix you add to either SparseA() or DenseA(), you must add
 * the corresponding b value to the corresponding vector SparseB() or DenseB().
 *
 * The objective matrix (C) may be stored as either dense or sparse depending on
 * the ObjectiveMatrixType parameter.
 *
 * @tparam ObjectiveMatrixType Should be either arma::mat or arma::sp_mat.
 */
template<typename ObjectiveMatrixType,
         typename DenseConstraintMatrixType =
             arma::Mat<typename ObjectiveMatrixType::elem_type>,
         typename SparseConstraintMatrixType =
             arma::SpMat<typename ObjectiveMatrixType::elem_type>,
         typename BVectorType =
             arma::Col<typename ObjectiveMatrixType::elem_type>>
class SDP
{
 public:
  //! Type of objective matrix.
  typedef ObjectiveMatrixType ObjectiveType;
  //! Type of element held by the SDP.
  typedef typename ObjectiveMatrixType::elem_type ElemType;
  //! Type of dense constraints.
  typedef DenseConstraintMatrixType DenseConstraintType;
  //! Type of sparse constraints.
  typedef SparseConstraintMatrixType SparseConstraintType;
  //! Type of B values.
  typedef BVectorType BType;

  /**
   * Initialize this SDP to an empty state.  To add constraints, you will have
   * to modify the constraints via the SparseA(), DenseA(), SparseB(), DenseB(),
   * and C() functions.  For the sake of speed, there is no error checking, so
   * if you specify an invalid SDP, whatever solver you use will gladly try to
   * solve it!  (And it will probably crash horribly.)
   */
  SDP();

  /**
   * Initialize this SDP to one which structurally has size n.  To set the
   * constraints you will still need to access through SparseA(), DenseA(),
   * SparseB(), DenseB(), and C().  Consider using move semantics to keep things
   * fast.  As with the previous constructor, there is no error checking for the
   * sake of speed, so if you build an invalid SDP, whatever solver you use will
   * gladly try to solve it!  (And it will probably crash horribly.)
   *
   * @param n Number of rows (and columns) in the objective matrix C.
   * @param numSparseConstraints Number of sparse constraints.
   * @param numDenseConstraints Number of dense constraints.
   */
  SDP(const size_t n,
      const size_t numSparseConstraints,
      const size_t numDenseConstraints);

  //! Return number of rows and columns in the objective matrix C.
  size_t N() const { return c.n_rows; }

  size_t N2bar() const { return N() * (N() + 1) / 2; }

  //! Return the number of sparse constraints (constraints with sparse Ai) in
  //! the SDP.
  size_t NumSparseConstraints() const { return sparseB.n_elem; }
  //! Return the number of dense constraints (constraints with dense Ai) in the
  //! SDP.
  size_t NumDenseConstraints() const { return denseB.n_elem; }

  //! Return the total number of constraints in the SDP.
  size_t NumConstraints() const { return sparseB.n_elem + denseB.n_elem; }

  //! Modify the sparse objective function matrix (sparseC).
  ObjectiveMatrixType& C() { return c; }
  //! Return the sparse objective function matrix (sparseC).
  const ObjectiveMatrixType& C() const { return c; }

  //! Return the vector of sparse A matrices (which correspond to the sparse
  //! constraints).
  const std::vector<SparseConstraintMatrixType>& SparseA() const
  { return sparseA; }

  //! Modify the vector of sparse A matrices (which correspond to the sparse
  //! constraints).
  std::vector<SparseConstraintMatrixType>& SparseA() { return sparseA; }

  //! Return the vector of dense A matrices (which correspond to the dense
  //! constraints).
  const std::vector<DenseConstraintMatrixType>& DenseA() const
  { return denseA; }

  //! Modify the vector of dense A matrices (which correspond to the dense
  //! constraints).
  std::vector<DenseConstraintMatrixType>& DenseA() { return denseA; }

  //! Return the vector of sparse B values.
  const BVectorType& SparseB() const { return sparseB; }
  //! Modify the vector of sparse B values.
  BVectorType& SparseB() { return sparseB; }

  //! Return the vector of dense B values.
  const BVectorType& DenseB() const { return denseB; }
  //! Modify the vector of dense B values.
  BVectorType& DenseB() { return denseB; }

  /**
   * Check whether or not the constraint matrices are linearly independent.
   *
   * Warning: possibly very expensive check.
   */
  bool HasLinearlyIndependentConstraints() const;

  //! Get an initial point for the primal coordinates.
  template<typename MatType = arma::mat>
  MatType GetInitialPoint() const;

  //! Get initial points for the primal and dual coordinates.
  template<typename MatType = arma::mat>
  void GetInitialPoints(MatType& coordinates,
                        MatType& ySparse,
                        MatType& yDense,
                        MatType& dualCoordinates) const;

 private:
  //! Objective function matrix c.
  ObjectiveMatrixType c;

  //! A_i for each sparse constraint.
  std::vector<SparseConstraintMatrixType> sparseA;
  //! b_i for each sparse constraint.
  BVectorType sparseB;

  //! A_i for each dense constraint.
  std::vector<DenseConstraintMatrixType> denseA;
  //! b_i for each dense constraint.
  BVectorType denseB;
};

} // namespace ens

// Include implementation.
#include "sdp_impl.hpp"

#endif
