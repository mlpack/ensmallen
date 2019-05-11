/**
 * @file sdp_impl.hpp
 * @author Stephen Tu
 *
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SDP_SDP_IMPL_HPP
#define ENSMALLEN_SDP_SDP_IMPL_HPP

#include "sdp.hpp"
#include "lin_alg.hpp"

namespace ens {

template<typename ObjectiveMatrixType,
         typename DenseConstraintMatrixType,
         typename SparseConstraintMatrixType,
         typename BVectorType>
SDP<ObjectiveMatrixType,
    DenseConstraintMatrixType,
    SparseConstraintMatrixType,
    BVectorType>::SDP() :
    c(),
    sparseA(),
    sparseB(),
    denseA(),
    denseB()
{ /* Nothing to do. */ }

template<typename ObjectiveMatrixType,
         typename DenseConstraintMatrixType,
         typename SparseConstraintMatrixType,
         typename BVectorType>
SDP<ObjectiveMatrixType,
    DenseConstraintMatrixType,
    SparseConstraintMatrixType,
    BVectorType>::SDP(const size_t n,
                      const size_t numSparseConstraints,
                      const size_t numDenseConstraints) :
    c(n, n),
    sparseA(numSparseConstraints),
    sparseB(numSparseConstraints),
    denseA(numDenseConstraints),
    denseB(numDenseConstraints)
{
  for (size_t i = 0; i < numSparseConstraints; i++)
    sparseA[i].zeros(n, n);
  for (size_t i = 0; i < numDenseConstraints; i++)
    denseA[i].zeros(n, n);
}

template<typename ObjectiveMatrixType,
         typename DenseConstraintMatrixType,
         typename SparseConstraintMatrixType,
         typename BVectorType>
bool SDP<ObjectiveMatrixType,
         DenseConstraintMatrixType,
         SparseConstraintMatrixType,
         BVectorType>::HasLinearlyIndependentConstraints() const
{
  // Very inefficient, should only be used for testing/debugging.

  const size_t n2bar = N2bar();
  DenseConstraintMatrixType A(NumConstraints(), n2bar);
  if (A.n_rows > n2bar)
    return false;

  for (size_t i = 0; i < NumSparseConstraints(); i++)
  {
    DenseConstraintMatrixType sa;
    math::Svec(DenseConstraintMatrixType(SparseA()[i]), sa);
    A.row(i) = sa.t();
  }
  for (size_t i = 0; i < NumDenseConstraints(); i++)
  {
    DenseConstraintMatrixType sa;
    math::Svec(DenseA()[i], sa);
    A.row(NumSparseConstraints() + i) = sa.t();
  }

  const DenseConstraintMatrixType s = arma::svd(A);
  return s(s.n_elem - 1) > 1e-5;
}

//! Get an initial point for the primal coordinates.
template<typename ObjectiveMatrixType,
         typename DenseConstraintMatrixType,
         typename SparseConstraintMatrixType,
         typename BVectorType>
template<typename MatType>
MatType SDP<ObjectiveMatrixType,
            DenseConstraintMatrixType,
            SparseConstraintMatrixType,
            BVectorType>::GetInitialPoint() const
{
  return arma::eye<MatType>(c.n_rows, c.n_rows);
}

//! Get initial points for the primal and dual coordinates.
template<typename ObjectiveMatrixType,
         typename DenseConstraintMatrixType,
         typename SparseConstraintMatrixType,
         typename BVectorType>
template<typename MatType>
void SDP<ObjectiveMatrixType,
         DenseConstraintMatrixType,
         SparseConstraintMatrixType,
         BVectorType>::GetInitialPoints(MatType& coordinates,
                                        MatType& ySparse,
                                        MatType& yDense,
                                        MatType& dualCoordinates) const
{
  coordinates = arma::eye<MatType>(c.n_rows, c.n_rows);
  ySparse = arma::ones<MatType>(NumSparseConstraints(), 1);
  yDense = arma::ones<MatType>(NumDenseConstraints(), 1);
  dualCoordinates = arma::eye<MatType>(c.n_rows, c.n_rows);
}

} // namespace ens

#endif
