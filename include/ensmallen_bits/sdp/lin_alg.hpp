/**
 * @file lin_alg.hpp
 * @author Nishant Mehta
 *
 * Linear algebra utilities.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SDP_LIN_ALG_HPP
#define ENSMALLEN_SDP_LIN_ALG_HPP

namespace ens {
namespace math {

inline size_t SvecIndex(size_t i, size_t j, size_t n)
{
  if (i > j)
    std::swap(i, j);

  return (j - i) + (n * (n + 1) - (n - i) * (n - i + 1)) / 2;
}

/**
 * Upper triangular representation of a symmetric matrix, scaled such that,
 * dot(Svec(A), Svec(B)) == dot(A, B) for symmetric A, B. Specifically,
 *
 * Svec(K) = [ K_11, sqrt(2) K_12, ..., sqrt(2) K_1n, K_22, ..., sqrt(2)
 * K_2n, ..., K_nn ]^T
 *
 * @param input A symmetric matrix.
 * @param output Upper triangular representation.
 */
template<typename MatAType, typename MatBType>
inline void Svec(const MatAType& input, MatBType& output)
{
  MatBType iMat(input);
  const size_t n = iMat.n_rows;
  const size_t n2bar = n * (n + 1) / 2;

  output.zeros(n2bar, 1);

  size_t idx = 0;
  for (size_t i = 0; i < n; i++)
  {
    for (size_t j = i; j < n; j++)
    {
      if (i == j)
        output(idx++, 0) = iMat(i, j);
      else
        output(idx++, 0) = arma::datum::sqrt2 * iMat(i, j);
    }
  }
}

template<typename MatAType, typename ElemType>
inline void Svec(const MatAType& input,
                 arma::SpMat<ElemType>& output)
{
  arma::SpMat<ElemType> iMat(input);

  const size_t n = iMat.n_rows;
  const size_t n2bar = n * (n + 1) / 2;

  output.zeros(n2bar, 1);

  const auto itEnd = iMat.end();
  for (auto it = iMat.begin(); it != itEnd; ++it)
  {
    const size_t i = it.row();
    const size_t j = it.col();
    if (i > j)
      continue;
    if (i == j)
      output(SvecIndex(i, j, n), 0) = *it;
    else
      output(SvecIndex(i, j, n), 0) = arma::datum::sqrt2 * (*it);
  }
}

/**
 * The inverse of Svec. That is, Smat(Svec(A)) == A.
 *
 * @param input Input matrix.
 * @param output The inverse of the input matrix.
 */
template<typename MatAType, typename MatBType>
inline void Smat(const MatAType& input, MatBType& output)
{
  MatBType iMat(input);

  const size_t n = static_cast<size_t>
      (ceil((-1. + sqrt(1. + 8. * iMat.n_elem))/2.));

  output.zeros(n, n);

  size_t idx = 0;
  for (size_t i = 0; i < n; i++)
  {
    for (size_t j = i; j < n; j++)
    {
      if (i == j)
        output(i, j) = iMat(idx++);
      else
        output(i, j) = output(j, i) = 0.5 * arma::datum::sqrt2 * iMat(idx++);
    }
  }
}

/**
 * If A is a symmetric matrix, then SymKronId returns an operator Op such that
 *
 *    Op * svec(X) == svec(0.5 * (AX + XA))
 *
 * for every symmetric matrix X
 *
 * @param A A symmetric matrix.
 * @param  The calculated operator.
 */
template<typename MatAType, typename MatBType>
inline void SymKronId(const MatAType& A, MatBType& op)
{
  MatBType iMat(A);

  // TODO(stephentu): there's probably an easier way to build this operator

  const size_t n = iMat.n_rows;
  const size_t n2bar = n * (n + 1) / 2;
  op.zeros(n2bar, n2bar);

  size_t idx = 0;
  for (size_t i = 0; i < n; i++)
  {
    for (size_t j = i; j < n; j++)
    {
      for (size_t k = 0; k < n; k++)
      {
        op(idx, SvecIndex(k, j, n)) +=
          ((k == j) ? 1. : 0.5 * arma::datum::sqrt2) * iMat(i, k);
        op(idx, SvecIndex(i, k, n)) +=
          ((k == i) ? 1. : 0.5 * arma::datum::sqrt2) * iMat(k, j);
      }
      op.row(idx) *= 0.5;
      if (i != j)
        op.row(idx) *= arma::datum::sqrt2;
      idx++;
    }
  }
}

} // namespace math
} // namespace ens

#endif
