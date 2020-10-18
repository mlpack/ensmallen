/**
 * @file affine_simplexer.hpp
 * @author Marcus Edel
 *
 * Definition of a simple affine simplex to ensure that the representation of
 * points in the affine hull is unique.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_NELDER_MEAD_AFFINE_SIMPLEXER_HPP
#define ENSMALLEN_NELDER_MEAD_AFFINE_SIMPLEXER_HPP 

namespace ens {

/*
 * Definition of the affine simplexer.
 */
class AffineSimplexer
{
 public:
  /**
   * Affine simplexer, a simplex is represented by an (n + 1)-dimensional vector
   * of n-dimensional vectors.  It is used together with the initial vector to
   * create the initial simplex. To construct the ith vertex, the simplexer
   * multiplies entry i in the initial vector with a constant.
   *
   * @param simplex Constructed simplex.
   * @param iterate Intial starting point.
   * @param function Function to optimize.
   */
  template<typename FunctionType, typename MatType>
  void Simplex(MatType& simplex,
               const MatType& iterate,
               FunctionType& /* function */)
  {
    // Convenience typedefs.
    typedef typename MatType::elem_type ElemType;

    // Construct the initial simplex.
    // Large initial simplex is used.
    const ElemType scalingFactor = std::min(std::max(
        arma::as_scalar(arma::max(
        arma::abs(iterate))), (ElemType) 1.0), (ElemType) 10.0);

    simplex = arma::eye<MatType>(iterate.n_rows, iterate.n_rows + 1);
    simplex.col(iterate.n_rows) = (1.0 -
        std::sqrt((ElemType) (iterate.n_rows + 1))) /
        (ElemType) iterate.n_rows * arma::ones<MatType>(iterate.n_rows, 1);
    
    simplex *= scalingFactor;
    simplex.each_col() += iterate;
  }
};

} // namespace ens

#endif
