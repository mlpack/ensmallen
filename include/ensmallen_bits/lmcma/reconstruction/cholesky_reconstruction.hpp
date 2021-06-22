/**
 * @file cholesky_reconstruction.hpp
 * @author Oleksandr Nikolskyy
 *
 * 
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */


#ifndef ENSMALLEN_LMCMA_CHOL_RECONSTRUCTION_HPP
#define ENSMALLEN_LMCMA_CHOL_RECONSTRUCTION_HPP

namespace ens {

/**
 * Reconstruction of cholesky factors of a covariance matrix 
 * of a Gaussian search distribution
 */
template <typename MatType>
class CholeskyReconstructor
{
  public:

    /**
     * @param n number of dimensions
     * @param m number of reconstriuction vectors
     */
    CholeskyReconstructor(
      size_t n_row,
      size_t n_col, 
      size_t m, 
      size_t T):
    P(m, MatType(n_row, n_col, arma::fill::zeros)),
    V(m, MatType(n_row, n_col, arma::fill::zeros)),
    T(T),
    c1(10 * std::log( (n_row >= n_col ? n_row : n_col) + 1)),
    N(n_row >= n_col ? n_row : n_col), // TODO: make definable
    L(m),
    J(m),
    c(1 / std::sqrt(1 - c1))
    {}

    
    /**
     * Approximate result of matrix-vector multiplication, 
     * given a fixed set of reconstruction vectors
     */
    void Az(
      MatType& x, 
      const size_t n_updates, 
      const size_t k)
    {

      size_t m_chosen;
      size_t sigma = 4;

      if(k == 1)
      {
        sigma = 40;
      }

      m_chosen = std::min(
          (size_t) std::floor(sigma * arma::norm(MatType(1,1, arma::fill::randn), 1)) , n_updates
        );

      MatType z = x;
      double b;
      for(size_t i = n_updates + 1 - m_chosen; i < n_updates; i++){
        b = std::sqrt(1 - c1) / std::pow(arma::norm(V[J[i]]),2);
        b *= std::sqrt(1 + c1 / (1 - c1) * std::pow(arma::norm(V[J[i]]), 2)) - 1; 
        x = std::sqrt(1 - c1) * x;
        x += b * arma::as_scalar((V[J[i]]).t() * z) * P[J[i]];
      }
    }


    /**
     * Approximate result of inverse matrix-vector multiplication, 
     * given a fixed set of reconstruction vectors
     */
    void InvAz(MatType& x, size_t n_updates)
    {
      double d;
      for(size_t i = 0; i < n_updates; i++){
        d = 1 / (std::sqrt(1 - c1) * std::pow(arma::norm(V[J[i]]), 2));
        d *= 1 - 1 / (std::sqrt(1 + c1 / (1 - c1) * std::pow( arma::norm( V[J[i]] ), 2)));
        x = c * x - d * V[J[i]].t() * x * V[J[i]];
      }
    }


    /**
     * Update the reconstruction Vectors
     * preserve the invariance of 
     */
    size_t Update(size_t t, MatType& p)
    {
      // The following line is incoorect! Update is meant to be called only in case t mod T = 0!
      /*t = std::floor(t/T);    // this is the t'th update*/

      // De facto, we call the Update method 
      // for the (t/T)'th time while t is always dividible by T.
      t = t/T;

      if(t < m)
      {
        // if less then m updates, straight forward
        J[t] = t;
      } else {
        // Find pair of reconstruction vectors, saved at lowest distance
        size_t i_min = -1;
        size_t dif = std::numeric_limits<std::size_t>::infinity();
        for(size_t i=0; i < m -1; i++)
        {
          if(L[J[i + 1]] - L[J[i]] - N < dif)
          {
            i_min = i;
            dif = L[J[i + 1]] - L[J[i]] - N;
          }
        }
        i_min++;

        if(dif >= 0)
          // distance is smaller then N => remove oldest of stored m  reconstruction vectors
          i_min = 0;   

        // Shuffle, starting from i_min. i_min will always be replaced   
        size_t tmp = J[i_min];
        for(size_t i = i_min; i < m - 1; i++)
        {
          J[i] = J[i + 1];
        }
        J[m - 1] = tmp;
      }

      // this guy will be replaced
      size_t j_cur = J[std::min(t, m-1)];
      L[j_cur] = t * T;

      // update P
      P[j_cur] = p;

      // Reconstruct inverse
      for(size_t j = 0; j <= std::min(t, m - 1); j++){
        /* TODO: nonefficient, only a subset of V should be updated */
        InvAz(V[j], j);
      }
      return j_cur;
    }


  private:
    std::vector<MatType> P;
    std::vector<MatType> V;

    size_t T;

    double c1;

    size_t N;

    std::vector<std::size_t> L;
    std::vector<std::size_t> J;
    
    double c; 

    size_t m;
    
};

}

# endif
