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



template <typename MatType>
class CholeskyReconstructor
{
  public:

    /**
     * @param n number of dimensions
     * @param m number of reconstriuction vectors 
     * 
     */
    CholeskyReconstructor<MatType>(size_t n, size_t m, size_t T):
    P(m, MatType(n)),
    V(m, MatType(n)),
    T(T)
    J(-1)
    {}


    /**
     * Approximate result of matrix-vector multiplication, 
     * given a fixed set of reconstruction vectors
     * 
     * 
     */
    void Az(MatType& x, size_t n_updates)
    {
      for(size_t t = 0; t < n_updates; t++)
      {
        size_t j = J[t];
        arma::mat v_j = V.col(j);

        float v_norm = arma::norm(v_j);
        float v_norm_sq = v_norm * v_norm;

        float a = std::sqrt(1-c1);    // TODO: move this line to constructor
        float b = a / (v_norm_sq) * ( std::sqrt(1+ c1/(a*a) *  v_norm_sq) - 1 );  // b^{J[I[t]}

        x = a * x  +  b * as_scalar(V.col(j).t() * z) * P.col(j);
      }
    }



    /**
     * Approximate result of inverse matrix-vector multiplication, 
     * given a fixed set of reconstruction vectors
     * 
     * 
     */
    void InvAz(MatType& x, size_t n_updates)
    {
      float c  = std::sqrt(1-c1);
      float c_sq = 1-c1;

      out = BaseMatType(z);
      for(size_t t = 0; t < n_updates; t++)
      {
        size_t j = J[t];
        BaseMatType v_j = V.col(j);

        float v_norm = arma::norm(v_j);
        float v_norm_sq = v_norm * v_norm;

        float d = 1/ ( c * v_norm_sq) * (1 - 1 / std::sqrt(1 + c1 / c_sq * v_norm_sq  ) );

        out = 1 / c * out - d * as_scalar(V.col(j).t() * out) * V.col(j);
      }
    }



    /**
     *
     * Update the reconstruction Vectors
     * 
     * 
     */
    void Update(size_t t)
    {
      t = std::floor(t/T);    // this is the t'th update

      if(t < m)
      {
        // if less then m updates, straight forward
        J[t] = t;
      } else {
        // Find pair of step vectors p, saved at lowest distance
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
          i_min = 1;    // the distance is longer then N (to long) => shift

        size_t tmp = J[i_min];
        for(size_t i = i_min; i < m  1; i++)
        {
          J[i] = J[i + 1];
        }
        J[m - 1] = tmp;
      }

      // this guy will be replaced
      std::size_t j_cur = J[std::min(t, m-1)];
      L[j_cur] = t*T;

      // update P
      P[j_cur] = p;

      // Reconstruct inverse
      arma::mat v;
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

    size_t m;

    std::vector<size_t> J;
}




}



# endif