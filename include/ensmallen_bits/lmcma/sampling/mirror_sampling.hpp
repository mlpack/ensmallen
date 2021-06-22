/**
 * @file full_selection.hpp
 * @author Oleksandr Nikolskyy
 *
 * 
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */


#ifndef ENSMALLEN_LMCMA_MIRROR_SAMPLING_HPP
#define ENSMALLEN_LMCMA_MIRROR_SAMPLING_HPP

#include <ensmallen_bits/lmcma/reconstruction/cholesky_reconstruction.hpp>

namespace ens {
/**
 * Sample only half of individuals with random Rademacher variables
 * The other half is "mirrored" against expectation 
 */

class MirrorSampling
{
 public:
   MirrorSampling(){}

   /** 
   * @tparam MatType space where individuals live in
   * @param x Individuals to sample
   * @param m expecataiton 
   * @param reconstructor 
   * @param t which update
   */
   template <typename MatType>
   void Sample(std::vector<MatType>& x, 
               const MatType& m, 
               CholeskyReconstructor<MatType>& reconstructor,
               const size_t t)
   {
      
      typedef typename MatType::elem_type ElemType;

      for(size_t i = 0; i < x.size(); i++)
      {
        if(i % 2 == 0)
        {
            // Radermacher
            x[i].fill(arma::fill::randu);
            x[i].transform( [](ElemType& val) {return (val = val >= 0.5 ? (ElemType) 1 : (ElemType) -1);});
            x[i].print("Radermacher");
            reconstructor.Az(x[i], t, i);
            // DEBUG
            
        }
        else
        {
            // Mirror
            x[i] = m - (x[i - 1] - m);
            x[i].print("Mirror");
        }
      }

  }
   
};

} // namespace ens

#endif
