//
// Created by o on 07.04.21.
//

#ifndef ENSMALLEN_LMCMA_LMCMA_HPP
#define ENSMALLEN_LMCMA_LMCMA_HPP


namespace ens {


class LMCMA
{
  public:

    LMCMA(std::size_t N_dim);

    template<typename SeparableFunctionType,
             typename MatType,
             typename... CallbackTypes>
    typename MatType::elem_type Optimize(SeparableFunctionType& f,
                                         MatType& z,
                                         float sigma,         // TODO: remove from here
                                         std::size_t n_iter,  // TODO: Remove from here
                                         CallbackTypes&&... callbacks);
  private:

    template <typename MatType>
    size_t Update(std::size_t t,
                  const MatType& p,
                  MatType& P,
                  MatType& V,
                  arma::umat& L,
                  arma::umat& J);


      template <typename MatType>
      void Reconstruct(const MatType& P,
                       const MatType& V,
                       const arma::umat& J,    /* TODO: why umat? */
                       const std::size_t n_updates,
                       MatType& z);


      template <typename MatType>
      void ReconstructInv(const MatType& V,
                          const MatType& J,          /* TODO: why umat? */
                          const std::size_t n_updates,   // number of updates
                          const MatType z,
                          MatType& out);


      template <typename  MatType>
      float PopulationSuccess(const arma::umat& ranks_cur,
                              const arma::umat& ranks_prev,
                              const MatType& F_cur,
                              const MatType& F_prev);



      //! The maximum number of allowed iterations.
      size_t maxIterations;
      std::size_t T;
      std::size_t lambda;

      float c_c;
      float c1;
      float z_bias;
      float c_sigma;  // conv combination factor for computing stepsize
      float d_sigma;  // scale combination factor for computing stepsize
      arma::mat w;        // weights for computation of one mean-step
      float mu_w;         //
      std::size_t mu;     // number of best species
      std::size_t m;      // number of step vectors stored

      std::size_t N;      // number of iterations between step vectors which are saved
  };

}

#include "lmcma_impl.hpp"

#endif