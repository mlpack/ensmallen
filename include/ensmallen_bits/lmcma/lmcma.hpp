//
// Created by o on 07.04.21.
//

#ifndef ENSMALLEN_LMCMA_LMCMA_HPP
#define ENSMALLEN_LMCMA_LMCMA_HPP

//#include <ensmallen_bits/cmaes/full_selection.hpp>
//#include <ensmallen_bits/cmaes/random_selection.hpp>

#include <ensmallen_bits/lmcma/sampling/mirror_sampling.hpp>

namespace ens {

template<typename SelectionPolicyType = FullSelection, typename SamplingType = MirrorSampling>
class LMCMA 
{
  public:

    LMCMA(const size_t lambda,
          const size_t batchSize,
          const size_t maxIterations,
          const double tolerance,
          const SamplingType& sampler, 
          const SelectionPolicyType& selectionPolicy);

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

      SelectionPolicyType selectionPolicy;
      SamplingType sampler;
  };

}

#include "lmcma_impl.hpp"

#endif