/**
 * @file cmaes_impl.hpp
 * @author Oleksandr Nikolskyy
 *
 * Implementation of the LM CMA algorithm - useful in a derivative-free large-scale
 * black-box optimization scenario
 *
 * For details see "LM-CMA: an Alternative to L-BFGS for Large Scale Black-box Optimization" by Loshchilov
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_LMCMA_LMCMA_IMPL_HPP
#define ENSMALLEN_LMCMA_LMCMA_IMPL_HPP

#include "lmcma.hpp"
#include <ensmallen_bits/function.hpp>


namespace ens {
    template<typename SelectionPolicyType, typename SamplingType>
    LMCMA<SelectionPolicyType, SamplingType>::LMCMA(std::size_t N_dim,
                  const SamplingType& sampler, 
                  const SelectionPolicyType& selectionPolicy) :
    T(std::ceil(std::log(N_dim))),
    c_c(0.5/std::sqrt(N_dim)),
    c1(1/(10 * std::log(N_dim + 1))),
    z_bias(0.25),
    c_sigma(0.3),
    d_sigma(0.7),
    m(4 + (size_t)std::floor(3*std::log(N_dim))),
    lambda(4 + (size_t)std::floor(3*std::log(N_dim))),
    w( 1,std::floor( (4 + (size_t)std::floor(3*std::log(N_dim))) / 2 ), arma::fill::zeros),
    mu(std::floor( (4 + (size_t)std::floor(3 * std::log(N_dim))) / 2 ) )
    {}

    template<typename SelectionPolicyType, typename SamplingType>
    template<typename SeparableFunctionType,
            typename MatType,
            typename... CallbackTypes>
    typename MatType::elem_type LMCMA<SelectionPolicyType, SamplingType>::Optimize(SeparableFunctionType& f,
                                                MatType& z,
                                                float sigma,         // TODO: remove from here
                                                std::size_t n_iter,  // TODO: Remove from here
                                                CallbackTypes&&... callbacks)
    {


      // Convenience typedefs.
      typedef typename MatType::elem_type ElemType;
      typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;

      // Make sure that we have the methods that we need.  Long name...
      traits::CheckArbitrarySeparableFunctionTypeAPI<
              SeparableFunctionType, BaseMatType>();
      RequireDenseFloatingPointType<BaseMatType>();

      // Approximation of covariance matrix
      CholeskyReconstructor<BaseMatType> reconstructor(z.n_elem, m, T);

      BaseMatType& iterate = (BaseMatType&) z;

      size_t N_dim = z.n_elem;
      size_t N = N_dim;

      // Pointer vectors
      arma::umat J(1, m, arma::fill::zeros);
      arma::umat L(1, m, arma::fill::zeros);

      // estimated Expectation values for samples
      BaseMatType m_old(N_dim, 1, arma::fill::zeros),
        m_new(N_dim, 1, arma::fill::zeros);

      // weights for estimation of Expectaton value
      float sum = 0;
      for(size_t i = 0; i < mu; i++)
        sum+= log(i+1);

      for(size_t i = 0; i < mu; i++)
      {
        float phi = std::log(mu+1) - std::log(i+1);
        w[i] = (std::log(mu+1) - std::log(i+1) ) / (mu * log(mu + 1) - sum);
      }

      // const to recompute covariance steps
      mu_w = 1 / std::pow(arma::norm(w,2), 2);

      // evaluation fitness function values
      arma::Mat<BaseMatType> f_eval(lambda, 1),       // arma::Mat<BaseMatType>, is actually equal to
        f_eval_old(lambda, 1);

      f_eval.fill(arma::datum::inf);
      f_eval_old.fill(arma::datum::inf);

      //  generation at t'th step
      std::vector<> generation(N_dim, lambda, arma::fill::zeros);

      float s = 0;

      //
      arma::uvec index(lambda, 1, arma::fill::zeros),
        index_old(lambda, 1, arma::fill::zeros);

      // Controls early termination of the optimization process.
      bool terminate = false;

      //********************
      // Now iterate!
      //********************
      terminate |= Callback::BeginOptimization(*this, f, iterate,
                                               callbacks...);
      for (size_t t = 1; t < maxIterations && !terminate; ++t)
      {
        float psr = 0; // population success

        sampler.Sample(generation, m_old, sigma, reconstructor, t);

        // recombination
        m_old = BaseMatType(m_new);
        m_new = BaseMatType(N_dim,1, arma::fill::zeros);                // TODO: possible without reinintialization - would save some time

        for (size_t j = 0; j < mu; j++)
        {
          arma::mat temp  = w(j) * X.col(index(j));
          m_new += w(j) * X.col(index(j));
        }

        // covariance
        p_c = (1- c_c) * p_c + std::sqrt(c_c* (2-c_c) * mu_w) * (m_new - m_old) * sigma;

        if(t % T == 0)
        {
          reconstructor.Update(t)
        }

        psr = populationSuccess(index, index_old, f_eval, f_eval_old) - z_bias;
        s = (1 - c_sigma) * s + c_sigma * psr;

        // step size
        sigma = sigma * std::exp(s/d_sigma);

        f_eval_old = BaseMatType(f_eval);
      }
    }

   
    template <typename  MatType>
    float LMCMA::PopulationSuccess(const arma::umat& ranks_cur,
                            const arma::umat& ranks_prev,
                            const MatType& F_cur,
                            const MatType& F_prev)
    {
      arma::umat ranks_mixed(2*lambda, 1, arma::fill::none);
      BaseMatType F_mixed = arma::join_rows(F_cur, F_prev);
      arma::umat idx = arma::sort_index(F_mixed);

      for(size_t i = 0; i < 2*lambda; i++)
      {
        ranks_mixed[idx[i]] = i;
      }

      float mean_prev = 0;
      float mean_cur = 0;

      for(size_t i = 0; i < lambda; i++)
      {
        mean_cur+= ranks_mixed[i];
        mean_prev+= ranks_mixed[lambda + i];
      }

      mean_prev /= std::pow(lambda,2);
      mean_cur /= std::pow(lambda,2);

      return mean_prev - mean_cur;
    }

}

#endif