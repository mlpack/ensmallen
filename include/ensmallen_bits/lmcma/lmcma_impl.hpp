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
    LMCMA::LMCMA(std::size_t N_dim) :
            T(std::ceil(std::log(N_dim))),
            c_c(0.5/std::sqrt(N_dim)),
            c1(1/(10 * std::log(N_dim + 1))),
            z_bias(0.25),
            c_sigma(0.3),
            d_sigma(0.7),
            m(4 + (size_t)std::floor(3*std::log(N_dim))),
            lambda(4 + (size_t)std::floor(3*std::log(N_dim))),
            //lambda(100),
            w( 1,std::floor( (4 + (size_t)std::floor(3*std::log(N_dim))) / 2 ), arma::fill::zeros),
            mu(std::floor( (4 + (size_t)std::floor(3 * std::log(N_dim))) / 2 ) )
    {}


    template<typename SeparableFunctionType,
            typename MatType,
            typename... CallbackTypes>
    typename MatType::elem_type LMCMA::Optimize(SeparableFunctionType& f,
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

      BaseMatType& iterate = (BaseMatType&) z;

      size_t N_dim = z.n_elem;
      size_t N = N_dim;

      // Pointer vectors
      arma::umat J(1, m, arma::fill::zeros);
      arma::umat L(1, m, arma::fill::zeros);

      // reconstruction step vectors
      BaseMatType P(N_dim, m, arma::fill::zeros);       // BaseMatType, because is vector in search space
      BaseMatType V(N_dim, m, arma::fill::zeros);
      BaseMatType p_c(N_dim, 1, arma::fill::zeros);

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
      BaseMatType X(N_dim, lambda, arma::fill::zeros);

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

        // sampling
        for (size_t k = 0; k < lambda; k++)
        {
          z = BaseMatType(N_dim, 1, arma::fill::randn);          // TODO: Radermacher sampling, mirror sampling
          // compute Az
          reconstruct(P, V, J, std::min((size_t)std::floor(t/T), m-1), z);
          z = m_new + sigma*z;
          f_eval[k] = f.Evaluate(z);
          X.col(k) = z;
        }

        // recombination
        index_old = arma::uvec(index);
        index = arma::sort_index(f_eval);

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
          update(t, p_c, P, V, L, J);
        }

        psr = populationSuccess(index, index_old, f_eval, f_eval_old) - z_bias;
        s = (1 - c_sigma) * s + c_sigma * psr;

        // step size
        sigma = sigma * std::exp(s/d_sigma);

        f_eval_old = BaseMatType(f_eval);
      }
    }

    template <typename BaseMatType>
    size_t LMCMA::Update(std::size_t t,
                         const BaseMatType& p,
                         BaseMatType& P,
                         BaseMatType& V,
                         arma::umat& L,
                         arma::umat& J)
    {
      t = std::floor(t/T);    // this is the t'th update

      if(t < m)
      {
        // if less then m updates, straight forward
        J[t] = t;
      } else {
        // Find pair of step vectors p, saved at lowest distance
        // TODO: can possibly be done in a more efficient way - with armadillo?
        size_t i_min = -1;
        size_t dif = std::numeric_limits<std::size_t>::infinity();
        for(size_t i=0; i < m -1; i++)
        {
          if(L[J[i+1]] - L[J[i]] - N < dif)
          {
            i_min = i;
            dif = L[J[i+1]] - L[J[i]] - N;
          }
        }
        i_min++;

        if(dif >= 0)
          i_min = 1;    // the distance is longer then N (to long) => shift

        std::size_t tmp = J[i_min];
        for(size_t i=i_min; i < m-1; i++)
        {
          J[i] = J[i+1];
        }
        J[m-1] = tmp;
      }

      // this guy will be replaced
      std::size_t j_cur = J[std::min(t, m-1)];
      L[j_cur] = t*T;

      // update P
      P.col(j_cur) = p;

      // Reconstruct inverse
      arma::mat v;
      for(std::size_t j = 0; j <= std::min(t,m-1); j++){
        /* TODO: nonefficient, only a subset of V should be updated */
        reconstructInv(V, J, j, P.col(j),v);
        V.col(j) = v;
      }
      return j_cur;
    }

    template <typename BaseMatType>
    void LMCMA::Reconstruct(const BaseMatType& P,
                     const BaseMatType& V,
                     const arma::umat& J,    /* TODO: why umat? */
                     const std::size_t n_updates,
                     BaseMatType& z)
    {
      for(size_t t = 0; t < n_updates; t++)
      {
        size_t j = J[t];
        arma::mat v_j = V.col(j);

        float v_norm = arma::norm(v_j);
        float v_norm_sq = v_norm * v_norm;

        float a = std::sqrt(1-c1);    // TODO: move this line to constructor
        float b = a / (v_norm_sq) * ( std::sqrt(1+ c1/(a*a) *  v_norm_sq) - 1 );  // b^{J[I[t]}

        z = a * z  +  b * as_scalar(V.col(j).t() * z) * P.col(j);
      }
    }

    template <typename BaseMatType>
    void LMCMA::ReconstructInv(const BaseMatType& V,
                               const BaseMatType& J,          /* TODO: why umat? */
                               const std::size_t n_updates,   // number of updates
                               const BaseMatType z,
                               BaseMatType& out)
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

    template <typename  BaseMatType>
    float LMCMA::PopulationSuccess(const arma::umat& ranks_cur,
                            const arma::umat& ranks_prev,
                            const BaseMatType& F_cur,
                            const BaseMatType& F_prev)
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