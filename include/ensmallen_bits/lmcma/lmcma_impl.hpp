/**
 * @file lmcma_impl.hpp
 * @author Oleksandr Nikolskyy
 *
 * Implementation of the LM CMA algorithm - useful in a derivative-free large-scale
 * black-box optimization scenario. Eg, where CMA-ES fails to scale.
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
    template<typename SamplingType>
    LMCMA<SamplingType>::LMCMA(
      const size_t lambda,
      const size_t maxIterations,
      const double tolerance) :
    lambda(lambda),
    maxIterations(maxIterations),
    tolerance(tolerance)
    {}

    template<typename SamplingType>
    template<typename ArbitraryFunctionType,
            typename MatType,
            typename... CallbackTypes>
    typename MatType::elem_type LMCMA<SamplingType>::Optimize(
      ArbitraryFunctionType& f,
      MatType& iterateIn,
      CallbackTypes&&... callbacks)
    {
      // Convenience typedefs.
      typedef typename MatType::elem_type ElemType;
      typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;

      // Make sure that we have the methods that we need.  Long name...
      traits::CheckArbitrarySeparableFunctionTypeAPI<
              ArbitraryFunctionType, BaseMatType>();
      RequireDenseFloatingPointType<BaseMatType>();

      // Approximate the covariance matrix-vector multiplication
      BaseMatType& iterate = (BaseMatType&) iterateIn;

      // Assign some default values to use during the optimization routine
      if (lambda == 0)
        lambda = (4 + std::round(3 * std::log(iterate.n_elem))) * 10;

      size_t n_dimensions = iterate.n_elem;
      lambda = (lambda == 0) ? 4 + (size_t)std::floor(3*std::log(n_dimensions)) : lambda;
      size_t T = std::ceil(std::log(n_dimensions));    
      std::size_t mu = (size_t)(std::floor(lambda / 2));     // number of best species       // TODO: Prio 0: define
      ElemType c_c = 0.5 / std::sqrt(n_dimensions);
      ElemType sigma = 1.;
      ElemType z_bias = 0.25;
      ElemType c_sigma = 0.3;                                             // convex combination factor for computing stepsize
      ElemType d_sigma = 0.7;                                             // scale combination factor for computing stepsize
      BaseMatType w(mu,1);                                              // recombination weights
      ElemType mu_w = 1 / arma::norm(w);        
      ElemType s = 0;
      size_t m = 4 + (size_t)std::floor(3 * std::log(n_dimensions));      // number of step vectors stored                                                   // number of iterations between step vectors which are saved

      ElemType currentObjective = f.Evaluate(iterate);
      ElemType overallObjective = currentObjective;
      //ElemType lastObjective = std::numeric_limits<ElemType>::max();
      
      CholeskyReconstructor<BaseMatType> reconstructor(iterate.n_rows, iterate.n_cols, m, T);
  
      BaseMatType p_c(iterate.n_rows, iterate.n_cols);

      // estimated Expectation values for samples
      std::vector<BaseMatType>expectation(2, 
        BaseMatType(iterate.n_rows, iterate.n_cols)
      );

      // weights for estimating the Expectaton value
      ElemType sum = 0;
      for(size_t i = 0; i < mu; i++)
        sum+= log(i + 1);

      for(size_t i = 0; i < mu; i++)
      {
        w[i] = (std::log(mu + 1) - std::log(i + 1) ) / (mu * log(mu + 1) - sum);
      }

      mu_w = 1 / std::pow(arma::norm(w,2), 2);

      // individuals at t'th and t-1'th step
      std::vector<BaseMatType> generation(lambda, BaseMatType(iterate.n_rows, iterate.n_cols));

      // objectives at t'th and t-1'th step
      std::vector<MatType> objective(2,
        MatType(lambda, 1)
      );

      objective[0].fill(std::numeric_limits<ElemType>::max());
      objective[1].fill(std::numeric_limits<ElemType>::max());

      // Controls early termination of the optimization process.
      bool terminate = false;
      size_t idx1, idx2;
      //********************
      // Now iterate!
      //********************
      terminate |= Callback::BeginOptimization(*this, f, iterate,
                                               callbacks...);
      for (size_t t = 0; t < maxIterations && !terminate; ++t)
      {
        ElemType psr = 0; // population success

        idx2 = (t + 1) % 2;
        idx1 = t % 2;

        // sampling
        sampler.Sample(generation, expectation[idx1], reconstructor, t);

        // evaluation
        for (size_t i = 0; i < lambda; i++)
        {
          objective[idx2][i] = f.Evaluate(generation[i]);
          Callback::Evaluate(*this, f, expectation[idx1], objective[idx2][i], callbacks...);
        }


        // recombination
        arma::umat sorted = arma::sort_index(objective[idx2]);

        
        expectation[idx2].zeros(); 
        for (size_t j = 0; j < mu; j++)
        {
          expectation[idx2] = expectation[idx2] + w(sorted) * generation[sorted[j]];
        }

        currentObjective = f.Evaluate(expectation[idx2]);

        if(currentObjective < overallObjective){
          overallObjective = currentObjective;
          iterate = expectation[1];
          terminate |= Callback::StepTaken(*this, f, iterate, callbacks...);
        }

        // covariance adaptation  
        if(t % T == 0)
        {
          p_c = (1 - c_c) * p_c + std::sqrt(c_c * (2 - c_c) * mu_w) * (expectation[idx2] - expectation[idx1]) * sigma;
          reconstructor.Update(t, p_c);
        }

        // step size adaptation
        psr = PopulationSuccess<ElemType, MatType>(objective) - z_bias; // TODO: possibly ugly
        s = (1 - c_sigma) * s + c_sigma * psr;
        sigma = sigma * std::exp(s / d_sigma);
      }

      return currentObjective;
    }


    template <typename SamplingType>
    template <typename ElemType, typename  MatType>
    ElemType LMCMA<SamplingType>::PopulationSuccess(
      const std::vector<MatType>&objectives)
    {
      arma::umat ranks_mixed(2*lambda, 1, arma::fill::none);
      MatType objectives_mixed(arma::join_rows(objectives[1], objectives[1]));
      arma::umat idx = arma::sort_index(objectives_mixed);

      for(size_t i = 0; i < 2*lambda; i++)
      {
        ranks_mixed[idx[i]] = i;
      }

      double mean_prev = 0;
      double mean_cur = 0;

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
