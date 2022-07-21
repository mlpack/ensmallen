/**
 * @file cmaesparametes.hpp
 * @author John Hoang
 * @author Marcus Edel 
 * 
 * Definition of the Covariance Matrix Adaptation Evolution Strategy as proposed
 * by N. Hansen et al. in "Completely Derandomized Self-Adaptation in Evolution
 * Strategies".
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_CMAES_CMA_PARAMETERS_HPP
#define ENSMALLEN_CMAES_CMA_PARAMETERS_HPP

#include "weight_init_policies/default_weight.hpp"
#include "weight_init_policies/negative_weight.hpp"
namespace ens{
  class CMAparameters {
    public:
    // To do list: different parameters constructor for various improvements or
    // usage or parameters tuning
      CMAparameters() {}; // doing nothing
      /**
       * @brief Construct a new CMAparameters object
       * @param N Starting point dimension
       * @param lambda The population(Offspring sampled in each step) size 
       * @param weightPolicy Policy generate the recombination weights
       */
      template<typename WeightPolicyType>  
      CMAparameters(
        const size_t N,
        const size_t population,
        WeightPolicyType& weightPolicy) : 
        dim(N),
        lambda(population)
      {
        chi = std::sqrt(dim)*(1.0 - 1.0 / (4.0 * dim) + 1.0 / (21 * std::pow(dim, 2)));
        if(lambda == 0)
          lambda = (4+std::round(3*std::log(dim)))*10;
      
        mu = std::round(lambda/2);

        weightPolicy.GenerateRaw(lambda);
        mu_eff = weightPolicy.Mu_eff();

        // Strategy parameter setting: Adaption
        cc = (4.0 + mu_eff/dim)/(4.0 + dim+2*mu_eff/dim);
        csigma = (mu_eff + 2.0)/(dim+mu_eff+5.0);
        c1 = 2 / (std::pow(dim+1.3, 2) + mu_eff);
        alphacov = 2.0;
        cmu = 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / (std::pow(dim+2.0, 2) + alphacov*mu_eff/2);
        cmu = std::min(1.0 - c1, cmu);

        weights = weightPolicy.Generate(dim, c1, cmu);

        // Controlling
        dsigma = 1 + csigma + 2 * std::max(std::sqrt((mu_eff-1)/(dim+1)) - 1, 0.0);
        hsigma = (1.4 + 2.0 / (dim + 1.0)) * chi;

      }

      size_t dim;
      size_t lambda;
      size_t mu; /**< number of candidate solutions used to update the distribution parameters. */
      size_t offsprings;
      // TODO: might need a more general type
      arma::Row<double> weights; /**< offsprings weighting scheme. */
      double csigma; /**< cumulation constant for step size. */
      double c1; /**< covariance matrix learning rate for the rank one update using pc. */
      double cmu; /**< covariance matrix learning reate for the rank mu update. */
      double cc; /**< cumulation constant for pc. */
      double mu_eff; /**< \sum^\mu _weights .*/
      double dsigma; /**< step size damping factor. */
      double alphamu;
      // computed once at init for speeding up operations.
      double fact_ps;
      double fact_pc;
      double chi; /**< norm of N(0,I) */
      double hsigma;

      // active cma.
      double cm; /**< learning rate for the mean. */
      double alphacov; /**< = 2 (active CMA only) */

      // stopping criteria parameters
      size_t countval;
      size_t maxfevals;
      
  };
}

#endif