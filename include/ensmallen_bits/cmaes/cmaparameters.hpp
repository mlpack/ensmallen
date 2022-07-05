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

namespace ens{
  template<typename MatType>
  class CMAparameters {
    template<typename U> friend class CMAES;
    public:
      typedef typename MatType::elem_type ElemType;
      typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType; 
    // To do list: different parameters constructor for various improvements or
    // usage or parameters tuning
      CMAparameters() {}; // doing nothing
      /**
       * @brief Construct a new CMAparameters object
       * @param dim Starting point dimension
       * @param lambda The population(Offspring sampled in each step) size 
       * @param negativeWeight The implementation of negative weight improvement, 1 means yes 0 means no
       */
      CMAparameters(
        const size_t dim,
        const size_t lambda = 0,
        const size_t negativeWeight = 1) : 
        dim(dim),
        lambda(lambda),
        negativeWeight(negativeWeight)
      {
        initialize_params();
      };
      void initialize_params(){
        if(lambda == 0)
          lambda = (4+std::round(3*std::log(dim)))*10;
        mu = std::round(lambda/2);
        if(negativeWeight)
        {
          //TODO: how to access a matrix from index 0 to j: by flattening the matrix weights
          arma::Row<ElemType> w = std::log(mu + 0.5) - arma::log(arma::linspace<arma::Row<ElemType>>(0, lambda-1, lambda) + 1.0);
          weights = w;
          double mu_neg_eff = 0;
          size_t idx_neg = mu;
          double sumPos =  arma::accu(w.cols(0, idx_neg-1));
          double sumNeg = std::abs(arma::accu(w.cols(idx_neg, lambda-1)));
          weights.cols(0, idx_neg-1) = w.cols(0, idx_neg-1) / sumPos;
          const double alpha_mu_negative = 1 + c1/cmu;
          const double mueff_negative = std::pow(sumNeg, 2) / arma::accu(arma::pow(w.cols(idx_neg, lambda-1), 2));
          const double alpha_mueff_negative = 1 + 2 * mueff_negative / (mueff_negative + 2);
          const double alpha_posdef_negative = (1 - c1 - cmu) / (dim * cmu);
          weights.cols(idx_neg, lambda-1) = w.cols(idx_neg, lambda-1) * 
            std::min(std::min(alpha_mu_negative, alpha_mueff_negative), alpha_posdef_negative) / sumNeg;
        } 
        else 
        {
          weights = std::log(mu + 0.5) - arma::log(arma::linspace<arma::Row<ElemType>>(0, mu-1, mu) + 1.0);
          weights /= arma::accu(weights);
        }
        offsprings = negativeWeight ? lambda : mu;
        muw = 1/arma::accu(arma::pow(weights, 2));
        csigma = (muw + 2.0)/(dim+muw+5.0);
        cc = (4.0 + muw/dim)/(4.0 + dim+2*muw/dim);
        c1 = 2 / (std::pow(dim+1.3, 2) + muw);
        alphacov = 2.0;
        cmu = 2.0 * (muw - 2.0 + 1.0 / muw) / (std::pow(dim+2.0, 2) + alphacov*muw/2);
        cmu = std::min(1.0 - c1, cmu);

        dsigma = 1 + csigma + 2 * std::max(std::sqrt((muw-1)/(dim+1)) - 1, 0.0);
        chi = std::sqrt(dim)*(1.0 - 1.0 / (4.0 * dim) + 1.0 / (21 * std::pow(dim, 2)));
        hsigma = (1.4 + 2.0 / (dim + 1.0)) * chi;
  
      };
      // size_t dim() const { return dim; }
      // size_t& dim() { return dim; }

      // size_t lambda() const { return lambda; }
      // size_t& lambda() { return lambda; }

      // size_t mu() const { return mu; }
      // size_t& mu() { return mu; }
      
      // BaseMatType weights() const { return weights; }
      // BaseMatType& weights() { return weights; }

      // double csigma() const { return csigma; }
      // double& csigma() { return csigma; }

      // double c1() const { return c1; }
      // double& c1() { return c1; }
           
      // double cmu() const { return cmu; }
      // double& cmu() { return cmu; }

      // double cc() const { return cc; }
      // double& cc() { return cc; }

      // double muw() const { return muw; }
      // double& muw() { return muw; }

      // double dsigma() const { return dsigma; }
      // double& dsigma() { return dsigma; }

      // double alphacov() const { return alphacov; }
      // double& alphacov() { return alphacov; }

      // double chi() const { return chi; }
      // double& chi() { return chi; }

      // double hsigma() const { return hsigma; }
      // double& hsigma() { return hsigma; }
    private:
      size_t dim;
      size_t lambda; 
  
      double lowerBound; /**< Lower bound of decision variables. */
      double upperBound; /**< Upper bound of decision variables */

      size_t mu; /**< number of candidate solutions used to update the distribution parameters. */
      // TODO: might need a more general type
      arma::Row<ElemType> weights; /**< offsprings weighting scheme. */
      double csigma; /**< cumulation constant for step size. */
      double c1; /**< covariance matrix learning rate for the rank one update using pc. */
      double cmu; /**< covariance matrix learning reate for the rank mu update. */
      double cc; /**< cumulation constant for pc. */
      double muw; /**< \sum^\mu _weights .*/
      double dsigma; /**< step size damping factor. */
      double alphamu;
      // computed once at init for speeding up operations.
      double fact_ps;
      double fact_pc;
      double chi; /**< norm of N(0,I) */
      double hsigma;
      arma::Row<ElemType> temp;
      // active cma.
      double cm; /**< learning rate for the mean. */
      double alphacov; /**< = 2 (active CMA only) */

      // negative weight
      size_t negativeWeight;
      size_t offsprings;
  };
}

#endif