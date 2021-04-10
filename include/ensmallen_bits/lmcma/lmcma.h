//
// Created by o on 07.04.21.
//

#ifndef ENSMALLEN_2_15_1_LMCMA_H
#define ENSMALLEN_2_15_1_LMCMA_H


namespace ens {

#endif //ENSMALLEN_2_15_1_LMCMA_H


class LMCMA {
public:

    LMCMA(std::size_t N_dim):

    template <typename FitnessFunction>
    void optimize(FitnessFunction& f,
                  arma::mat& z,
                  float sigma,
                  std::size_t n_iter)
    {
      // Convenience typedefs.
      typedef typename MatType::elem_type ElemType;
      typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;

      // Pointer vectors
      arma::umat J(1, m, arma::fill::zeros);
      arma::umat L(1, m, arma::fill::zeros);

      // reconstruction step vectors
      arma::mat P(N_dim, m, arma::fill::zeros);
      arma::mat V(N_dim, m, arma::fill::zeros);
      arma::mat p_c(N_dim, 1, arma::fill::zeros);      // update for step vector

      arma::mat m_old(N_dim, 1, arma::fill::zeros), // mean
      m_new(N_dim, 1, arma::fill::zeros);
      arma::mat f_eval(lambda, 1), // evaluation fitness function values
      f_eval_old(lambda, 1);

      f_eval.fill(arma::datum::inf);
      f_eval_old.fill(arma::datum::inf);

      arma::mat X(N_dim, lambda, arma::fill::zeros); // generation at t'th step

      size_t n_updates = 0;

      float s = 0;

      uvec index(lambda, 1, arma::fill::zeros), index_old(lambda, 1, arma::fill::zeros);

      size_t t  = 0;

      while(t <= n_iter)
      {
        float psr = 0; // population success

        for (size_t k = 0; k < lambda; k++)
        {
          z = arma::mat(N_dim, 1, arma::fill::randn);          // TODO: Radermacher sampling, mirror sampling
          reconstruct(P, V, J, std::min((size_t)std::floor(t/T), m-1), z);                     // compute Az
          z = m_new + sigma*z;
          float eval = f.Evaluate(z); // DEBUG
          f_eval[k] = eval;
          X.col(k) = z;
        }
        std::cout << "min: "<<  arma::min(f_eval) << "\n";

        // recombination
        index_old = arma::uvec(index);

        index = arma::sort_index(f_eval);

        m_old = arma::mat(m_new);
        m_new = arma::mat(N_dim,1, arma::fill::zeros);

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

        sigma = sigma * std::exp(s/d_sigma);
        cout << "sigma:" << sigma << "\n";
        t++;

        f_eval_old = arma::mat(f_eval);
      }

    }

private:

    std::size_t update(std::size_t t,
                       const arma::mat&p,
                       arma::mat&P,
                       arma::mat&V,
                       arma::umat&L,
                       arma::umat&J)
    {
      t = std::floor(t/T);    // this is the t'th update
      if(t < m)
      {
        J[t] = t;     // if less then m updates, straight forward
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

      std::size_t j_cur = J[std::min(t, m-1)]; // this guy will be replaced


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


    void reconstruct(const arma::mat&P,
                     const arma::mat&V,
                     const arma::umat&J,    /* TODO: why umat? */
                     const std::size_t n_updates,
                     arma::mat&z)
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


    void reconstructInv(const arma::mat&V,
                        const arma::umat&J,    /* TODO: why umat? */
                        const std::size_t n_updates,   // number of updates
                        const arma::mat z,
                        arma::mat&out)
    {
      float c  = std::sqrt(1-c1);
      float c_sq = 1-c1;

      out = arma::mat(z);
      for(size_t t = 0; t < n_updates; t++)
      {
        size_t j = J[t];
        arma::mat v_j = V.col(j);

        float v_norm = arma::norm(v_j);
        float v_norm_sq = v_norm * v_norm;

        float d = 1/ ( c * v_norm_sq) * (1 - 1 / std::sqrt(1 + c1 / c_sq * v_norm_sq  ) );

        out = 1 / c * out - d * as_scalar(V.col(j).t() * out) * V.col(j);
      }

    }


    float populationSuccess(const arma::umat& ranks_cur,
                            const arma::umat& ranks_prev,
                            const arma::mat& F_cur,
                            const arma::mat& F_prev)
    {
      arma::umat ranks_mixed(2*lambda, 1, arma::fill::none);

      arma::mat F_mixed = arma::join_rows(F_cur, F_prev);
      //F_mixed.print("F:");


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

      std::cout << "psr:" << mean_prev - mean_cur << "\n";
      return mean_prev - mean_cur;
    }

    //! The maximum number of allowed iterations.
    size_t maxIterations;
    std::size_t T;
    std::size_t lambda;

    float c_c;
    float c1;
    float z_bias;
    float c_sigma;  // conv combination factor for computing stepsize
    float d_sigma;  // scale combination factor for computing stepsize
    std::size_t N_dim;  // dimension of problem
    arma::mat w;        // weights for computation of one mean-step
    float mu_w;         //
    std::size_t mu;     // number of best species
    std::size_t m;      // number of step vectors stored

    std::size_t N;      // number of iterations between step vectors which are saved
};

}