/**
 * @file vd_update.hpp
 * @author John Hoang 
 * 
 * Vanila update the distribution's paramters
 * 
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 * 
 */
#ifndef ENSMALLEN_CMAES_VANILA_VD_HPP
#define ENSMALLEN_CMAES_VANILA_VD_HPP

namespace ens{

class VDUpdate{

 public:
  /**
   * Constructor
   * 
   */
  VDUpdate()
  {
    // Doing nothing
  }

  template<typename MatType, typename BaseMatType>
  std::vector<BaseMatType> samplePop(
    double sigma
    size_t lambda,
    MatType& iterate,
    std::vector<BaseMatType> z,
    std::vector<BaseMatType> y,
    BaseMatType mCandidate,
    BaseMatType B,
    BaseMatType D, 
    BaseMatType sepCovsqinv,
    BaseMatType sepCov,
    BaseMatType C)
  {
    double sqv = arma::accu(arma::pow(v, 2));
    double fact = std::sqrt(1 + sqv) - 1;
	  BaseMatType vbar = v / std::sqrt(sqv);
    for (size_t j = 0; j < lambda; ++j)
    {
      // z_j ~ N(0, I)
      z[j] = arma::randn<BaseMatType>(iterate.n_rows, iterate.n_cols);
      // y_j ~ N(0, I + vv^t)
      y[j] = z[j] + fact * vbar * arma::dot(vbar, z[j]);
      // x_j ~ N(x_mean, sigma * D(I+vv^t)D)
      candidates[j] = mCandidate + sigma * (sepCov % y[j]);
    }
    return candidates;
  }
  /**
   * This function will update ps-step size control vector variable
   * 
   * @tparam MatType runtime matrix type 
   * @param iterate on-going optimizing point
   * @param ps step size control vector variable
   * @param B orthogonal basis matrix: results of eigendecomposing covariance matrix = B*D^2*B.t()
   * @param stepz vector of z[j]*weights(j)
   * @param mu_eff weights effective  
   */
  template<typename MatType, typename BaseMatType>
  MatType updatePs(
    MatType& iterate, 
    BaseMatType& ps, 
    BaseMatType& B,
    BaseMatType& sepCovsqinv,
    BaseMatType& v,
    std::vector<BaseMatType>& /** z **/,
    std::vector<BaseMatType>& y,
    double mu_eff)
  {
    BaseMatType step(iterate.n_rows, iterate.n_cols);
    step.zeros();

    for (size_t j = 0; j < mu; ++j)
    {
      step += weights(j) * y[idx(j)];
    }
    double csigma = (mu_eff + 2.0) / (iterate.n_elem + mu_eff + 5.0);
    BaseMatType upTerm = BaseMatType(iterate.n_rows, iterate.n_cols, fill:value(1.0));
    ps = (1 - csigma) * ps + std::sqrt(csigma * (2 - csigma) * mu_eff) * 
        ((1 / arma::sqrt(upTerm + v%v)) % (sepCovsqinv % step));
    return ps;
  }

  /**
   * This function will update pSigma
   * 
   * @tparam MatType runtime matrix type 
   * @param 
   */
  template<typename BaseMatType>
  BaseMatType updatePc(
    double cc,
    BaseMatType& pc,
    size_t hs,
    double mu_eff,
    std::vector<BaseMatType>& y)
  {
    BaseMatType step(iterate.n_rows, iterate.n_cols);
    step.zeros();

    for (size_t j = 0; j < mu; ++j)
    {
      step += weights(j) * y[idx(j)];
    }

    pc = (1 - cc) * pc + hs * std::sqrt(cc * (2 - cc) * mu_eff) * step; 
    return pc;
  }

  /**
   * This function will update pSigma
   * 
   * @param 
   */
  template<typename MatType, typename BaseMatType>
  BaseMatType updateC(
    MatType& iterate,
    double cc,
    double c1,
    double cmu,
    double mu_eff,
    size_t lambda,
    size_t hs,
    BaseMatType& C,
    BaseMatType& pc,
    arma::uvec& idx,
    std::vector<BaseMatType>& z,
    std::vector<BaseMatType>& y,
    arma::Row<double>& weights
    BaseMatType& sepCov,
    BaseMatType& sepCovsqinv,
    BaseMatType& v)
  {
    size_t mu = std::round(lambda / 2);
    double sqv = arma::accu(arma::pow(v, 2));
    BaseMatType vbar = v / sqrt(sqv);
    BaseMatType vbarbar = vbar % vbar;

    bool update = cmu + c1 * hs > 0;
    if (update)
    {
      double gammav = 1.0 + sqv;
      double alpha = std::sqrt(sqv*sqv + (2 - 1.0/std::sqrt(gammav))*gammav / 
          arma::max(vbarbar)) / (2.0 + sqv); // Eq(7)
      alpha = std::min(1.0, alpha);
      double b = -(1 - alpha*alpha) * sqv * sqv / gammav + 2.0*alpha*alpha;
      BaseMatType A = BaseMatType(iterate.n_rows, iterate.n_cols, fill:value(2.0)) - 
          (b + 2*alpha*alpha) * vbarbar;
      
      BaseMatType ym = sepCovsqinv % pc; // Dimension is same with y(i)
      arma::uvec yvbar(mu, fill::value(0.0));

      std::vector<BaseMatType> pvec(mu, BaseMatType(iterate.n_row, iterate.n_col));
      std::vector<BaseMatType> qvec = pvec;

      std::vector<BaseMatType> pone = pvec;
      std::vector<BaseMatType> qone = pvec;

      for (size_t i = 0; i < mu; i++)
      {
        yvbar(i) = dot(vbar, y[idx(i)]);
        pvec(i) = y[idx(i)] % y[idx(i)] - (sqv / gammav) * (yvbar(i) * vbar) % y[idx(i)];
        pvec(i) = (pvec(i) - BaseMatType(iterate.n_row, iterate.n_col, fill::value(1.0))) * weight(idx(i));
        qvec(i) = (yvbar(i) * y[idx(i)] - 0.5 * (yvbar(i) * yvbar(i) + gammav) * vbar) * weight(idx(i));

        pone(i) = ym % ym - (sqv / gammav) * (yvbar(i) * vbar) % ym;
        pone(i) = (pone(i) - BaseMatType(iterate.n_row, iterate.n_col, fill::value(1.0))) * weight(idx(i));
        qone(i) = (yvbar(i) * ym - 0.5 * (yvbar(i) * yvbar(i) + gammav) * vbar) * weight(idx(i));
      }
      for (size_t i = 0; i < mu; i++)
      {
        if (hs)
        {
          pvec = cmu * pvec + c1 * pone;
          qvec = cmu * qvec + c1 * qone;
        }
        else
        {
          pvec = _cmu * pvec;
          qvec = cmu * qvec;
        }
      }

      double nu = dot(vbar, qvec(0)); // maximum value
      std::vector<BaseMatType> rvec(mu, BaseMatType(iterate.n_row, iterate.n_col)); 
      for (size_t i = 0; i < mu; i++)
      {
        rvec(i) = pvec(i) - (alpha/gammav) * ((2.0+sqv) * (vbar % qvec(i)) - sqv*nu*vbarbar);
      }
      nu = arma::pow(A, -1) % vbarbar;
    }
  }

};

} // namespace ens

#endif