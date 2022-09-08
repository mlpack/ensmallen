/**
 * @file vd_update.hpp
 * @author John Hoang 
 * 
 * VD-CMA: Linear Time/Space Comparison-based Natural Gradient Optimization
 * The covariance matrix is limited as C = D * (I + v*v^t) * D,
 * where D is a diagonal, v is a vector.
 * Reference
 * ---------
 * Youhei Akimoto, Anne Auger, and Nikolaus Hansen.
 * Comparison-Based Natural Gradient Optimization in High Dimension.
 * In Proc. of GECCO 2014, pp. 373 -- 380 (2014)
 * 
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
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

  /**
   * This function will sample z[j] firstly from Gaussian distribution, transformed it
   * into new child under new distribution by multiplying square root of covariance matrix
   * 
   * @tparam MatType runtime matrix type
   * @tparam BaseMatType runtime base matrix type - either rowvector or column otherwise Mat
   * @param sigma step-size 
   * @param lambda population size, sample size, number of offspring
   * @param iterate on-going optimizing point
   * @param z storing container for new sampled candidates ~ N(0,I)
   * @param y storing container for new transformed candidates 
   * @param candidates storing container for new candidates following updated distribution
   * @param mCandidate current population's mean vector
   * @param sepCov Root square of covariance matrix - for sampling purpose
   * @param v new introduced parameter vector in VD-CMAES C = D(I+vv.t())D
   * @param idx uvec vector - the sorted indices of candidates vector according to their fitness/obejctive 
   */
  template<typename MatType, typename BaseMatType>
  std::vector<BaseMatType> samplePop(
    double sigma,
    size_t lambda,
    MatType& iterate,
    std::vector<BaseMatType>& z,
    std::vector<BaseMatType>& y,
    std::vector<BaseMatType>& candidates,
    BaseMatType mCandidate,
    BaseMatType& /** B **/,
    BaseMatType& /** D **/, 
    BaseMatType& /** sepCovinv **/,
    BaseMatType& sepCov,
    BaseMatType& v,
    arma::uvec& idx)
  {
    double sqv = arma::accu(arma::pow(v, 2));
    double fact = std::sqrt(1 + sqv) - 1.0;
	  BaseMatType vbar = v / std::sqrt(sqv);
    for (size_t j = 0; j < lambda; ++j)
    {
      // z_j ~ N(0, I)
      z[j] = arma::randn<BaseMatType>(iterate.n_rows, iterate.n_cols);
      // y_j ~ N(0, I + vv^t)
      y[idx(j)] = z[j] + fact * arma::dot(vbar, z[j]) * vbar;
      // x_j ~ N(x_mean, sigma * D(I+vv^t)D)
      candidates[idx(j)] = mCandidate + sigma * (sepCov % y[idx(j)]);
    }
    return candidates;
  }

  /**
   * This function will rescale the learning rate parameters for Vd-Update
   * 
   * @tparam MatType runtime matrix type
   * @tparam BaseMatType runtime base matrix type - either rowvector or column otherwise Mat
   * @param iterate on-going optimizing point
   * @param c1 learning rate for the rank-one update of the covariance matrix update
   * @param cmu  learning rate for the rank-µ update of the covariance matrix update
   * @param csigma learning rate for step-size control
   * @param mu_eff effective of weights vector  
   */
  template<typename MatType>
  void rescaleParam(MatType& iterate,
                    double& c1,
                    double& cmu,
                    double& csigma,
                    double& mu_eff)
  {
    // New setting for VD update - Effective when dimension is large
    double cfactor = std::max((iterate.n_elem - 5.0) / 6.0, 0.5);
    c1 = cfactor * c1; 
    cmu = std::min(1 - c1, cfactor * 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / 
        (std::pow(iterate.n_elem + 2.0, 2) + mu_eff));
    csigma = std::sqrt(mu_eff) / (2*(std::sqrt(iterate.n_elem) + std::sqrt(mu_eff)));
  }
  
  /**
   * This function will update ps-step size control vector variable
   * 
   * @tparam MatType runtime matrix type 
   * @param iterate on-going optimizing point
   * @param ps step size control vector variable
   * @param sepCovinv Root square of sepCov - for sampling purpose
   * @param v new introduced parameter vector in VD-CMAES C = D(I+vv.t())D
   * @param stepY vector of y[j]*weights(j)
   * @param mu_eff weights effective  
   */
  template<typename MatType, typename BaseMatType>
  MatType updatePs(
    MatType& iterate, 
    BaseMatType& ps, 
    BaseMatType& /** B **/,
    BaseMatType& sepCovinv,
    BaseMatType& v,
    BaseMatType& /** stepZ **/,
    BaseMatType& stepY,
    double mu_eff)
  {
    double csigma = (mu_eff + 2.0) / (iterate.n_elem + mu_eff + 5.0);
    BaseMatType upTerm = BaseMatType(iterate.n_rows, iterate.n_cols, arma::fill::ones);
    ps = (1 - csigma) * ps + std::sqrt(csigma * (2 - csigma) * mu_eff) * 
        ((1 / arma::sqrt(upTerm + v%v)) % (sepCovinv % stepY));
    return ps;
  }

  /**
   * This function will update pc - evolution path vector 
   * 
   * @tparam BaseMatType runtime base matrix type - either rowvector or column otherwise Mat
   * @param cc learning rate for cumulation for the rank-one update of the covariance matrix
   * @param pc evolution path - needed update 
   * @param hs binary number prevent update pc if |ps| too large - refer to its formulate
   * @param mu_eff weights effective
   * @param stepY vector of y[j]*weights(j)
   */
  template<typename BaseMatType>
  BaseMatType updatePc(
    double cc,
    BaseMatType& pc,
    size_t hs,
    double mu_eff,
    BaseMatType& stepY)
  {
    pc = (1 - cc) * pc + hs * std::sqrt(cc * (2 - cc) * mu_eff) * stepY; 
    return pc;
  }

  /**
   * This function will update sepCov vector - square root of diagonal of covariance matrix,
   * since VD-CMAES only update the diagonal elements and 2*dimension other elements vv.t()
   * 
   * @tparam MatType runtime matrix type
   * @tparam BaseMatType runtime base matrix type - either rowvector or column otherwise Mat
   * @param iterate on-going optimizing point
   * @param c1 learning rate for the rank-one update of the covariance matrix update
   * @param cmu  learning rate for the rank-µ update of the covariance matrix update
   * @param mu_eff effective of weights vector
   * @param lambda population size, sample size, number of offspring
   * @param hs binary number prevent update pc if |ps| too large - refer to its formulate
   * @param pc evolution path - needed update 
   * @param idx uvec vector - the sorted indices of candidates vector according to their fitness/obejctive 
   * @param y vector of lambda transformed candidates from z
   * @param weights vector of weights of candidates in mutation process
   * @param sepCov Diagonal vector of covariance matrix
   * @param sepCovinv Root square of sepCov - for sampling purpose
   * @param v new introduced parameter vector in VD-CMAES C = D(I+vv.t())D
   */
  template<typename MatType, typename BaseMatType>
  void updateC(
    MatType& iterate,
    double /** cc **/,
    double c1,
    double cmu,
    double /** mu_eff **/,
    size_t lambda,
    size_t hs,
    BaseMatType& /** C **/,
    BaseMatType& /** B **/,
    BaseMatType& /** D **/,
    BaseMatType& pc,
    arma::uvec& idx,
    std::vector<BaseMatType>& /** z **/,
    std::vector<BaseMatType>& y,
    arma::Row<double>& weights,
    BaseMatType& sepCov,
    BaseMatType& sepCovinv,
    BaseMatType& v,
    size_t& /** eigenval **/,
    size_t& /** countval **/)
  {
    size_t mu = std::round(lambda / 2);
    double sqv = arma::accu(arma::pow(v, 2));
    BaseMatType vbar = v / std::sqrt(sqv);
    BaseMatType vbarbar = vbar % vbar;
    // Useful matrix
    BaseMatType idenMat(iterate.n_rows, iterate.n_cols, arma::fill::ones);

    // TODO: changing csigma as in the paper too: \sqrt(mu_eff)/(2(dim + \sqrt(mu_eff))) -
    // problem: csigma is out of this scope function
    if ((cmu + c1) > 0)
    {
      double gammav = 1.0 + sqv;
      double maxvbarbar = arma::max(arma::conv_to<arma::uvec>::from(vbarbar));
      double alpha = sqv*sqv + (2.0 - 1.0/std::sqrt(gammav)) * gammav / 
          maxvbarbar; // Eq(7)
      alpha = std::min(1.0, std::sqrt(alpha) / (2.0 + sqv));
      double b = -(1 - alpha*alpha) * sqv * sqv / gammav + 2.0 * alpha*alpha;

      BaseMatType A = 2.0 * idenMat - (b + 2.0 * alpha*alpha) * vbarbar;
      
      BaseMatType ym = sepCovinv % pc; // Dimension is same with y(i)
      arma::uvec yvbar(mu, arma::fill::zeros);

      std::vector<BaseMatType> pmat(mu, BaseMatType(iterate.n_rows, iterate.n_cols));
      std::vector<BaseMatType> qmat = pmat;

      BaseMatType pvec(iterate.n_rows, iterate.n_cols, arma::fill::zeros);
      BaseMatType qvec = pvec;
      BaseMatType ponevec = pvec;
      BaseMatType qonevec = pvec;

      for (size_t i = 0; i < mu; i++)
      {
        yvbar(i) = dot(vbar, y[idx(i)]);
        pmat[i] = y[idx(i)] % y[idx(i)] - (sqv / gammav * yvbar(i)) * (vbar % y[idx(i)]);
        pmat[i] = (pmat[i] - idenMat) * weights(idx(i));
        pvec += pmat[i];
        qmat[i] = (yvbar(i) * y[idx(i)] - 0.5 * (yvbar(i) * yvbar(i) + gammav) * vbar) * weights(idx(i));
        qvec += qmat[i];
      }
      // ponevec and qonevec shape is same like iterate 
      double ymvbar = arma::dot(vbar, ym);

      ponevec = ym % ym - (sqv / gammav) * ymvbar * (vbar % ym) - idenMat;
      qonevec = ymvbar * ym - 0.5 * (ymvbar * ymvbar + gammav) * vbar;

      pvec = cmu * pvec + hs * c1 * ponevec;
      qvec = cmu * qvec + hs * c1 * qonevec;
      
      BaseMatType Ainvbb = vbarbar / A;
      // Reusable variable
      double nu = arma::dot(vbar, qvec); 
      BaseMatType rvec = pvec - (alpha/gammav) * ((2.0 + sqv) * (vbar % qvec) - sqv * nu * vbarbar);

      double nu2 = arma::dot(Ainvbb, vbarbar);
      BaseMatType svec = rvec / A - (b * arma::dot(Ainvbb, rvec) /(1.0 + b * nu2)) * Ainvbb;

      nu = arma::dot(svec, vbarbar);
      BaseMatType ngv = (qvec - alpha * ((2+sqv) * (vbar % svec) - nu * vbar)) / std::sqrt(sqv);
      BaseMatType ngd = sepCov % svec;

      double upfactor = 1.0;
      upfactor = std::min(upfactor, 0.7 * std::sqrt(sqv) / std::sqrt(arma::dot(ngv, ngv)));
      double minEffCov = arma::min(arma::conv_to<arma::uvec>::from(sepCov / arma::abs(ngd)));
      upfactor = std::min(upfactor, 0.7 * minEffCov);
      
      v = v + upfactor * ngv;
      sepCov = sepCov + upfactor * ngd;
      sepCovinv = arma::pow(sepCov, -1);
    }
  }

};

} // namespace ens

#endif