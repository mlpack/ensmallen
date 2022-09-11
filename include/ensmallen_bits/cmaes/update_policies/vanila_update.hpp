/**
 * @file vanila_update.hpp
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
#ifndef ENSMALLEN_CMAES_VANILA_UPDATE_HPP
#define ENSMALLEN_CMAES_VANILA_UPDATE_HPP

namespace ens {

class VanilaUpdate
{
 public:
  /**
   * Constructor
   * 
   */
  VanilaUpdate()
  {
    // Doing nothing.
  }

  /**
   * This function will sample z[j] firstly from Gaussian distribution, 
   *    transformed it into new child under new distribution by multiplying 
   *    square root of covariance matrix.
   *
   * @tparam MatType runtime matrix type
   * @tparam BaseMatType runtime base matrix type - either 
   *    rowvector or column otherwise Mat.
   * @param sigma step-size.
   * @param lambda population size, sample size, number of offspring.
   * @param iterate on-going optimizing point.
   * @param z storing container for new sampled candidates ~ N(0,I).
   * @param y storing container for new transformed candidates.
   * @param candidates storing container for new candidates
   *    following updated distribution.
   * @param mCandidate current population's mean vector
   * @param B Results of eigendecomposing covariance matrix = B*D^2*B.t().
   * @param D a diagonal matrix. The diagonal elements of D are
   *    square roots of eigenvalues of C.
   * @param idx uvec vector - the sorted indices of candidates vector
   *    according to their fitness/obejctive.
   */
  template<typename MatType, typename BaseMatType>
  std::vector<BaseMatType> SamplePop(
    double sigma,
    size_t lambda,
    MatType& iterate,
    std::vector<BaseMatType>& z,
    std::vector<BaseMatType>& y,
    std::vector<BaseMatType>& candidates,
    BaseMatType mCandidate,
    BaseMatType& B,
    BaseMatType& D,
    BaseMatType& /** sepCovinv **/,
    BaseMatType& /** sepCov **/,
    BaseMatType& /** v **/,
    arma::uvec& idx)
  {
    BaseMatType BD = B*D;
    for (size_t j = 0; j < lambda; ++j)
    {
      // z_j ~ N(0, I).
      if(iterate.n_rows > iterate.n_cols)
      {
        z[j] = arma::randn<BaseMatType>(iterate.n_rows, iterate.n_cols);
        y[idx(j)] = BD * z[j];
      }
      else
      {
        z[j] = arma::randn<BaseMatType>(iterate.n_rows, iterate.n_cols);
        y[idx(j)] = z[j] * BD.t();
      }
      candidates[idx(j)] = mCandidate + sigma * y[idx(j)];
    }
    return candidates;
  }

  /**
   * This function is not needed - vanila update not need to rescale parameters.
   */
  template<typename MatType>
  void RescaleParam(MatType& /** iterate **/,
                    double& /** c1 **/,
                    double& /** cmu **/,
                    double& /** csigma **/,
                    double& /** mu_eff **/)
  {
    // Doing nothing.
  }

  /**
   * This function will update ps-step size control vector variable.
   *
   * @tparam MatType runtime matrix type.
   * @tparam BaseMatType runtime base matrix type - either 
   *    rowvector or column otherwise Mat.
   * @param iterate on-going optimizing point.
   * @param ps step size control vector variable - needed update.
   * @param B results of eigendecomposing covariance matrix = B*D^2*B.t().
   * @param stepZ vector of z[j]*weights(j).
   * @param mu_eff weights effective.
   */
  template<typename MatType, typename BaseMatType>
  MatType UpdatePs(
    MatType& iterate,
    BaseMatType& ps,
    BaseMatType& B,
    BaseMatType& /** sepCovinv **/,
    BaseMatType& /** v **/,
    BaseMatType& stepZ,
    BaseMatType& /** stepY **/,
    double mu_eff)
  {
    double csigma = (mu_eff + 2.0) / (iterate.n_elem + mu_eff + 5.0);
    if (iterate.n_rows > iterate.n_cols)
    {
      ps = (1 - csigma) * ps + std::sqrt(
          csigma * (2 - csigma) * mu_eff) * B * stepZ;
    }
    else
    {
      ps = (1 - csigma) * ps + std::sqrt(
          csigma * (2 - csigma) * mu_eff) * stepZ * B.t();
    }  
    return ps;
  }

  /**
   * This function will update pc - evolution path vector.
   *
   * @tparam BaseMatType runtime base matrix type - either 
   *    rowvector or column otherwise Mat.
   * @param cc learning rate for cumulation of the rank-one update.
   * @param pc evolution path - needed update.
   * @param hs binary number prevent update pc if |ps| too large.
   * @param mu_eff weights effective.
   * @param stepY vector of y[j]*weights(j).
   */
  template<typename BaseMatType>
  BaseMatType UpdatePc(
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
   * This function will update covariance matrix C.
   *
   * @tparam MatType runtime matrix type.
   * @tparam BaseMatType runtime base matrix type - 
   *    either rowvector or column otherwise Mat.
   * @param iterate on-going optimizing point.
   * @param cc learning rate for cumulation of rank-one update.
   * @param c1 learning rate of rank-one update of the covariance matrix update.
   * @param cmu learning rate of rank-µ update of the covariance matrix update.
   * @param lambda population size, sample size, number of offspring.
   * @param hs binary number prevent update pc if |ps| too large.
   * @param C covariance matrix.
   * @param B results of eigendecomposing covariance matrix = B*D^2*B.t().
   * @param D The diagonal elements of D are square roots of eigenvalues of C.
   * @param pc evolution path - needed update.
   * @param idx the sorted indices of candidates vector according to their 
   *    fitness/obejctive.
   * @param z vector of lambda sampled candidates ~ N(0,I).
   * @param y vector of lambda transformed candidates from z.
   * @param weights vector of weights of candidates in mutation process.
   * @param eigenval
   * @param countval number of passed iteration.
   */
  template<typename MatType, typename BaseMatType>
  void UpdateC(
    MatType& iterate,
    double cc,
    double c1,
    double cmu,
    double /** mu_eff **/,
    size_t lambda,
    size_t hs,
    BaseMatType& C,
    BaseMatType& B,
    BaseMatType& D,
    BaseMatType& pc,
    arma::uvec& idx,
    std::vector<BaseMatType>& z,
    std::vector<BaseMatType>& y,
    arma::Row<double>& weights,
    BaseMatType& /** sepCov **/,
    BaseMatType& /** sepCovinv **/,
    BaseMatType& /** v **/,
    size_t& eigenval,
    size_t& countval)
  
  {
    double deltahs = (1 - hs) * cc * (2 - cc);
    C = (1 + c1 * deltahs - c1 - cmu * arma::accu(weights)) * C;
    if (iterate.n_rows > iterate.n_cols)
    {
      C = C + c1 * (pc * pc.t());
      for (size_t j = 0; j < lambda; ++j)
      {

        if (weights(j) < 0) weights(j) *= iterate.n_elem / 
            std::pow(arma::norm(z[j]), 2);
        if (weights(j) == 0) break;
        C = C + cmu * weights(j) *
            y[idx(j)] * y[idx(j)].t();
      }
    }
    else
    {
      C = C + c1 * (pc.t() * pc);
      for (size_t j = 0; j < lambda; ++j)
      {
        if (weights(j) < 0) weights(j) *= iterate.n_elem / 
            std::pow(arma::norm(z[j]), 2);
        if (weights(j) == 0) break;
        C = C + cmu * weights(j) *
            y[idx(j)].t() * y[idx(j)];
      }
    }  
    typedef typename MatType::elem_type ElemType;
    // To ensure that new result covariance matrix is positive definite.
    arma::Col<ElemType> eigval; // TODO: might need a more general type.
    BaseMatType eigvec;
    BaseMatType eigvalZero(iterate.n_elem, 1); // eigvalZero is vector-shaped.
    eigvalZero.zeros();

    if (countval - eigenval > lambda / ((c1 + cmu) * iterate.n_elem * 10))
    {
      eigenval = countval;
      C = arma::trimatu(C) + arma::trimatu(C).t();
      arma::eig_sym(eigval, eigvec, C);
      B = eigvec;
      D = arma::diagmat(arma::sqrt(eigval));
    }  
  }

};

} // namespace ens

#endif