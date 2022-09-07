/**
 * @file sep_update.hpp
 * @author John Hoang 
 * Sep-CMA: The covariance matrix is limited as a diagonal vector only C = D^2 with D is a vector
 * as proposed in Raymond Ros et al. in "A Simple Modification in CMA-ES Achieving Linear
 * Time and Space Complexitys".
 * 
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CMAES_SEP_UPDATE_HPP
#define ENSMALLEN_CMAES_SEP_UPDATE_HPP

namespace ens{

class SepUpdate{

 public:
  /**
   * Constructor
   * 
   */
  SepUpdate()
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
   * @param sepCovinv Root square of sepCov - for sampling purpose
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
    BaseMatType& sepCovinv,
    BaseMatType& /** sepCov **/,
    BaseMatType& /** v **/,
    arma::uvec& idx)
 {
    for (size_t j = 0; j < lambda; ++j)
    {
      // z_j ~ N(0, I)
      z[j] = arma::randn<BaseMatType>(iterate.n_rows, iterate.n_cols);
      y[idx(j)] = sepCovinv % z[j];
      // candidates_j ~ N(mean, sigma^2 * sepCovinv^2)
      candidates[idx(j)] = mCandidate + sigma * y[idx(j)];
    }
    return candidates;
  }

  /**
   * This function is not needed - vanila update not need to rescale param
   */
  template<typename MatType>
  void rescaleParam(MatType& /** iterate **/,
                    double& /** c1 **/,
                    double& /** cmu **/,
                    double& /** csigma **/,
                    double& /** mu_eff **/)
  {
    // Doing nothing
  }

  /**
   * This function will update ps-step size control vector variable
   * 
   * @tparam MatType runtime matrix type 
   * @tparam BaseMatType runtime base matrix type - either rowvector or column otherwise Mat
   * @param iterate on-going optimizing point
   * @param ps step size control vector variable - needed update
   * @param stepZ vector of z[j]*weights(j)
   * @param mu_eff weights effective  
   */
  template<typename MatType, typename BaseMatType>
  MatType updatePs(
    MatType& iterate, 
    BaseMatType& ps, 
    BaseMatType& /** B **/,
    BaseMatType& /** sepCovinv **/,
    BaseMatType& /** v **/,
    BaseMatType& stepZ,
    BaseMatType& /** stepY **/,
    double mu_eff)
  {
    double csigma = (mu_eff + 2.0) / (iterate.n_elem + mu_eff + 5.0);
    // B effectively I
    ps = (1 - csigma) * ps + std::sqrt(
        csigma * (2 - csigma) * mu_eff) * stepZ;
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
   * This function will update covariance matrix - sep-CMAES only update the diagonal
   * of the covariance matrix so we only need to update vector sepCov - diagonal vector
   * 
   * @tparam MatType runtime matrix type
   * @tparam BaseMatType runtime base matrix type - either rowvector or column otherwise Mat
   * @param iterate on-going optimizing point
   * @param mu_eff weights effective
   * @param lambda population size, sample size, number of offspring
   * @param pc evolution path - needed update 
   * @param idx uvec vector - the sorted indices of candidates vector according to their fitness/obejctive 
   * @param z vector of lambda sampled candidates ~ N(0,I)
   * @param y vector of lambda transformed candidates from z
   * @param weights vector of weights of candidates in mutation process
   * @param sepCov Diagonal vector of covariance matrix
   * @param sepCovinv Root square of sepCov - for sampling purpose
   */
  template<typename MatType, typename BaseMatType>
  void updateC(
    MatType& iterate,
    double /** cc **/,
    double /** c1 **/,
    double /** cmu **/,
    double mu_eff,
    size_t lambda,
    size_t /** hs **/,
    BaseMatType& /** C **/,
    BaseMatType& /** B **/,
    BaseMatType& /** D **/,
    BaseMatType& pc,
    arma::uvec& idx,
    std::vector<BaseMatType>& z,
    std::vector<BaseMatType>& y,
    arma::Row<double>& weights,
    BaseMatType& sepCov,
    BaseMatType& sepCovinv,
    BaseMatType& /** v **/,
    size_t& /** eigenval **/,
    size_t& /** countval **/)
  {
    mu_cov = mu_eff; // default value;
    c_cov = 1/mu_cov * 2/iterate.n_elem + (1 - 1/mu_cov) * 
        std::min((double)1, (2*mu_cov) / (std::pow(iterate.n_elem+2, 2) + mu_cov)); // default value

    sepCov = (1-c_cov) * sepCov + 1/mu_cov * c_cov * arma::pow(pc, 2);
    for (size_t j = 0; j < lambda; ++j)
    {
      if (weights(j) < 0) weights(j) *= iterate.n_elem / 
          std::pow(arma::norm(z[j]), 2);
      if (weights(j) == 0) break;
      sepCov = sepCov + c_cov * (1-1/mu_cov) * weights(j) * 
          arma::pow(y[idx(j)], 2);
    }
    sepCovinv = arma::sqrt(sepCov);
  }

  // Return variance-effective before the Generate function is called since c1 and cmu is 
  // calculated beforehand 
  double Mu_cov() const { return mu_cov; }
  double& Mu_cov() { return mu_cov; }

  // These functions might be unnecessary since Generate function is already return the desired weights 
  double C_cov() const { return c_cov; }
  double& C_cov() { return c_cov; }

 private:
  double c_cov;
  double mu_cov;
};

} // namespace ens

#endif