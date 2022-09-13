/**
 * @file sep_update.hpp
 * @author John Hoang 
 *
 * Sep-CMA: The covariance matrix is limited as a diagonal vector only C = D^2
 *  with D is a vector as proposed in Raymond Ros et al. in "A Simple 
 * Modification in CMA-ES Achieving Linear Time and Space Complexitys".
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
    // Doing nothing.
  }

  /**
   * This function will sample z[j] firstly from Gaussian distribution, 
   *    transformed it into new child under new distribution by multiplying 
   *    square root of covariance matrix.
   *
   * @tparam MatType runtime matrix type.
   * @tparam BaseMatType runtime base matrix type - either 
   *    rowvector or column otherwise Mat.
   * @param sigma step-size.
   * @param lambda population size, sample size, number of offspring.
   * @param iterate on-going optimizing point.
   * @param z storing container for new sampled candidates ~ N(0,I).
   * @param y storing container for new transformed candidates.
   * @param candidates storing container for new candidates following 
   *    updated distribution.
   * @param mCandidate current population's mean vector.
   * @param sepCovinv Root square of sepCov - for sampling purpose.
   * @param idx the sorted indices of candidates vector 
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
    BaseMatType& /** B **/,
    BaseMatType& /** D **/,
    BaseMatType& sepCovinv,
    BaseMatType& /** sepCov **/,
    BaseMatType& /** v **/,
    arma::uvec& idx)
 {
    for (size_t j = 0; j < lambda; ++j)
    {
      // z_j ~ N(0, I).
      z[j] = arma::randn<BaseMatType>(iterate.n_rows, iterate.n_cols);
      y[idx(j)] = sepCovinv % z[j];
      // candidates_j ~ N(mean, sigma^2 * sepCovinv^2).
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
                    double& /** mueff **/)
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
   * @param stepZ vector of z[j]*weights(j).
   * @param mueff weights effective.
   */
  template<typename MatType, typename BaseMatType>
  MatType UpdatePs(
    MatType& iterate,
    BaseMatType& ps,
    BaseMatType& /** B **/,
    BaseMatType& /** sepCovinv **/,
    BaseMatType& /** v **/,
    BaseMatType& stepZ,
    BaseMatType& /** stepY **/,
    double mueff)
  {
    double csigma = (mueff + 2.0) / (iterate.n_elem + mueff + 5.0);
    // B effectively I.
    ps = (1 - csigma) * ps + std::sqrt(
        csigma * (2 - csigma) * mueff) * stepZ;
    return ps;
  }

  /**
   * This function will update pc - evolution path vector.
   *
   * @tparam BaseMatType runtime base matrix type - either 
   *    rowvector or column otherwise Mat.
   * @param cc learning rate for cumulation of rank-one 
   *    update of the covariance matrix.
   * @param pc evolution path - needed update.
   * @param hs binary number prevent update pc if |ps| too large.
   * @param mueff weights effective.
   * @param stepY vector of y[j]*weights(j).
   */
  template<typename BaseMatType>
  BaseMatType UpdatePc(
    double cc,
    BaseMatType& pc,
    size_t hs,
    double mueff,
    BaseMatType& stepY)
  {
    pc = (1 - cc) * pc + hs * std::sqrt(cc * (2 - cc) * mueff) * stepY;
    return pc;
  }

  /**
   * This function will update covariance matrix - sep-CMAES only 
   *    update the diagonal of the covariance matrix so we only need to 
   *    update vector sepCov.
   *
   * @tparam MatType runtime matrix type.
   * @tparam BaseMatType runtime base matrix type - either 
   *    rowvector or column otherwise Mat.
   * @param iterate on-going optimizing point.
   * @param mueff weights effective.
   * @param lambda population size, sample size, number of offspring.
   * @param pc evolution path - needed update.
   * @param idx uvec vector - the sorted indices of candidates vector 
   *    according to their fitness/obejctive.
   * @param z vector of lambda sampled candidates ~ N(0,I).
   * @param y vector of lambda transformed candidates from z.
   * @param weights vector of weights of candidates in mutation process.
   * @param sepCov Diagonal vector of covariance matrix.
   * @param sepCovinv Root square of sepCov - for sampling purpose.
   */
  template<typename MatType, typename BaseMatType>
  void UpdateC(
    MatType& iterate,
    double /** cc **/,
    double /** c1 **/,
    double /** cmu **/,
    double mueff,
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
    mucov = mueff; // default value.
    ccov = 1/mucov * 2/iterate.n_elem + (1 - 1/mucov) * std::min(1.0, 
        (2*mucov) / (std::pow(iterate.n_elem+2, 2) + mucov)); 

    sepCov = (1-ccov) * sepCov + 1/mucov * ccov * arma::pow(pc, 2);
    for (size_t j = 0; j < lambda; ++j)
    {
      if (weights(j) < 0) weights(j) *= iterate.n_elem / 
          std::pow(arma::norm(z[j]), 2);
      if (weights(j) == 0) break;
      sepCov = sepCov + ccov * (1-1/mucov) * weights(j) * 
          arma::pow(y[idx(j)], 2);
    }
    sepCovinv = arma::sqrt(sepCov);
  }

  // Return the modified effective-variance of sep-CMAES.
  double MuCov() const { return mucov; }
  double& MuCov() { return mucov; }

  // Return learning rate for covariance matrix update.
  double Ccov() const { return ccov; }
  double& Ccov() { return ccov; }

 private:
  double ccov;
  double mucov;
};

} // namespace ens

#endif