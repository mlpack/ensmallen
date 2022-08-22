/**
 * @file sep_update.hpp
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
    BaseMatType& stepz,
    double mu_eff)
  {
    double csigma = (mu_eff + 2.0) / (iterate.n_elem + mu_eff + 5.0);
    if (iterate.n_rows > iterate.n_cols)
    {
      ps = (1 - csigma) * ps + std::sqrt(
          csigma * (2 - csigma) * mu_eff) * B * stepz;
    }
    else
    {
      ps = (1 - csigma) * ps + std::sqrt(
          csigma * (2 - csigma) * mu_eff) * stepz * B.t();
    }  
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
    BaseMatType& step)
  {
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
    double /** cc **/,
    double /** c1 **/,
    double /** cmu **/,
    double mu_eff,
    size_t lambda,
    size_t /** hs **/,
    BaseMatType& C,
    BaseMatType& pc,
    arma::uvec& idx,
    std::vector<BaseMatType>& z,
    std::vector<BaseMatType>& pStep,
    arma::Row<double>& weights)
  {
    double mu_cov = mu_eff; // default value;
    double c_cov = 1/mu_cov * 2/iterate.n_elem + (1 - 1/mu_cov) * 
        std::min((double)1, (2*mu_cov) / (std::pow(iterate.n_elem+2, 2) + mu_cov)); // default value

    C.diag() = (1-c_cov) * C.diag() + 1/mu_cov * c_cov * arma::pow(pc, 2);
    for (size_t j = 0; j < lambda; ++j)
    {
      if (weights(j) < 0) weights(j) *= iterate.n_elem / 
          std::pow(arma::norm(z[j]), 2);
      if (weights(j) == 0) break;
      C.diag() += + c_cov * (1-1/mu_cov) * weights(j) * 
          arma::pow(pStep[idx(j)], 2);
    }
    return C;
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