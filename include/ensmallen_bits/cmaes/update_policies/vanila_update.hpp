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

namespace ens{

class VanilaUpdate{

 public:
  /**
   * Constructor
   * 
   */
  VanilaUpdate()
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
  MatType updatePC(
    MatType& iterate, 
    BaseMatType& ps, 
    BaseMatType& B,
    BaseMatType& stepz,
    double mu_eff)
  {
    double csigma = (mu_eff + 2.0)/(iterate.n_elem+mu_eff+5.0);;
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
  BaseMatType updatePS(
    double cc,
    BaseMatType &pc,
    size_t hs,
    double mu_eff,
    BaseMatType &step
  )
  {
    pc = (1 - cc) * pc + hs * std::sqrt(cc * (2 - cc) * mu_eff) * step; 
    return pc;
  }

  /**
   * This function will update pSigma
   * 
   * @param 
   */
  void updateC()
  {

  }

  private:

};

} // namespace ens

#endif