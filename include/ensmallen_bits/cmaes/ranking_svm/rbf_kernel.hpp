/**
 * @file rbf_kernel.hpp
 * @author Suvarsha Chennareddy
 *
 * Implementation of Radial Basis Function Kernel.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CMAES_RBF_KERNEL_HPP
#define ENSMALLEN_CMAES_RBF_KERNEL_HPP


namespace ens {

/**
 * Compute the RBF Kernel value.
 *
 * @tparam CoordinateType The type of data (coordinate vectors).
 */
  template<typename CoordinateType = arma::mat>
  class RBFKernel
  {
  public:

   /**
    * Construct the RBF kernel.
    *
    * @param covarianceMatrixInv Inverse of the covariance matrix used
    *      to calculate the Mahalanobis distance.
    * @param sigma Sigma parameter used in the RBF kernel function.
    */
    RBFKernel(const CoordinateType& covarianceMatrixInv, 
              const typename CoordinateType::elem_type& sigma) :
      CInv(covarianceMatrixInv),
      sigma(sigma)
    {/* Nothing to do */}


   /**
    * Evaluate the RBF kernel at points x and y.
    *
    * @param x The first point.
    * @param y The second point.
    */
    typename CoordinateType::elem_type Evaluate(
      const CoordinateType& x,
      const CoordinateType& y)
    {
      // Compute the difference vector (x - y)
      CoordinateType diff(arma::size(x));

      for (size_t i = 0; i < x.n_elem; i++) {
        diff(i) = x(i) - y(i);
      }

      // Compute the value of the RBF kernel at x and y
      if (diff.n_rows > diff.n_cols) {
        return std::exp(-(arma::dot(diff, CInv * diff))/(2 * sigma * sigma));
      }
      return std::exp(-(arma::dot(diff, diff * CInv))/(2 * sigma * sigma));
      
    }
    

    //! Get the inverse of the covariance matrix.
    CoordinateType CovarianceMatrixInv() const
    { return CInv; }
    //! Modify the inverse of the covariance matrix.
    CoordinateType& CovarianceMatrixInv()
    { return CInv; }

    //! Get the sigma parameter.
    typename CoordinateType::elem_type Sigma() const
    { return sigma; }
    //! Modify the sigma parameter.
    typename CoordinateType::elem_type& Sigma()
    { return sigma; }

  private:

    // The inverse of the covariance matrix.
    CoordinateType CInv;

    //The sigma parameter.
    typename CoordinateType::elem_type sigma;

  };

} // namespace ens

#endif