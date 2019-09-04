/**
 * @file gradient_clipping.hpp
 * @author Konstantin Sidorov
 *
 * Gradient clipping update wrapper.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SGD_GRADIENT_CLIPPING_HPP
#define ENSMALLEN_SGD_GRADIENT_CLIPPING_HPP

namespace ens {

/**
 * Interface for wrapping around update policies (e.g., VanillaUpdate)
 * and feeding a clipped gradient to them instead of the normal one.
 * (Clipping here is implemented as
 * \f$ g_{\text{clipped}} = \max(g_{\text{min}}, \min(g_{\text{min}}, g))) \f$.)
 *
 * @tparam UpdatePolicy A type of UpdatePolicy that sould be wrapped around.
 */
template<typename UpdatePolicyType>
class GradientClipping
{
 public:
  /**
   * Constructor for creating a GradientClipping instance.
   *
   * @param minGradient Minimum possible value of gradient element.
   * @param maxGradient Maximum possible value of gradient element.
   * @param updatePolicy An instance of the UpdatePolicyType
   *                     used for actual optimization.
   */
  GradientClipping(const double minGradient,
                   const double maxGradient,
                   UpdatePolicyType& updatePolicy) :
      minGradient(minGradient),
      maxGradient(maxGradient),
      updatePolicy(updatePolicy)
  {
    // Nothing to do here.
  }

  //! Get the update policy.
  UpdatePolicyType& UpdatePolicy() const { return updatePolicy; }
  //! Modify the update policy.
  UpdatePolicyType& UpdatePolicy() { return updatePolicy; }

  //! Get the minimum gradient value.
  double MinGradient() const { return minGradient; }
  //! Modify the minimum gradient value.
  double& MinGradient() { return minGradient; }

  //! Get the maximum gradient value.
  double MaxGradient() const { return maxGradient; }
  //! Modify the maximum gradient value.
  double& MaxGradient() { return maxGradient; }

  /**
   * The UpdatePolicyType policy classes must contain an internal 'Policy'
   * template class with two template arguments: MatType and GradType.  This is
   * instantiated at the start of the optimization, and holds parameters
   * specific to an individual optimization.
   */
  template<typename MatType, typename GradType>
  class Policy
  {
   public:
    /**
     * This is called by the optimizer method before the start of the iteration
     * update process.
     *
     * @param parent Instantiated parent class.
     * @param rows Number of rows in the gradient matrix.
     * @param cols Number of columns in the gradient matrix.
     */
    Policy(const GradientClipping<UpdatePolicyType>& parent,
           const size_t rows,
           const size_t cols) :
        parent(parent),
        instPolicy(parent.UpdatePolicy(), rows, cols)
    {
      // Nothing to do.
    }

    /**
     * Update step. First, the gradient is clipped, and then the actual update
     * policy does whatever update it needs to do.
     *
     * @param iterate Parameters that minimize the function.
     * @param stepSize Step size to be used for the given iteration.
     * @param gradient The gradient matrix.
     */
    void Update(MatType& iterate,
                const double stepSize,
                const GradType& gradient)
    {
      typedef typename GradType::elem_type GradElemType;

      // First, clip the gradient.
      GradType clippedGradient = arma::clamp(gradient,
          GradElemType(parent.minGradient),
          GradElemType(parent.maxGradient));

      // And only then do the update.
      instPolicy.Update(iterate, stepSize, clippedGradient);
    }

   private:
    // The instantiated parent class.
    const GradientClipping<UpdatePolicyType>& parent;
    // The instantiated update policy we will use.
    typename UpdatePolicyType::template Policy<MatType, GradType> instPolicy;
  };

 private:
  //! Minimum possible value of gradient element.
  double minGradient;

  //! Maximum possible value of gradient element.
  double maxGradient;

  //! An instance of the UpdatePolicy used for actual optimization.
  UpdatePolicyType updatePolicy;
};

} // namespace ens

#endif
