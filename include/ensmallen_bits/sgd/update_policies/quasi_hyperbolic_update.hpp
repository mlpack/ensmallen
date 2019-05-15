/**
 * @file quasi_hyperbolic_update.hpp
 * @author Niteya Shah
 *
 * Quasi Hyperbolic Update for Stochastic Gradient Descent.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SGD_QH_UPDATE_HPP
#define ENSMALLEN_SGD_QH_UPDATE_HPP

namespace ens {

/**
 * Quasi Hyperbolic Update policy for Stochastic Gradient Descent (QHSGD).
 *
 * QHSGD is a update policy that is defined to be a parent of most update
 * policies (most update policies can reconstructed from QHSGD). This allows
 * this method to combine the features of many optimisers and provide better
 * optimisation control.
 *
 * @code
 * @inproceedings{ma2019qh,
 *   title={Quasi-hyperbolic momentum and Adam for deep learning},
 *   author={Jerry Ma and Denis Yarats},
 *   booktitle={International Conference on Learning Representations},
 *   year={2019}
 * }
 * @endcode
 */
class QHUpdate
{
 public:
  /**
   * Construct the Quasi Hyperbolic update policy with the given parameters.
   *
   * @param v The quasi hyperbolic term.
   * @param momentum The momentum term.
   */
  QHUpdate(const double v = 0.7,
           const double momentum = 0.999) :
       momentum(momentum),
       v(v)
  {
    // Nothing to do.
  }

  //! Get the value used to initialize the momentum coefficient.
  double Momentum() const { return momentum; }
  //! Modify the value used to initialize the momentum coefficient.
  double& Momentum() { return momentum; }

  //! Get the value used to initialize the momentum coefficient.
  double V() const { return v; }
  //! Modify the value used to initialize the momentum coefficient.
  double& V() { return v; }

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
     * This constructor is called by the SGD Optimize() method before the start
     * of the iteration update process.
     *
     * @param parent AdamUpdate object.
     * @param rows Number of rows in the gradient matrix.
     * @param cols Number of columns in the gradient matrix.
     */
    Policy(QHUpdate& parent, const size_t rows, const size_t cols) :
        parent(parent)
    {
      // Initialize an empty velocity matrix.
      velocity.zeros(rows, cols);
    }

    /**
     * Update step for QHSGD.
     *
     * @param iterate Parameters that minimize the function.
     * @param stepSize Step size to be used for the given iteration.
     * @param gradient The gradient matrix.
     */
    void Update(MatType& iterate,
                const double stepSize,
                const GradType& gradient)
    {
      velocity *= parent.momentum;
      velocity += (1 - parent.momentum) * gradient;

      iterate -= stepSize * ((1 - parent.v) * gradient + parent.v * velocity);
    }

   private:
    //! Instantiated parent object.
    QHUpdate& parent;

    //! The velocity matrix.
    GradType velocity;
  };

 private:
  // The momentum coefficient.
  double momentum;

  //The quasi-hyperbolic term.
  double v;
};

} // namespace ens

#endif
