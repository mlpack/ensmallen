/**
 * @file demon_sgd_update.hpp
 * @author Marcus Edel
 *
 * Implementation of DemonSGD.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_DEMON_SGD_DEMON_SGD_UPDATE_HPP
#define ENSMALLEN_DEMON_SGD_DEMON_SGD_UPDATE_HPP

namespace ens {

/**
 * DemonSGD automatically decays momentum, motivated by decaying the total
 * contribution of a gradient to all future updates. 
 *
 * For more information, see the following.
 *
 * @code
 * @misc{
 *   title   = {Decaying momentum helps neural network training},
 *   author  = {John Chen and Cameron Wolfe and Zhao Li
 *              and Anastasios Kyrillidis},
 *   url     = {https://arxiv.org/abs/1910.04952}
 *   year    = {2019}
 * }
 * @endcode
 */
class DemonSGDUpdate
{
 public:
  /**
   * Construct the DemonSGD update policy with the given parameters.
   *
   * @param momentumIterations The number of iterations before the momentum
   *     will decay to zero.
   * @param momentum The initial momentum coefficient.
   */
  DemonSGDUpdate(const size_t momentumIterations = 100,
                 const double momentum = 0.9) :
      T(momentumIterations),
      betaInit(momentum),
      t(0)
  {
    // Make sure the momentum iterations parameter is non-zero.
    assert(momentumIterations != 0 && "The number of iterations before the "
        "momentum will decay is zero, make sure the max iterations and "
        "batch size parameter is set correctly. "
        "Default: momentumIterations = maxIterations * batchSize.");
  }

  //! Get the momentum coefficient.
  double Momentum() const { return betaInit; }
  //! Modify the momentum coefficient.
  double& Momentum() { return betaInit; }

  //! Get the current iteration number.
  size_t Iteration() const { return t; }
  //! Modify the current iteration number.
  size_t& Iteration() { return t; }

  //! Get the momentum iteration number.
  size_t MomentumIterations() const { return T; }
  //! Modify the momentum iteration number.
  size_t& MomentumIterations() { return T; }

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
     * @param parent Instantiated PadamUpdate parent object.
     * @param rows Number of rows in the gradient matrix.
     * @param cols Number of columns in the gradient matrix.
     */
    Policy(DemonSGDUpdate& parent,
           const size_t /* rows */,
           const size_t /* cols */) :
      parent(parent)
    { /* Nothing to do here */ }

    /**
     * Update step for DemonSGD.
     *
     * @param iterate Parameters that minimize the function.
     * @param stepSize Step size to be used for the given iteration.
     * @param gradient The gradient matrix.
     */
    void Update(MatType& iterate,
                const double stepSize,
                const GradType& gradient)
    {
      double decayRate = 1;
      if (parent.t > 0)
        decayRate = 1.0 - (double) parent.t / (double) parent.T;

      const double betaDecay = parent.betaInit * decayRate;
      const double beta = betaDecay / ((1.0 - parent.betaInit) + betaDecay);

      // Perform the update.
      iterate *= beta;
      iterate -= stepSize * gradient;

      // Increment the iteration counter variable.
      ++parent.t;
    }

   private:
    //! Instantiated parent object.
    DemonSGDUpdate& parent;
  };

 private:
  //! The number of momentum iterations.
  size_t T;

  //! Initial momentum coefficient.
  double betaInit;

  //! The number of iterations.
  size_t t;
};

} // namespace ens

#endif
