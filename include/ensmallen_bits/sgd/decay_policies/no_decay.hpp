/**
 * @file no_decay.hpp
 * @author Marcus Edel
 *
 * Definition of the policy type for the decay class.
 *
 * You should define your own decay update that looks like NoDecay.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_SGD_DECAY_POLICIES_NO_DECAY_HPP
#define ENSMALLEN_SGD_DECAY_POLICIES_NO_DECAY_HPP

namespace ens {

/**
 * Definition of the NoDecay class. Use this as a template for your own.
 */
class NoDecay
{
 public:
  /**
   * This constructor is called before the first iteration.
   */
  NoDecay() { }

  /**
   * The DecayPolicyType policy classes must contain an internal 'Policy'
   * template class with two template arguments: MatType and GradType.  This is
   * initialized at the start of the optimization, and holds parameters specific
   * to an individual optimization.
   */
  template<typename MatType, typename GradType>
  class Policy
  {
   public:
    /**
     * This constructor is called by the SGD Optimize() method before the start
     * of the iteration update process.
     */
    Policy(NoDecay& /* parent */) { }

    /**
     * This function is called in each iteration after the policy update.
     *
     * @param iterate Parameters that minimize the function.
     * @param stepSize Step size to be used for the given iteration.
     * @param gradient The gradient matrix.
     */
    void Update(MatType& /* iterate */,
                double& /* stepSize */,
                const GradType& /* gradient */)
    {
      // Nothing to do here.
    }

    /**
     * This function is called in each iteration after the SVRG update step.
     *
     * @param iterate Parameters that minimize the function.
     * @param iterate0 The last function parameters at time t - 1.
     * @param gradient The current gradient matrix at time t.
     * @param fullGradient The computed full gradient.
     * @param stepSize Step size to be used for the given iteration.
     */
    void Update(const MatType& /* iterate */,
                const MatType& /* iterate0 */,
                const GradType& /* gradient */,
                const GradType& /* fullGradient */,
                const size_t /* numBatches */,
                double& /* stepSize */)
    {
      // Nothing to do here.
    }
  };
};

} // namespace ens

#endif // ENSMALLEN_SGD_DECAY_POLICIES_NO_DECAY_HPP
