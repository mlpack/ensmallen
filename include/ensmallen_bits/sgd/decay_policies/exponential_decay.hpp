/**
 * @file exponential_decay.hpp
 * @author Gaurav Sharma
 *
 * Exponential based decay policy for Stochastic Gradient Descent. 
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_SGD_DECAY_POLICIES_EXPONENTIAL_DECAY_HPP
#define ENSMALLEN_SGD_DECAY_POLICIES_EXPONENTIAL_DECAY_HPP

namespace ens {

/**
 * Definition of the ExponentialDecay class.
 */
class ExponentialDecay
{
 public:
  /**
   * This constructor is called before the first iteration. The
   * defaults here are not necessarily good for the given problem,
   * so it is suggested that the values used be tailored to the
   * task at hand.
   *
   * @param initialStepSize Step Size at the beginning.
   * @param decayRate Rate at which stepSize is decayed.
   */
  ExponentialDecay(const double initialStepSize = 0.01,
		   const double decayRate = 0.1) :
	initialStepSize(initialStepSize),
	decayRate(decayRate),
	epoch(0)
	{/* Nothing to do here. */ }

  /**
   * This function is called in each iteration after the policy update.
   *
   * @param iterate Parameters that minimize the function.
   * @param stepSize Step size to be used for the given iteration.
   * @param gradient The gradient matrix.
   */
  void Update(arma::mat& /* iterate */,
              double& stepSize,
              const arma::mat& /* gradient */)
  {
    stepSize = initialStepSize * exp(-decayRate * epoch);
    ++epoch;
  }

  private:
	// stepSize at the beginning. 
  	double initialStepSize;

	// Rate at which stepSize is decayed.
	double decayRate;

	// Current epoch.
	size_t epoch;
}; // class ExponentialDecay

}  // namespace ens

#endif // ENSMALLEN_SGD_DECAY_POLICIES_EXPONENTIAL_DECAY_HPP
