/**
 * @file drop_decay.hpp
 * @author Gaurav Sharma
 *
 * Drop based decay policy for Stochastic Gradient Descent. 
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_SGD_DECAY_POLICIES_DROP_DECAY_HPP
#define ENSMALLEN_SGD_DECAY_POLICIES_DROP_DECAY_HPP

namespace ens {

/**
 * Definition of the DropDecay class.
 */
class DropDecay
{
 public: 
  /**
   * This constructor is called before the first iteration.The
   * defaults here are not necessarily good for the given problem,
   * so it is suggested that the values used be tailored to the
   * task at hand.
   *
   * @param initialStepSize Step Size at the beginning.
   * @param dropRate Factor by which stepSize is dropped.
   * @param epochDrop Number of epochs after which stepSize is dropped.
   */
  DropDecay(const double initialStepSize = 0.01,
	    const double dropRate = 0.99,
	    const size_t epochDrop = 10000) :
	initialStepSize(initialStepSize),
	dropRate(dropRate),
	epochDrop(epochDrop),
	epoch(0)
	{ /* Nothing to do here. */}

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
    stepSize = initialStepSize * pow(dropRate, floor((1.0 + epoch) / epochDrop));
    ++epoch;
  }

  private:
	// stepSize at the beginning.
	double initialStepSize;

	// Factor by which stepSize is dropped.
	double dropRate;

	// Number of epochs after which stepSize is dropped.
	size_t epochDrop;

	// Current epoch.
	size_t epoch;
}; // class DropDecay

}  // namespace ens

#endif // ENSMALLEN_SGD_DECAY_POLICIES_DROP_DECAY_HPP
