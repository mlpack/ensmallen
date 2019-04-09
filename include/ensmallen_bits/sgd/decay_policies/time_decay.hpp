/**
 * @file time_decay.hpp
 * @author Gaurav Sharma
 *
 * Time based decay policy for Stochastic Gradient Descent. 
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_SGD_DECAY_POLICIES_TIME_DECAY_HPP
#define ENSMALLEN_SGD_DECAY_POLICIES_TIME_DECAY_HPP

namespace ens {

/**
 * Definition of the TimeDecay class.
 */
class TimeDecay
{
 public:
 /**
  * This constructor is called before the first iteration.
  * @param decay Factor by which stepSize is decayed
  */
  TimeDecay(const double decay = 0.01) :
	decay(decay),
	epoch(0),
	effectiveBatchSize(1)
	{ /* Nothing to do here.*/ }

 /**
  * This function is called in each iteration after the policy update.
  * It sets the value of effective batch size.
  *
  * @param effectiveBatchSize current effective batch size.
  */	
  void setEffectiveBatchSize(const size_t& effBatchSize)
  {
		effectiveBatchSize = effBatchSize;
  }
	
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
	epoch += effectiveBatchSize;
	stepSize *= 1.0 / (1.0 + decay * epoch);
  }

  private:
	// Factor by which stepSize is decayed.
	double decay;
	
	// Current epoch.
	size_t epoch;	
	
	// Effective batch size.	
	size_t effectiveBatchSize;

}; // class TimeDecay
}  // namespace ens

#endif // ENSMALLEN_SGD_DECAY_POLICIES_TIME_DECAY_HPP
