/**
 * @file early_stop_at_min_loss.hpp
 * @author Marcus Edel
 *
 * Implementation of the early stop at minimum loss callback function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CALLBACKS_EARLY_STOP_AT_MIN_LOSS_HPP
#define ENSMALLEN_CALLBACKS_EARLY_STOP_AT_MIN_LOSS_HPP

namespace ens {

/**
 * Early stopping to terminate the optimization process early if the loss stops
 * decreasing.
 */
class EarlyStopAtMinLoss
{
 public:
  /**
   * Set up the early stop at min loss class, which keeps track of the minimum
   * loss and stops the optimization process if the loss stops decreasing.
   *
   * @param patienceIn The number of epochs to wait after the minimum loss has
   *    been reached or no improvement has been made (Default: 10).
   */
  EarlyStopAtMinLoss(const size_t patienceIn = 10) :
      patience(patienceIn),
      bestObjective(std::numeric_limits<double>::max()),
      steps(0)
  { /* Nothing to do here */ }

  /**
   * Callback function called at the end of a pass over the data.
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param epoch The index of the current epoch.
   * @param objective Objective value of the current point.
   */
  template<typename OptimizerType, typename FunctionType, typename MatType>
  bool EndEpoch(OptimizerType& /* optimizer */,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const size_t /* epoch */,
                const double objective)
  {
    if (objective < bestObjective)
    {
      steps = 0;
      bestObjective = objective;
      return false;
    }

    steps++;
    if (steps >= patience)
    {
      Info << "Minimum loss reached; terminate optimization." << std::endl;
      return true;
    }

    return false;
  }

 private:
  //! The number of epochs to wait before terminating the optimization process.
  size_t patience;

  //! Locally-stored best objective.
  double bestObjective;

  //! Locally-stored number of steps since the loss improved.
  size_t steps;
};

} // namespace ens

#endif
