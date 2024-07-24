/**
 * @file early_stop_at_min_loss.hpp
 * @author Marcus Edel
 * @author Omar Shrit
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

#include <functional>

namespace ens {

/**
 * Early stopping to terminate the optimization process early if the loss stops
 * decreasing.
 */
template<typename MatType = arma::mat>
class EarlyStopAtMinLossType
{
 public:
  /**
   * Set up the early stop at min loss class, which keeps track of the minimum
   * loss and stops the optimization process if the loss stops decreasing.
   *
   * @param patienceIn The number of epochs to wait after the minimum loss has
   *    been reached or no improvement has been made (Default: 10).
   */
  EarlyStopAtMinLossType(const size_t patienceIn = 10) :
      callbackUsed(false),
      patience(patienceIn),
      bestObjective(std::numeric_limits<double>::max()),
      steps(0)
  { /* Nothing to do here */ }

  /**
   * Set up the early stop at min loss class, which keeps track of the minimum
   * loss and stops the optimization process if the loss stops decreasing.
   *
   * @param func, callback to return immediate loss evaluated by the function
   * @param patienceIn The number of epochs to wait after the minimum loss has
   *    been reached or no improvement has been made (Default: 10).
   */
  EarlyStopAtMinLossType(
      std::function<double(const MatType&)> func,
      const size_t patienceIn = 10)
    : callbackUsed(true),
      patience(patienceIn),
      bestObjective(std::numeric_limits<double>::max()),
      steps(0),
      localFunc(func)
  {
    // Nothing to do here
  }

  /**
   * Callback function called at the end of a pass over the data.
   *
   * @param optimizer The optimizer used to update the function.
   * @param function Function to optimize.
   * @param coordinates Starting point.
   * @param epoch The index of the current epoch.
   * @param objective Objective value of the current point.
   */
  template<typename OptimizerType, typename FunctionType>
  bool EndEpoch(OptimizerType& /* optimizer */,
                FunctionType& /* function */,
                const MatType& coordinates,
                const size_t /* epoch */,
                double objective)
  {
    if (callbackUsed)
    {
      objective = localFunc(coordinates);
    }

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
  //! False if the first constructor is called, true if the user passed a
  //! lambda.
  bool callbackUsed;

  //! The number of epochs to wait before terminating the optimization process.
  size_t patience;

  //! Locally-stored best objective.
  double bestObjective;

  //! Locally-stored number of steps since the loss improved.
  size_t steps;

  //! Function to call at the end of the epoch.
  std::function<double(const MatType&)> localFunc;
};

/*
 * Note that the using definition is temporary, this definition should
 * be removed when releasing ensmallen 3.0
 * The renaming of the class is only to avoid a major version bump
 * because if the template type added to this class
 */
using EarlyStopAtMinLoss = EarlyStopAtMinLossType<arma::mat>;

} // namespace ens

#endif
